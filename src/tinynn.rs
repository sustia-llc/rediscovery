//! Tier 2: the TinyNN of §B.2.2 and its associative-vs-geometric competition.
//!
//! [`TinyNn`] holds the graph-derived quantities the passes reuse — the target
//! distribution D⁻¹A, the adjacency matrix, the degrees, the vertex pairs at
//! each graph distance, and the spectrum of −L together with its Fiedler-like
//! index set. The model of decision D9 is one wide trainable W ∈ ℝ^{m×m}
//! between a tied embedding/unembedding E ∈ ℝ^{n×m}: the logit of v given u is
//! E[u] W E[v]ᵀ, optionally with a GELU on the hidden state E W. [`run`]
//! drives full-batch gradient descent on the degree-normalized cross-entropy
//! of the bidirectional edge bigrams, recording one [`StepRecord`] per step,
//! streaming it to a CSV, and writing the final node-node cosine matrix;
//! [`Regime`] selects whether E is frozen (§B.2.2's associative setting) or
//! trained alongside W (the geometric one). Each record carries two
//! independent embedding measurements: [`TinyNn::fiedler_alignment`], which
//! reads the §4.1 spectral geometry, and
//! [`TinyNn::deepest_shell_separation`], which reads the deepest shell of the
//! distance-shell cosine profile of Fig. 23. [`run`] takes an explicit output
//! path and seed (decision D8, `docs/2510.26745v2-poc-analysis.md` §8).
//!
//! Four [`Params`] knobs move a run between the Node2Vec-equivalent corner and
//! the committed regime: [`WeightInit`] picks W's initialization,
//! [`Params::weight_rate_ratio`] scales W's rate against E's,
//! [`Optimizer`] picks constant-rate descent or the decoupled AdamW of §B.3
//! under [`scheduled_rate`], and [`Params::alignment_stop`] ends a run at the
//! first step whose Fiedler alignment reaches a threshold.

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — E[u] W E[v]ᵀ, W_ij, ‖ΔW‖_F — that the lint reads as unbackticked identifiers"
)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

use nalgebra::DMatrix;

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::node2vec::{DEGENERACY_TOLERANCE, Outcome, cosine_similarity, second_factor_seed};
use crate::numerics::{gaussian_matrix, row_softmax, weighted_log_likelihood};
use crate::output::write_matrix_csv;
use crate::spectral::{Spectrum, symmetrize, transition};

/// Number of distinct graph distances [`TinyNn::new`] requires a vertex pair
/// at, below which it returns [`Error::InsufficientDistanceShells`].
pub const MINIMUM_DISTANCE_SHELLS: usize = 2;

/// Fiedler alignment at or above which an embedding counts as geometric for
/// [`Run::alignment_step`].
///
/// Measured on the four D-graphs by
/// `the_fiedler_alignment_calibration_separates_the_references`: the
/// Fiedler-like eigenvectors of −L score 1.000000 and a Tier-1 `Node2Vec`
/// embedding at D7 defaults and seed 20260829 scores 0.980406–1.000000, while
/// a rank-1 Fiedler-sign embedding scores 0.288786–0.408585, an
/// all-rows-identical embedding 0.000000–0.031149, and Gaussian draws at seeds
/// 0..200 peak at 0.380369–0.491188.
pub const FIEDLER_ALIGNMENT: f64 = 0.75;

/// Fraction of ‖E‖_F² below which a principal direction of the deflated
/// embedding counts as absent, contributing nothing to
/// [`TinyNn::fiedler_alignment`].
const PRINCIPAL_DIRECTION_FLOOR: f64 = 1e-12;

/// Slack below 1 within which [`Run::associative_step`] reads the associative
/// score as its maximum, absorbing the rounding of a mean of exact fractions.
pub const FULL_MEMORIZATION_SLACK: f64 = 1e-12;

/// Hidden-layer activation of the TinyNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// The hidden state is E W, making the logits the bilinear form E W Eᵀ.
    Linear,
    /// The hidden state is GELU(E W), in the tanh approximation
    /// 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))).
    Gelu,
}

/// Cubic coefficient of the GELU tanh approximation.
const GELU_CUBIC: f64 = 0.044_715;

impl Activation {
    /// Applies the activation entrywise to a pre-activation matrix.
    fn apply(self, pre: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::Linear => pre.clone(),
            Self::Gelu => pre.map(gelu),
        }
    }

    /// The entrywise derivative at a pre-activation matrix, or `None` where it
    /// is the all-ones matrix.
    fn derivative(self, pre: &DMatrix<f64>) -> Option<DMatrix<f64>> {
        match self {
            Self::Linear => None,
            Self::Gelu => Some(pre.map(gelu_derivative)),
        }
    }
}

/// √(2/π), the scale of the GELU tanh approximation's argument.
fn gelu_scale() -> f64 {
    std::f64::consts::FRAC_2_PI.sqrt()
}

/// The tanh argument of the GELU approximation, √(2/π)(x + 0.044715x³).
fn gelu_inner(x: f64) -> f64 {
    gelu_scale() * (x + GELU_CUBIC * x * x * x)
}

/// GELU in the tanh approximation.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + gelu_inner(x).tanh())
}

/// The derivative of [`gelu`].
fn gelu_derivative(x: f64) -> f64 {
    let tangent = gelu_inner(x).tanh();
    let inner_derivative = gelu_scale() * (1.0 + 3.0 * GELU_CUBIC * x * x);
    0.5 * (1.0 + tangent + x * (1.0 - tangent * tangent) * inner_derivative)
}

/// Which parameter blocks gradient descent updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Only W is updated; E stays at its draw — §B.2.2's associative setting.
    FrozenEmbedding,
    /// Both E and W are updated — the geometric setting of Figs. 8 and 22.
    LearnableEmbedding,
}

/// How [`TinyNn::initial_parameters`] builds W.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightInit {
    /// W is drawn from N(0, [`Params::weight_sigma`]²) over the stream keyed
    /// by [`second_factor_seed`] of the run seed.
    Gaussian,
    /// W is the m×m identity, the setting under which a linear hidden layer
    /// makes the logits Tier 1's V Vᵀ and −∂L/∂E Lemma 6's ascent direction
    /// (`the_identity_weight_reproduces_the_tier1_ascent_direction`).
    Identity,
}

/// The decoupled-AdamW knobs of §B.3.
///
/// The paper states the weight decay 0.01 and a cosine schedule with warm-up;
/// [`AdamW::default`] carries that decay together with PyTorch's β₁ = 0.9,
/// β₂ = 0.999 and ε = 1e-8, which §B.3 does not state, and a warm-up over the
/// first 5 % of the step budget, whose length §B.3 does not state either.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamW {
    /// First-moment decay β₁.
    pub beta1: f64,
    /// Second-moment decay β₂.
    pub beta2: f64,
    /// Denominator floor ε.
    pub epsilon: f64,
    /// Decoupled weight decay, applied to the stepped parameter rather than
    /// through the gradient.
    pub weight_decay: f64,
    /// Fraction of [`Params::max_steps`] the linear warm-up spans.
    pub warmup_fraction: f64,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            warmup_fraction: 0.05,
        }
    }
}

/// How a run turns a gradient into the movement it subtracts from a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// p ← p − η g at the constant η of [`Params::learning_rate`], the reading
    /// the captions of Figs. 7, 8 and 22 state.
    GradientDescent,
    /// Decoupled AdamW at the rate [`scheduled_rate`] gives, the reading §B.3
    /// states.
    AdamW(AdamW),
}

/// Run parameters for gradient descent on the TinyNN cross-entropy.
///
/// [`Params::default`] carries decision D10's associative setting: the
/// §B.2.2 width m = 512, η = 0.1, a frozen embedding, and a linear hidden
/// layer, with the Gaussian W draw, a weight rate equal to the embedding's,
/// constant-rate descent, and no geometry stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Hidden width m, the column count of E and the order of W.
    pub width: usize,
    /// Standard deviation of the N(0, σ²) entries of E.
    pub embedding_sigma: f64,
    /// Standard deviation of the N(0, σ²) entries of W.
    pub weight_sigma: f64,
    /// Descent step size η.
    pub learning_rate: f64,
    /// Upper bound on applied updates.
    pub max_steps: usize,
    /// Value of the relative update at or below which the run stops as
    /// converged.
    pub tolerance: f64,
    /// Hidden-layer activation.
    pub activation: Activation,
    /// Which parameter blocks the descent updates.
    pub regime: Regime,
    /// How [`TinyNn::initial_parameters`] builds W.
    pub weight_init: WeightInit,
    /// The relative rate ρ = η_W/η_E: E moves at the run's rate and W at ρ
    /// times it, so ρ = 0 holds W at its initialization for the whole run.
    pub weight_rate_ratio: f64,
    /// How a gradient becomes the movement subtracted from a block.
    pub optimizer: Optimizer,
    /// Fiedler alignment at or above which [`run`] ends with
    /// [`StopReason::Aligned`], checked on each recorded step before the
    /// convergence and budget checks. `None` leaves the other stopping rules
    /// — convergence, the step budget, and the caller's `should_stop` — to
    /// end the run.
    pub alignment_stop: Option<f64>,
}

impl Params {
    /// Associative-setting parameters at hidden width `width`: the embedding
    /// drawn from N(0, 1/width), W from N(0, 1/width²), η = 0.1, a frozen
    /// embedding, and a linear hidden layer. The paper states no initializer.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "a width above 2^53 is unallocatable long before the conversion rounds"
    )]
    pub fn for_width(width: usize) -> Self {
        let scale = 1.0 / (width as f64);
        Self {
            width,
            embedding_sigma: scale.sqrt(),
            weight_sigma: scale,
            learning_rate: 0.1,
            max_steps: 2_000,
            tolerance: 1e-10,
            activation: Activation::Linear,
            regime: Regime::FrozenEmbedding,
            weight_init: WeightInit::Gaussian,
            weight_rate_ratio: 1.0,
            optimizer: Optimizer::GradientDescent,
            alignment_stop: None,
        }
    }

    /// Rejects a zero width or step budget, a non-positive or non-finite σ, η,
    /// or tolerance, a negative or non-finite weight rate ratio, and
    /// out-of-range AdamW knobs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimension`] for `width == 0`,
    /// [`Error::ZeroMaxSteps`] for `max_steps == 0`,
    /// [`Error::InvalidRunParameter`] naming the first of `embedding_sigma`,
    /// `weight_sigma`, `learning_rate`, `tolerance`, `epsilon` that is not
    /// positive and finite, [`Error::NegativeRunParameter`] naming the first
    /// of `weight_rate_ratio`, `weight_decay` that is not non-negative and
    /// finite, and [`Error::RunParameterNotAFraction`] naming the first of
    /// `beta1`, `beta2`, `warmup_fraction` outside [0, 1) — or a
    /// `Some` `alignment_stop` outside [0, 1], the range
    /// [`TinyNn::fiedler_alignment`] takes values in.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 {
            return Err(Error::InvalidDimension {
                dimension: self.width,
            });
        }
        if self.max_steps == 0 {
            return Err(Error::ZeroMaxSteps);
        }
        let mut positive = vec![
            ("embedding_sigma", self.embedding_sigma),
            ("weight_sigma", self.weight_sigma),
            ("learning_rate", self.learning_rate),
            ("tolerance", self.tolerance),
        ];
        let mut non_negative = vec![("weight_rate_ratio", self.weight_rate_ratio)];
        let mut fractions = Vec::new();
        if let Optimizer::AdamW(settings) = self.optimizer {
            positive.push(("epsilon", settings.epsilon));
            non_negative.push(("weight_decay", settings.weight_decay));
            fractions.extend([
                ("beta1", settings.beta1),
                ("beta2", settings.beta2),
                ("warmup_fraction", settings.warmup_fraction),
            ]);
        }

        for (parameter, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidRunParameter { parameter, value });
            }
        }
        for (parameter, value) in non_negative {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Error::NegativeRunParameter { parameter, value });
            }
        }
        for (parameter, value) in fractions {
            // A NaN or infinite value is outside the range, so this covers the
            // non-finite cases the two loops above test for explicitly.
            if !(0.0..1.0).contains(&value) {
                return Err(Error::RunParameterNotAFraction { parameter, value });
            }
        }
        if let Some(value) = self.alignment_stop
            && !(0.0..=1.0).contains(&value)
        {
            return Err(Error::RunParameterNotAFraction {
                parameter: "alignment_stop",
                value,
            });
        }
        Ok(())
    }
}

impl Default for Params {
    fn default() -> Self {
        Self::for_width(512)
    }
}

/// The model state: the tied embedding/unembedding E and the wide matrix W.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    embedding: DMatrix<f64>,
    weight: DMatrix<f64>,
}

impl Parameters {
    /// Pairs an n×m embedding with an m×m weight matrix.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WeightShapeMismatch`] when `weight` is not square of
    /// order `embedding.ncols()`.
    pub fn new(embedding: DMatrix<f64>, weight: DMatrix<f64>) -> Result<Self> {
        let width = embedding.ncols();
        if weight.nrows() != width || weight.ncols() != width {
            return Err(Error::WeightShapeMismatch {
                rows: weight.nrows(),
                columns: weight.ncols(),
                width,
            });
        }
        Ok(Self { embedding, weight })
    }

    /// The tied embedding/unembedding E, row u being vertex u's embedding.
    #[must_use]
    pub fn embedding(&self) -> &DMatrix<f64> {
        &self.embedding
    }

    /// The wide trainable matrix W.
    #[must_use]
    pub fn weight(&self) -> &DMatrix<f64> {
        &self.weight
    }

    /// The hidden width m.
    #[must_use]
    pub fn width(&self) -> usize {
        self.embedding.ncols()
    }
}

/// The loss gradient in each trainable block.
#[derive(Debug, Clone)]
pub struct Gradients {
    weight: DMatrix<f64>,
    embedding: Option<DMatrix<f64>>,
}

impl Gradients {
    /// ∂L/∂W.
    #[must_use]
    pub fn weight(&self) -> &DMatrix<f64> {
        &self.weight
    }

    /// ∂L/∂E, present under [`Regime::LearnableEmbedding`].
    #[must_use]
    pub fn embedding(&self) -> Option<&DMatrix<f64>> {
        self.embedding.as_ref()
    }
}

/// The pass whose intermediates the loss, the metrics, and the backward pass
/// all read.
struct Forward {
    pre: DMatrix<f64>,
    hidden: DMatrix<f64>,
    logits: DMatrix<f64>,
    probabilities: DMatrix<f64>,
}

/// The TinyNN system for one graph.
///
/// Construction computes the target distribution D⁻¹A, the degrees, the
/// vertex pairs at each graph distance, and the spectrum of −L with its
/// Fiedler-like index range once; every forward, backward, and metric
/// evaluation reuses them.
#[derive(Debug, Clone)]
pub struct TinyNn {
    order: usize,
    walk: DMatrix<f64>,
    adjacency: DMatrix<f64>,
    degrees: Vec<usize>,
    shells: Vec<Vec<(usize, usize)>>,
    spectrum: Spectrum,
    fiedler: Range<usize>,
}

impl TinyNn {
    /// Builds the system for `graph`.
    ///
    /// # Errors
    ///
    /// Propagates [`transition`]'s [`Error::IsolatedVertex`] and
    /// [`Spectrum::of_negative_laplacian`]'s errors, and returns
    /// [`Error::InsufficientDistanceShells`] when `graph` has vertex pairs at
    /// fewer than [`MINIMUM_DISTANCE_SHELLS`] distinct distances.
    pub fn new(graph: &Graph) -> Result<Self> {
        let walk = transition(graph)?;
        let order = graph.order();
        let adjacency = graph.adjacency().clone();
        let degrees = graph
            .degrees()
            .iter()
            .map(|degree| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "`Graph` builds degrees as row sums of a {0, 1} adjacency matrix, so each is an exact non-negative integer in f64"
                )]
                let count = degree.round() as usize;
                count
            })
            .collect();
        let profile = distance_profile(&adjacency, order)?;
        let spectrum = Spectrum::of_negative_laplacian(graph)?;
        let fiedler = fiedler_like_set(&spectrum, profile.components);
        Ok(Self {
            order,
            walk,
            adjacency,
            degrees,
            shells: profile.shells,
            spectrum,
            fiedler,
        })
    }

    /// The vertex count n, equal to the row count of a valid embedding.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// The row-normalized target distribution D⁻¹A: row u is uniform over the
    /// neighbours of u, the degree-normalized form Tier 1's objective uses.
    #[must_use]
    pub fn walk(&self) -> &DMatrix<f64> {
        &self.walk
    }

    /// The spectrum of −L the geometry measurements read.
    #[must_use]
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }

    /// The Fiedler-like eigenvector index range of −L, from
    /// [`fiedler_like_set`].
    #[must_use]
    pub fn fiedler_like(&self) -> Range<usize> {
        self.fiedler.clone()
    }

    /// The eigenvector index range of −L that [`TinyNn::fiedler_alignment`]
    /// projects out of an embedding before measuring it: everything above the
    /// Fiedler-like set.
    #[must_use]
    pub fn trivial_block(&self) -> Range<usize> {
        0..self.fiedler.start
    }

    /// The number of populated distance shells, the largest graph distance
    /// between two vertices of one component.
    #[must_use]
    pub fn shell_count(&self) -> usize {
        self.shells.len()
    }

    /// The unordered vertex pairs at graph distance `distance`, for
    /// `distance` in `1..=`[`TinyNn::shell_count`]; an empty slice outside
    /// that range.
    #[must_use]
    pub fn shell(&self, distance: usize) -> &[(usize, usize)] {
        self.shells
            .get(distance.wrapping_sub(1))
            .map_or(&[], Vec::as_slice)
    }

    /// Draws E from `seed` with `params`-scaled standard normals and builds W
    /// as [`Params::weight_init`] asks: the same draw from
    /// [`second_factor_seed`] of `seed` under [`WeightInit::Gaussian`], the
    /// identity under [`WeightInit::Identity`].
    ///
    /// # Errors
    ///
    /// Propagates [`Params::validate`]'s errors and
    /// [`gaussian_matrix`]'s [`Error::MatrixTooLarge`].
    pub fn initial_parameters(&self, params: &Params, seed: u64) -> Result<Parameters> {
        params.validate()?;
        let embedding = gaussian_matrix(self.order, params.width, params.embedding_sigma, seed)?;
        let weight = match params.weight_init {
            WeightInit::Gaussian => gaussian_matrix(
                params.width,
                params.width,
                params.weight_sigma,
                second_factor_seed(seed),
            )?,
            WeightInit::Identity => DMatrix::identity(params.width, params.width),
        };
        Parameters::new(embedding, weight)
    }

    /// Runs the forward pass, keeping the intermediates the backward pass
    /// reads.
    fn forward(&self, parameters: &Parameters, activation: Activation) -> Result<Forward> {
        self.check_order(parameters)?;
        let pre = &parameters.embedding * &parameters.weight;
        let hidden = activation.apply(&pre);
        let logits = &hidden * parameters.embedding.transpose();
        let probabilities = row_softmax(&logits);
        Ok(Forward {
            pre,
            hidden,
            logits,
            probabilities,
        })
    }

    /// The logit matrix, entry (u, v) being the logit of v given u.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `parameters` does not
    /// carry one embedding row per vertex.
    pub fn logits(&self, parameters: &Parameters, activation: Activation) -> Result<DMatrix<f64>> {
        Ok(self.forward(parameters, activation)?.logits)
    }

    /// The next-token probability matrix, row u being the softmax of u's
    /// logits over all vertices, the self term included.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `parameters` does not
    /// carry one embedding row per vertex.
    pub fn probabilities(
        &self,
        parameters: &Parameters,
        activation: Activation,
    ) -> Result<DMatrix<f64>> {
        Ok(self.forward(parameters, activation)?.probabilities)
    }

    /// The full-batch cross-entropy
    /// L = −Σ_u Σ_v (D⁻¹A)_uv log p(v | u) over the bidirectional edge
    /// bigrams.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `parameters` does not
    /// carry one embedding row per vertex.
    pub fn loss(&self, parameters: &Parameters, activation: Activation) -> Result<f64> {
        Ok(self.loss_of(&self.forward(parameters, activation)?))
    }

    /// The cross-entropy of a completed forward pass.
    fn loss_of(&self, forward: &Forward) -> f64 {
        -weighted_log_likelihood(&self.walk, &forward.logits)
    }

    /// The loss gradient in each block `regime` trains.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `parameters` does not
    /// carry one embedding row per vertex.
    pub fn gradients(
        &self,
        parameters: &Parameters,
        activation: Activation,
        regime: Regime,
    ) -> Result<Gradients> {
        let forward = self.forward(parameters, activation)?;
        Ok(self.gradients_of(parameters, activation, regime, &forward))
    }

    /// The loss gradient of a completed forward pass.
    ///
    /// With G = P − D⁻¹A the derivative in the logits, the hidden gradient is
    /// G E, the pre-activation gradient that times the activation derivative,
    /// ∂L/∂W is Eᵀ times the pre-activation gradient, and ∂L/∂E is
    /// Gᵀ H + (∂L/∂pre) Wᵀ.
    fn gradients_of(
        &self,
        parameters: &Parameters,
        activation: Activation,
        regime: Regime,
        forward: &Forward,
    ) -> Gradients {
        let residual = &forward.probabilities - &self.walk;
        let mut pre_gradient = &residual * &parameters.embedding;
        if let Some(derivative) = activation.derivative(&forward.pre) {
            pre_gradient.component_mul_assign(&derivative);
        }

        let weight = parameters.embedding.transpose() * &pre_gradient;
        let embedding = match regime {
            Regime::FrozenEmbedding => None,
            Regime::LearnableEmbedding => Some(
                residual.transpose() * &forward.hidden
                    + &pre_gradient * parameters.weight.transpose(),
            ),
        };
        Gradients { weight, embedding }
    }

    /// Figure 7's associative metric (md 298): the fraction of each vertex's
    /// d(u) neighbours that appear among its top-d(u) next-token
    /// probabilities, averaged over vertices.
    ///
    /// Ties are broken by ascending vertex index, so the value is a function
    /// of `probabilities` alone. It is 1 exactly when every vertex ranks all
    /// of its neighbours above every non-neighbour.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "vertex counts and degrees are small integers, exact in f64"
    )]
    pub fn associative_score(&self, probabilities: &DMatrix<f64>) -> f64 {
        // One buffer for the whole sweep: the comparator is a total order, so
        // each sort yields the same permutation whatever the buffer held.
        let mut ranked: Vec<usize> = (0..self.order).collect();
        let mut total = 0.0;
        for u in 0..self.order {
            ranked.sort_by(|&a, &b| {
                probabilities[(u, b)]
                    .total_cmp(&probabilities[(u, a)])
                    .then(a.cmp(&b))
            });
            // `TinyNn::new` builds `walk` through `transition`, which rejects
            // a degree-zero vertex, so this degree is at least one.
            let degree = self.degrees[u];
            let hits = ranked[..degree]
                .iter()
                .filter(|&&v| self.adjacency[(u, v)] > 0.0)
                .count();
            total += hits as f64 / degree as f64;
        }
        total / self.order as f64
    }

    /// The mean of `cosines` over the vertex pairs at each distance
    /// `1..=`[`TinyNn::shell_count`], in that order.
    ///
    /// # Panics
    ///
    /// Panics if a shell is empty. [`TinyNn::new`] returns
    /// [`Error::InsufficientDistanceShells`] rather than build a system with
    /// one, so a `TinyNn` this crate constructed has none.
    #[allow(
        clippy::cast_precision_loss,
        reason = "pair counts are bounded by order², exact in f64 at this scale"
    )]
    fn shell_means_of(&self, cosines: &DMatrix<f64>) -> Vec<f64> {
        self.shells
            .iter()
            .map(|pairs| {
                assert!(
                    !pairs.is_empty(),
                    "invariant: `TinyNn::new` rejects a graph with an empty distance shell"
                );
                let total: f64 = pairs.iter().map(|&(u, v)| cosines[(u, v)]).sum();
                total / pairs.len() as f64
            })
            .collect()
    }

    /// The mean cosine similarity between rows of `embedding` over the vertex
    /// pairs at each distance `1..=`[`TinyNn::shell_count`], in that order.
    /// Every entry is NaN when `embedding` carries a non-finite entry.
    ///
    /// # Panics
    ///
    /// Panics if a shell is empty. [`TinyNn::new`] returns
    /// [`Error::InsufficientDistanceShells`] rather than build a system with
    /// one, so a `TinyNn` this crate constructed has none.
    #[must_use]
    pub fn shell_means(&self, embedding: &DMatrix<f64>) -> Vec<f64> {
        if embedding.iter().any(|entry| !entry.is_finite()) {
            return vec![f64::NAN; self.shells.len()];
        }
        self.shell_means_of(&cosine_similarity(embedding))
    }

    /// The deepest distance shell's mean cosine measured by its distance from
    /// zero, capped by its drop below the shell above it. NaN for an embedding
    /// carrying a non-finite entry. On the four D-graphs an embedding whose
    /// rows are the adjacency rows scores zero
    /// (`an_adjacency_row_embedding_scores_zero_on_the_deepest_shell`).
    #[must_use]
    pub fn deepest_shell_separation(&self, embedding: &DMatrix<f64>) -> f64 {
        profile_separation(&self.shell_means(embedding))
    }

    /// The fraction of `embedding`'s leading principal directions that lie in
    /// the Fiedler-like eigenspace of −L, the §4.1 spectral geometry.
    ///
    /// [`TinyNn::trivial_block`]'s eigenvectors — one per connected component,
    /// the degenerate directions Remark 5 tracks — are projected out of the
    /// embedding first. Of the remainder's principal directions, the
    /// eigenvectors of its Gram matrix in descending order, the leading k are
    /// taken, k being the width of [`TinyNn::fiedler_like`]; each contributes
    /// its squared projection onto that eigenspace, or nothing when its share
    /// of ‖E‖_F² falls below `PRINCIPAL_DIRECTION_FLOOR`, and the total is
    /// divided by k. Each term is at most 1, so the value lies in [0, 1]; it
    /// is unchanged by scaling `embedding` over the range
    /// `the_fiedler_alignment_is_scale_invariant` measures.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `embedding` does not
    /// carry one row per vertex, and propagates [`Spectrum::new`]'s
    /// [`Error::NonFinite`] when the embedding's Gram matrix leaves the finite
    /// range.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the Fiedler-like range is bounded by the vertex count, exact in f64 at this scale"
    )]
    pub fn fiedler_alignment(&self, embedding: &DMatrix<f64>) -> Result<f64> {
        self.check_rows(embedding)?;
        let trivial = self
            .spectrum
            .eigenvectors()
            .columns(0, self.fiedler.start)
            .into_owned();
        let deflated = embedding - &trivial * (trivial.transpose() * embedding);
        let principal = Spectrum::new(symmetrize(&(&deflated * deflated.transpose())))?;

        let fiedler = self
            .spectrum
            .eigenvectors()
            .columns(self.fiedler.start, self.fiedler.len())
            .into_owned();
        let projected = fiedler.transpose() * principal.eigenvectors();
        // `symmetrize` doubles the Gram, so eigenvalue j is twice the squared
        // singular value the floor is stated against.
        let floor = 2.0 * PRINCIPAL_DIRECTION_FLOOR * embedding.norm_squared();
        let carried: f64 = (0..self.fiedler.len())
            .filter(|&j| principal.eigenvalues()[j] > floor)
            .map(|j| projected.column(j).norm_squared())
            .sum();
        Ok(carried / self.fiedler.len() as f64)
    }

    /// Rejects a parameter set whose embedding row count is not the vertex
    /// count.
    fn check_order(&self, parameters: &Parameters) -> Result<()> {
        self.check_rows(&parameters.embedding)
    }

    /// Rejects an embedding whose row count is not the vertex count.
    fn check_rows(&self, embedding: &DMatrix<f64>) -> Result<()> {
        if embedding.nrows() == self.order {
            Ok(())
        } else {
            Err(Error::EmbeddingOrderMismatch {
                rows: embedding.nrows(),
                order: self.order,
            })
        }
    }
}

/// The separation of a shell-mean profile: its last entry's distance from
/// zero, capped by that entry's drop below the one above it. NaN for a profile
/// shorter than two entries or carrying a non-finite one, so such a profile
/// fails every threshold comparison.
fn profile_separation(shell_means: &[f64]) -> f64 {
    let [.., above, deepest] = shell_means else {
        return f64::NAN;
    };
    let separation = above - deepest;
    if deepest.is_finite() && separation.is_finite() {
        deepest.abs().min(separation)
    } else {
        f64::NAN
    }
}

/// What one all-pairs breadth-first sweep of a graph yields.
struct DistanceProfile {
    /// The unordered vertex pairs at each distance `1..=d`, `d` being the
    /// largest graph distance between two vertices of one component.
    shells: Vec<Vec<(usize, usize)>>,
    /// The number of connected components.
    components: usize,
}

/// The distance shells and connected-component count of `adjacency`, by
/// breadth-first search from every vertex.
///
/// # Errors
///
/// Returns [`Error::InsufficientDistanceShells`] when the shell count is below
/// [`MINIMUM_DISTANCE_SHELLS`], reporting it.
fn distance_profile(adjacency: &DMatrix<f64>, order: usize) -> Result<DistanceProfile> {
    let mut shells: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut components = 0_usize;
    let mut visited = vec![false; order];
    let mut distance = vec![usize::MAX; order];
    let mut queue = VecDeque::with_capacity(order);

    for source in 0..order {
        if !visited[source] {
            components += 1;
        }
        distance.fill(usize::MAX);
        distance[source] = 0;
        queue.clear();
        queue.push_back(source);
        while let Some(vertex) = queue.pop_front() {
            let next = distance[vertex] + 1;
            for other in 0..order {
                if adjacency[(vertex, other)] > 0.0 && distance[other] == usize::MAX {
                    distance[other] = next;
                    queue.push_back(other);
                }
            }
        }
        for (other, &reached) in distance.iter().enumerate() {
            if reached == usize::MAX {
                continue;
            }
            visited[other] = true;
            if other <= source {
                continue;
            }
            if shells.len() < reached {
                shells.resize(reached, Vec::new());
            }
            shells[reached - 1].push((source, other));
        }
    }

    if shells.len() < MINIMUM_DISTANCE_SHELLS {
        return Err(Error::InsufficientDistanceShells {
            available: shells.len(),
        });
    }
    Ok(DistanceProfile { shells, components })
}

/// The Fiedler-like eigenvector indices of `spectrum` for a graph with
/// `components` connected components: the `components` indices below the
/// leading `components` of the spectrum, extended forward while the next
/// eigenvalue is within [`DEGENERACY_TOLERANCE`] of the last one taken. The
/// returned range is non-empty.
///
/// On the three connected D-graphs it is
/// [`crate::node2vec::fiedler_like_range`] at
/// [`crate::node2vec::fiedler_spread`]
/// (`the_fiedler_like_set_agrees_with_tier1_on_a_connected_graph`).
fn fiedler_like_set(spectrum: &Spectrum, components: usize) -> Range<usize> {
    let order = spectrum.order();
    let start = components.min(order.saturating_sub(1));
    let mut end = (start + components).clamp(start + 1, order);
    while end < order
        && (spectrum.eigenvalues()[end - 1] - spectrum.eigenvalues()[end]).abs()
            <= DEGENERACY_TOLERANCE
    {
        end += 1;
    }
    start..end
}

/// The number of warm-up steps `fraction` of `budget` gives: the rounded
/// product, held to at least one step and at most the budget.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the float-to-usize cast saturates on any fraction, finite or not, and the clamp below bounds the result to [1, max(budget, 1)]"
)]
fn warmup_steps(budget: usize, fraction: f64) -> usize {
    let raw = (budget as f64 * fraction).round() as usize;
    raw.clamp(1, budget.max(1))
}

/// The §B.3 rate at `step` of a `budget`-step run: a linear warm-up from 0 to
/// `peak` over the first `fraction` of the budget, then a cosine decay from
/// `peak` to 0 over the remainder.
///
/// It is 0 at step 0 and `peak` at the warm-up count, the first cosine step.
/// When the cosine phase is non-empty it is half `peak` halfway through that
/// phase, 0 at `budget` — the endpoint the run approaches, one past its last
/// applied update — and 0 at every step beyond. When the warm-up consumes the
/// whole budget the cosine phase is empty and every step at or past the
/// warm-up count returns `peak`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "step counts and budgets here are far below 2^53 and exact in f64"
)]
pub fn scheduled_rate(step: usize, budget: usize, peak: f64, fraction: f64) -> f64 {
    let warmup = warmup_steps(budget, fraction);
    if step < warmup {
        return peak * (step as f64) / (warmup as f64);
    }
    let remaining = budget.saturating_sub(warmup);
    if remaining == 0 {
        return peak;
    }
    let progress = (((step - warmup) as f64) / (remaining as f64)).min(1.0);
    peak * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// One block's decoupled-AdamW moment state.
#[derive(Debug, Clone)]
struct Moments {
    first: DMatrix<f64>,
    second: DMatrix<f64>,
    updates: i32,
}

impl Moments {
    /// Zero moments shaped like the block they track.
    fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            first: DMatrix::zeros(rows, columns),
            second: DMatrix::zeros(rows, columns),
            updates: 0,
        }
    }

    /// Advances the moments by `gradient` and returns the movement to subtract
    /// from `parameter`: the bias-corrected step η m̂/(√v̂ + ε) followed by the
    /// decoupled decay of the stepped value by 1 − η·wd, as one difference.
    ///
    /// The decay reaches the parameter without passing through `gradient`, so
    /// the moments track the loss gradient alone.
    fn advance(
        &mut self,
        parameter: &DMatrix<f64>,
        gradient: &DMatrix<f64>,
        rate: f64,
        settings: AdamW,
    ) -> DMatrix<f64> {
        self.updates = self.updates.saturating_add(1);
        self.first *= settings.beta1;
        self.first += gradient * (1.0 - settings.beta1);
        self.second *= settings.beta2;
        self.second += gradient.map(|entry| entry * entry) * (1.0 - settings.beta2);

        let first_correction = 1.0 - settings.beta1.powi(self.updates);
        let second_correction = 1.0 - settings.beta2.powi(self.updates);
        let step = DMatrix::from_fn(parameter.nrows(), parameter.ncols(), |i, j| {
            let first = self.first[(i, j)] / first_correction;
            let second = self.second[(i, j)] / second_correction;
            rate * first / (second.sqrt() + settings.epsilon)
        });

        let decayed = (parameter - step) * (1.0 - rate * settings.weight_decay);
        parameter - decayed
    }
}

/// The movement one step subtracts from each block a run trains, the embedding
/// term being absent under [`Regime::FrozenEmbedding`].
struct Deltas {
    weight: DMatrix<f64>,
    embedding: Option<DMatrix<f64>>,
}

/// The optimizer state a run carries between steps.
#[allow(
    clippy::large_enum_variant,
    reason = "one `Updater` lives on the stack per run; the 216 bytes are four matrix headers and the knobs, the moment entries themselves already being behind the matrices' own allocations"
)]
enum Updater {
    /// Constant-rate descent, which carries none.
    Descent,
    /// Decoupled AdamW, which carries one moment pair per block.
    Adam {
        settings: AdamW,
        weight: Moments,
        embedding: Moments,
    },
}

impl Updater {
    /// The state [`Params::optimizer`] asks for, shaped for `parameters`.
    fn new(params: &Params, parameters: &Parameters) -> Self {
        match params.optimizer {
            Optimizer::GradientDescent => Self::Descent,
            Optimizer::AdamW(settings) => Self::Adam {
                settings,
                weight: Moments::zeros(parameters.width(), parameters.width()),
                embedding: Moments::zeros(
                    parameters.embedding.nrows(),
                    parameters.embedding.ncols(),
                ),
            },
        }
    }

    /// The movement `step` subtracts from each trained block, beside the rate
    /// the embedding block moved at.
    ///
    /// W moves at [`Params::weight_rate_ratio`] times that rate, so a ratio of
    /// 0 leaves W where it was under either optimizer. The AdamW arm advances
    /// its moments here; a run that ends without applying the returned movement
    /// discards them with the rest of the state.
    fn propose(
        &mut self,
        parameters: &Parameters,
        gradients: &Gradients,
        params: &Params,
        step: usize,
    ) -> (Deltas, f64) {
        match self {
            Self::Descent => {
                let rate = params.learning_rate;
                let deltas = Deltas {
                    weight: gradients.weight() * (rate * params.weight_rate_ratio),
                    embedding: gradients.embedding().map(|gradient| gradient * rate),
                };
                (deltas, rate)
            }
            Self::Adam {
                settings,
                weight,
                embedding,
            } => {
                let settings = *settings;
                let rate = scheduled_rate(
                    step,
                    params.max_steps,
                    params.learning_rate,
                    settings.warmup_fraction,
                );
                let deltas = Deltas {
                    weight: weight.advance(
                        &parameters.weight,
                        gradients.weight(),
                        rate * params.weight_rate_ratio,
                        settings,
                    ),
                    embedding: gradients.embedding().map(|gradient| {
                        embedding.advance(&parameters.embedding, gradient, rate, settings)
                    }),
                };
                (deltas, rate)
            }
        }
    }
}

/// Why a [`run`] step loop ended, separating the geometry stop of
/// [`Params::alignment_stop`] from the `should_stop` one that
/// [`Run::outcome`] merges it with under [`Outcome::Stopped`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The relative update fell to or below [`Params::tolerance`] at a step
    /// whose rate was above zero.
    Converged,
    /// [`Params::max_steps`] updates were applied.
    StepLimit,
    /// A recorded step's Fiedler alignment reached [`Params::alignment_stop`].
    Aligned,
    /// The `should_stop` predicate returned `true`.
    Stopped,
}

/// One recorded step: the state after `step` applied updates, together with
/// the relative size of the update pending from it.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    step: usize,
    loss: f64,
    associative_score: f64,
    fiedler_alignment: f64,
    deepest_shell_separation: f64,
    relative_update: f64,
    shell_means: Vec<f64>,
}

impl StepRecord {
    /// The number of updates applied before this state.
    #[must_use]
    pub fn step(&self) -> usize {
        self.step
    }

    /// The cross-entropy at this state.
    #[must_use]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Figure 7's associative metric at this state.
    #[must_use]
    pub fn associative_score(&self) -> f64 {
        self.associative_score
    }

    /// [`TinyNn::fiedler_alignment`] at this state.
    #[must_use]
    pub fn fiedler_alignment(&self) -> f64 {
        self.fiedler_alignment
    }

    /// [`TinyNn::deepest_shell_separation`] at this state.
    #[must_use]
    pub fn deepest_shell_separation(&self) -> f64 {
        self.deepest_shell_separation
    }

    /// (‖ΔW‖_F + ‖ΔE‖_F)/(‖W‖_F + ‖E‖_F) for the update pending from this
    /// state, the ΔE term being zero under [`Regime::FrozenEmbedding`].
    #[must_use]
    pub fn relative_update(&self) -> f64 {
        self.relative_update
    }

    /// The mean cosine similarity at each distance `1..=shell_count`.
    #[must_use]
    pub fn shell_means(&self) -> &[f64] {
        &self.shell_means
    }
}

/// The result of a run.
#[derive(Debug, Clone)]
pub struct Run {
    parameters: Parameters,
    records: Vec<StepRecord>,
    stop: StopReason,
    steps: usize,
}

impl Run {
    /// The final parameters.
    #[must_use]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// The recorded steps, in the order written to the CSV.
    #[must_use]
    pub fn records(&self) -> &[StepRecord] {
        &self.records
    }

    /// The last recorded step.
    #[must_use]
    pub fn last(&self) -> Option<&StepRecord> {
        self.records.last()
    }

    /// The number of applied updates.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Why the step loop ended, reading [`StopReason::Aligned`] as
    /// [`Outcome::Stopped`] — both end the loop early through a stop signal
    /// rather than through the budget or the tolerance;
    /// [`Run::stop_reason`] separates the two.
    #[must_use]
    pub fn outcome(&self) -> Outcome {
        match self.stop {
            StopReason::Converged => Outcome::Converged,
            StopReason::StepLimit => Outcome::StepLimit,
            StopReason::Aligned | StopReason::Stopped => Outcome::Stopped,
        }
    }

    /// Why the step loop ended, with the geometry stop named separately from
    /// the `should_stop` one.
    #[must_use]
    pub fn stop_reason(&self) -> StopReason {
        self.stop
    }

    /// The first step whose associative score is within
    /// [`FULL_MEMORIZATION_SLACK`] of 1, the metric's maximum.
    #[must_use]
    pub fn associative_step(&self) -> Option<usize> {
        self.records
            .iter()
            .find(|record| record.associative_score >= 1.0 - FULL_MEMORIZATION_SLACK)
            .map(StepRecord::step)
    }

    /// The largest associative score over the recorded steps.
    #[must_use]
    pub fn peak_associative_score(&self) -> f64 {
        self.records
            .iter()
            .map(StepRecord::associative_score)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The first step whose Fiedler alignment reaches `threshold`.
    #[must_use]
    pub fn alignment_step(&self, threshold: f64) -> Option<usize> {
        self.records
            .iter()
            .find(|record| record.fiedler_alignment >= threshold)
            .map(StepRecord::step)
    }

    /// The largest Fiedler alignment over the recorded steps.
    #[must_use]
    pub fn peak_alignment(&self) -> f64 {
        self.records
            .iter()
            .map(StepRecord::fiedler_alignment)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The largest deepest-shell separation over the recorded steps.
    #[must_use]
    pub fn peak_deepest_shell_separation(&self) -> f64 {
        self.records
            .iter()
            .map(StepRecord::deepest_shell_separation)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// The paths a run writes.
#[derive(Debug, Clone, Copy)]
pub struct Outputs<'a> {
    /// Per-step instrumentation.
    pub history: &'a Path,
    /// The node-node cosine-similarity matrix of the final E.
    pub cosines: &'a Path,
}

/// Runs full-batch training on `graph`, streaming one CSV row per recorded
/// step to `outputs.history` and writing the final node-node cosine-similarity
/// matrix to `outputs.cosines`.
///
/// Each step subtracts the movement [`Params::optimizer`] proposes from every
/// block [`Params::regime`] trains, every block reading off the same
/// pre-update parameters, polling `should_stop` once per applied update. It
/// stops on the geometry criterion of [`Params::alignment_stop`], on
/// convergence (relative update ≤ [`Params::tolerance`] at a step whose rate
/// is above zero), on reaching [`Params::max_steps`], or on `should_stop`
/// returning `true`, checked in that order; the CSV holds a header row
/// followed by one complete row per recorded step in each case.
///
/// # Errors
///
/// Propagates [`TinyNn::new`]'s, [`Params::validate`]'s and
/// [`TinyNn::fiedler_alignment`]'s errors and [`Error::Io`] from creating or
/// writing either file.
pub fn run<S: Fn() -> bool>(
    graph: &Graph,
    params: &Params,
    seed: u64,
    outputs: &Outputs<'_>,
    should_stop: S,
) -> Result<Run> {
    params.validate()?;
    let system = TinyNn::new(graph)?;
    let mut parameters = system.initial_parameters(params, seed)?;
    let mut updater = Updater::new(params, &parameters);

    let mut sink = BufWriter::new(File::create(outputs.history)?);
    write_header(&mut sink, system.shell_count())?;

    let mut records = Vec::new();
    let mut steps = 0_usize;
    let stop = loop {
        let forward = system.forward(&parameters, params.activation)?;
        let gradients =
            system.gradients_of(&parameters, params.activation, params.regime, &forward);
        let (deltas, rate) = updater.propose(&parameters, &gradients, params, steps);

        let moved = deltas.weight.norm() + deltas.embedding.as_ref().map_or(0.0, DMatrix::norm);
        let relative_update = moved / (parameters.weight.norm() + parameters.embedding.norm());
        let shell_means = system.shell_means(&parameters.embedding);

        let record = StepRecord {
            step: steps,
            loss: system.loss_of(&forward),
            associative_score: system.associative_score(&forward.probabilities),
            fiedler_alignment: system.fiedler_alignment(&parameters.embedding)?,
            deepest_shell_separation: profile_separation(&shell_means),
            relative_update,
            shell_means,
        };
        let aligned = params
            .alignment_stop
            .is_some_and(|threshold| record.fiedler_alignment >= threshold);
        write_row(&mut sink, &record)?;
        records.push(record);

        if aligned {
            break StopReason::Aligned;
        }
        // A step the schedule gave a zero rate moves nothing, which is the
        // warm-up doing its job rather than the descent having converged.
        if rate > 0.0 && relative_update <= params.tolerance {
            break StopReason::Converged;
        }
        if steps >= params.max_steps {
            break StopReason::StepLimit;
        }
        // Polled before the update so that on every outcome the returned
        // state is the one the last record and CSV row describe.
        if should_stop() {
            break StopReason::Stopped;
        }

        parameters.weight -= &deltas.weight;
        if let Some(update) = &deltas.embedding {
            parameters.embedding -= update;
        }
        steps += 1;
    };
    sink.flush()?;

    write_matrix_csv(outputs.cosines, &cosine_similarity(&parameters.embedding))?;

    Ok(Run {
        parameters,
        records,
        stop,
        steps,
    })
}

/// Writes the history header: the fixed columns followed by one
/// `shell_mean_k` per distance shell.
fn write_header<W: Write>(sink: &mut W, shell_count: usize) -> Result<()> {
    write!(
        sink,
        "step,loss,associative_score,fiedler_alignment,deepest_shell_separation,relative_update"
    )?;
    for distance in 1..=shell_count {
        write!(sink, ",shell_mean_{distance}")?;
    }
    writeln!(sink)?;
    Ok(())
}

/// Writes one history row, each float in Rust's shortest round-tripping form.
fn write_row<W: Write>(sink: &mut W, record: &StepRecord) -> Result<()> {
    write!(
        sink,
        "{},{},{},{},{},{}",
        record.step,
        record.loss,
        record.associative_score,
        record.fiedler_alignment,
        record.deepest_shell_separation,
        record.relative_update
    )?;
    for value in &record.shell_means {
        write!(sink, ",{value}")?;
    }
    writeln!(sink)?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "test indices and shell counts are small integers, exact in f64"
)]
mod tests {
    use super::*;
    use crate::node2vec::{self, Node2Vec, fiedler_like_range, fiedler_spread};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::prelude::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Worker threads the sweep pools of
    /// [`the_learnable_run_misses_the_geometry_and_peaks_at_distance_two`] and
    /// [`the_frozen_run_memorizes_without_moving_its_embedding`] run, bounding
    /// how many runs of either are in flight at once. Each test builds its own
    /// pool, so the two do not contend over a shared one.
    const SWEEP_THREADS: usize = 6;

    /// A rayon pool of [`SWEEP_THREADS`] workers, scoped to one test.
    fn sweep_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(SWEEP_THREADS)
            .build()
            .expect("a rayon pool of SWEEP_THREADS workers")
    }

    /// A unique path under the system temp directory, removed on drop.
    struct TempPath(PathBuf);

    impl TempPath {
        fn new(label: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos());
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!(
                "rediscovery-tinynn-{label}-{}-{nanos}-{counter}.csv",
                std::process::id()
            );
            Self(std::env::temp_dir().join(name))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Seed shared by the pins.
    const SEED: u64 = 20_260_829;

    /// The four graphs of decisions D1–D4.
    fn d_graphs() -> Vec<(&'static str, Graph)> {
        vec![
            (
                "path_star(4,4)",
                Graph::path_star(4, 4).expect("path_star(4,4)"),
            ),
            ("grid(4,4)", Graph::grid(4, 4).expect("grid(4,4)")),
            ("cycle(15)", Graph::cycle(15).expect("cycle(15)")),
            ("irregular()", Graph::irregular().expect("irregular()")),
        ]
    }

    /// Central finite differences of `evaluate` at `base`, entry by entry.
    fn central_differences<F>(base: &DMatrix<f64>, step: f64, mut evaluate: F) -> DMatrix<f64>
    where
        F: FnMut(&DMatrix<f64>) -> f64,
    {
        let mut gradient = DMatrix::zeros(base.nrows(), base.ncols());
        let mut probe = base.clone();
        for i in 0..base.nrows() {
            for j in 0..base.ncols() {
                let original = base[(i, j)];
                probe[(i, j)] = original + step;
                let forward = evaluate(&probe);
                probe[(i, j)] = original - step;
                let backward = evaluate(&probe);
                probe[(i, j)] = original;
                gradient[(i, j)] = (forward - backward) / (2.0 * step);
            }
        }
        gradient
    }

    /// Central finite differences of `evaluate` at a seeded sample of `count`
    /// distinct entries of `base`, each paired with the (row, column) it was
    /// taken at. The sample is a partial Fisher–Yates shuffle of the entry
    /// positions over a `ChaCha20` stream keyed by `seed`, so a given
    /// `(seed, count, shape)` probes the same positions every run. `count`
    /// above the entry count probes every entry.
    fn sampled_central_differences<F>(
        base: &DMatrix<f64>,
        step: f64,
        count: usize,
        seed: u64,
        mut evaluate: F,
    ) -> Vec<((usize, usize), f64)>
    where
        F: FnMut(&DMatrix<f64>) -> f64,
    {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut positions: Vec<usize> = (0..base.len()).collect();
        let count = count.min(positions.len());
        for taken in 0..count {
            let pick = taken + rng.random_range(0..positions.len() - taken);
            positions.swap(taken, pick);
        }

        let rows = base.nrows();
        let mut probe = base.clone();
        positions[..count]
            .iter()
            .map(|&position| {
                let (i, j) = (position % rows, position / rows);
                let original = base[(i, j)];
                probe[(i, j)] = original + step;
                let forward = evaluate(&probe);
                probe[(i, j)] = original - step;
                let backward = evaluate(&probe);
                probe[(i, j)] = original;
                ((i, j), (forward - backward) / (2.0 * step))
            })
            .collect()
    }

    /// Widths and draw scales the finite-difference pins sample, chosen so the
    /// logits E W Eᵀ stay in a range where a central difference resolves the
    /// derivative and the GELU is off its linear part: an entry of E W Eᵀ has
    /// scale `width * embedding_sigma^2 * weight_sigma`.
    const FD_SETTINGS: [(usize, f64); 2] = [(8, 0.5), (16, 0.35)];

    /// Central-difference probe size.
    const FD_STEP: f64 = 1e-5;

    /// Bound on the entrywise deviation between an analytic gradient and its
    /// central difference. The measured maximum over `FD_SETTINGS`, the four
    /// D-graphs, both blocks, and both activations is 2.871e-9, so this leaves
    /// an order of magnitude over f64 rounding at `FD_STEP`.
    const FD_TOLERANCE: f64 = 1e-7;

    /// Parameters for a finite-difference probe at `width` and `sigma`.
    fn fd_params(width: usize, sigma: f64, activation: Activation) -> Params {
        Params {
            width,
            embedding_sigma: sigma,
            weight_sigma: sigma,
            activation,
            regime: Regime::LearnableEmbedding,
            ..Params::default()
        }
    }

    /// Both gradient blocks agree with entrywise central differences of the
    /// loss, on every D-graph and both activations. This is the check the
    /// hand-derived backward pass of `gradients_of` stands on.
    #[test]
    fn gradients_match_central_differences_on_every_d_graph() {
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();

        for activation in [Activation::Linear, Activation::Gelu] {
            for (name, graph) in d_graphs() {
                let system = TinyNn::new(&graph).expect("TinyNn::new");
                for (width, sigma) in FD_SETTINGS {
                    let params = fd_params(width, sigma, activation);
                    let parameters = system
                        .initial_parameters(&params, SEED)
                        .expect("initial_parameters");
                    let analytic = system
                        .gradients(&parameters, activation, Regime::LearnableEmbedding)
                        .expect("gradients");

                    let numeric_weight =
                        central_differences(&parameters.weight, FD_STEP, |probe| {
                            let probed =
                                Parameters::new(parameters.embedding.clone(), probe.clone())
                                    .expect("Parameters::new");
                            system.loss(&probed, activation).expect("loss")
                        });
                    let numeric_embedding =
                        central_differences(&parameters.embedding, FD_STEP, |probe| {
                            let probed = Parameters::new(probe.clone(), parameters.weight.clone())
                                .expect("Parameters::new");
                            system.loss(&probed, activation).expect("loss")
                        });

                    let embedding_gradient = analytic
                        .embedding()
                        .expect("a learnable-embedding run carries an embedding gradient");
                    for (block, analytic, numeric) in [
                        ("W", analytic.weight(), &numeric_weight),
                        ("E", embedding_gradient, &numeric_embedding),
                    ] {
                        let deviation = (analytic - numeric).amax();
                        if deviation > worst {
                            worst = deviation;
                            worst_label = format!(
                                "{name} {block} at m = {width}, sigma = {sigma}, \
                                         {activation:?}"
                            );
                        }
                        assert!(
                            deviation < FD_TOLERANCE,
                            "{name} {block} at m = {width}, sigma = {sigma}, {activation:?}: \
                             max |analytic − central difference| = {deviation:.6e}, tolerance \
                             {FD_TOLERANCE:e}; max |analytic| = {:.6e}, probe step {FD_STEP:e}",
                            analytic.amax()
                        );
                    }
                }
            }
        }

        println!(
            "gradients_match_central_differences_on_every_d_graph: \
             max deviation {worst:.6e} at {worst_label}"
        );
    }

    /// Entries per block the sampled finite-difference pin probes, of the
    /// 262 144 of W and the order × 512 of E at the production width. Each
    /// probe is two forward passes through a 512-wide matrix product, which is
    /// what holds the sample here rather than at the dense sweep
    /// `central_differences` runs at m = 8 and 16.
    const SAMPLED_FD_ENTRIES: usize = 64;

    /// Seed of the `ChaCha20` stream that picks those entries.
    const SAMPLED_FD_SEED: u64 = 20_260_830;

    /// Bound on the entrywise deviation between an analytic gradient and its
    /// central difference at the production width. The measured maximum over
    /// the 1024 comparisons — the four D-graphs, both blocks and both
    /// activations, with same-shaped blocks sampling the same seeded
    /// positions — is 2.147e-9, so this leaves an order of magnitude over f64
    /// rounding at `FD_STEP`.
    const SAMPLED_FD_TOLERANCE: f64 = 1e-7;

    /// Smallest per-block maximum |analytic| the sampled pin requires of its
    /// sample. The measured minimum over the eight (graph, activation) pairs
    /// and both blocks is 1.857e-3.
    const SAMPLED_FD_FLOOR: f64 = 1e-4;

    /// Both gradient blocks agree with central differences at a seeded sample
    /// of their entries, at [`Params::default`]'s width m = 512 and its draw
    /// scales, on every D-graph and both activations. `FD_SETTINGS` probes
    /// m = 8 and 16 densely; this probes the width every run of the crate
    /// uses. The measured maximum deviation and the sampled gradient
    /// magnitudes are printed.
    #[test]
    fn gradients_match_sampled_central_differences_at_the_production_width() {
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();
        let mut smallest_magnitude = f64::INFINITY;
        let mut largest_pre = 0.0_f64;
        let mut compared = 0_usize;

        for activation in [Activation::Linear, Activation::Gelu] {
            for (name, graph) in d_graphs() {
                let system = TinyNn::new(&graph).expect("TinyNn::new");
                let params = Params {
                    activation,
                    regime: Regime::LearnableEmbedding,
                    ..Params::default()
                };
                let parameters = system
                    .initial_parameters(&params, SEED)
                    .expect("initial_parameters");
                let analytic = system
                    .gradients(&parameters, activation, Regime::LearnableEmbedding)
                    .expect("gradients");
                largest_pre =
                    largest_pre.max((parameters.embedding() * parameters.weight()).amax());

                let numeric_weight = sampled_central_differences(
                    parameters.weight(),
                    FD_STEP,
                    SAMPLED_FD_ENTRIES,
                    SAMPLED_FD_SEED,
                    |probe| {
                        let probed = Parameters::new(parameters.embedding().clone(), probe.clone())
                            .expect("Parameters::new");
                        system.loss(&probed, activation).expect("loss")
                    },
                );
                let numeric_embedding = sampled_central_differences(
                    parameters.embedding(),
                    FD_STEP,
                    SAMPLED_FD_ENTRIES,
                    SAMPLED_FD_SEED,
                    |probe| {
                        let probed = Parameters::new(probe.clone(), parameters.weight().clone())
                            .expect("Parameters::new");
                        system.loss(&probed, activation).expect("loss")
                    },
                );

                let embedding_gradient = analytic
                    .embedding()
                    .expect("a learnable-embedding run carries an embedding gradient");
                for (block, gradient, sample) in [
                    ("W", analytic.weight(), &numeric_weight),
                    ("E", embedding_gradient, &numeric_embedding),
                ] {
                    let label = format!("{name} {block} at m = {}, {activation:?}", params.width);
                    let mut magnitude = 0.0_f64;
                    for &(position, numeric) in sample {
                        let deviation = (gradient[position] - numeric).abs();
                        magnitude = magnitude.max(gradient[position].abs());
                        compared += 1;
                        if deviation > worst {
                            worst = deviation;
                            worst_label = format!("{label}, entry {position:?}");
                        }
                        assert!(
                            deviation < SAMPLED_FD_TOLERANCE,
                            "{label}: entry {position:?} deviates by {deviation:.6e} from its \
                             central difference, tolerance {SAMPLED_FD_TOLERANCE:e}; analytic \
                             {:.6e}, numeric {numeric:.6e}, probe step {FD_STEP:e}",
                            gradient[position]
                        );
                    }
                    smallest_magnitude = smallest_magnitude.min(magnitude);
                }
            }
        }

        println!(
            "gradients_match_sampled_central_differences_at_the_production_width: \
             {compared} entries compared at {SAMPLED_FD_ENTRIES} per block, max deviation \
             {worst:.6e} at {worst_label}, smallest sampled max |analytic| over the blocks \
             {smallest_magnitude:.6e}, max |E W| {largest_pre:.6e}"
        );
        assert!(
            smallest_magnitude > SAMPLED_FD_FLOOR,
            "one block's sampled entries reach only {smallest_magnitude:.6e}, below \
             {SAMPLED_FD_FLOOR:e}, so the agreement above would hold for a sample of \
             near-zero gradient entries"
        );
    }

    /// The frozen regime carries no embedding gradient and leaves the weight
    /// gradient unchanged, so the two regimes differ only in which blocks the
    /// descent moves.
    #[test]
    fn the_frozen_regime_drops_only_the_embedding_gradient() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let params = fd_params(12, 0.4, Activation::Linear);
            let parameters = system
                .initial_parameters(&params, SEED)
                .expect("initial_parameters");

            let frozen = system
                .gradients(&parameters, Activation::Linear, Regime::FrozenEmbedding)
                .expect("gradients");
            let learnable = system
                .gradients(&parameters, Activation::Linear, Regime::LearnableEmbedding)
                .expect("gradients");

            assert!(
                frozen.embedding().is_none(),
                "{name}: the frozen regime produced an embedding gradient"
            );
            let deviation = (frozen.weight() - learnable.weight()).amax();
            assert!(
                deviation < 1e-15,
                "{name}: the two regimes' weight gradients differ by {deviation:.6e}"
            );
            let magnitude = learnable
                .embedding()
                .expect("a learnable-embedding run carries an embedding gradient")
                .amax();
            assert!(
                magnitude > 1e-6,
                "{name}: the learnable embedding gradient is {magnitude:.6e}, so the contrast \
                 above would hold for a system with no embedding gradient at all"
            );
        }
    }

    /// With W the identity and a linear hidden layer the TinyNN logits are
    /// Tier 1's VVᵀ, and the descent direction −∂L/∂E is Lemma 6's ascent
    /// direction CV. This pins both the sign of the residual P − D⁻¹A and the
    /// two tiers' shared objective.
    #[test]
    fn the_identity_weight_reproduces_the_tier1_ascent_direction() {
        let width = 12;
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let tier1 = Node2Vec::new(&graph).expect("Node2Vec::new");
            let tier1_params = node2vec::Params {
                dimension: width,
                sigma: 0.5,
                ..node2vec::Params::default()
            };
            let embedding = tier1
                .initial_embedding(&tier1_params, SEED)
                .expect("initial_embedding");
            let parameters = Parameters::new(embedding.clone(), DMatrix::identity(width, width))
                .expect("Parameters::new");

            let descent = -system
                .gradients(&parameters, Activation::Linear, Regime::LearnableEmbedding)
                .expect("gradients")
                .embedding()
                .expect("a learnable-embedding run carries an embedding gradient");
            let ascent = tier1.gradient(&embedding).expect("gradient");

            let deviation = (&descent - &ascent).amax();
            assert!(
                deviation < 1e-12,
                "{name}: −∂L/∂E deviates from Lemma 6's CV by {deviation:.6e}"
            );
            assert!(
                ascent.amax() > 1e-3,
                "{name}: Lemma 6's CV has magnitude {:.6e}, so the agreement above would hold \
                 for any pair of near-zero matrices",
                ascent.amax()
            );
        }
    }

    /// The associative score is 1 exactly where every vertex ranks its
    /// neighbours above every non-neighbour, and below 1 at a uniform
    /// distribution — so the Figure-7 pin measures the ranking rather than the
    /// metric's floor.
    #[test]
    fn the_associative_score_reads_the_top_d_ranking() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let order = graph.order();

            let memorized = row_softmax(&(graph.adjacency() * 10.0));
            let score = system.associative_score(&memorized);
            assert!(
                (score - 1.0).abs() < FULL_MEMORIZATION_SLACK,
                "{name}: a distribution peaked on the neighbours scores {score:.15}, expected 1"
            );

            let uniform = DMatrix::from_element(order, order, 1.0 / order as f64);
            let flat = system.associative_score(&uniform);
            assert!(
                flat < 1.0 - FULL_MEMORIZATION_SLACK,
                "{name}: a uniform distribution scores {flat:.15}, so the maximum above is \
                 reached by any distribution at all"
            );
        }
    }

    /// The shell means read the distance structure of a cosine matrix rather
    /// than its edge set: on a matrix built to hold a fixed value per shell,
    /// every mean is that value, so the pair lists index the shells they name.
    #[test]
    fn the_shell_means_read_the_value_planted_in_each_shell() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let order = graph.order();
            let shells = system.shell_count();

            let mut planted = DMatrix::<f64>::identity(order, order);
            for distance in 1..=shells {
                let value = 1.0 - 0.2 * distance as f64;
                for &(u, v) in system.shell(distance) {
                    planted[(u, v)] = value;
                    planted[(v, u)] = value;
                }
            }

            let means = system.shell_means_of(&planted);
            assert_eq!(
                means.len(),
                shells,
                "{name}: the profile has {} entries, expected {shells}",
                means.len()
            );
            for (index, mean) in means.iter().enumerate() {
                let expected = 1.0 - 0.2 * (index + 1) as f64;
                assert!(
                    (mean - expected).abs() < 1e-12,
                    "{name}: shell {} has mean {mean:.15}, planted {expected:.15}",
                    index + 1
                );
            }
        }
    }

    /// The separation on the profile shape the learnable runs actually reach:
    /// a negative deepest shell below a positive one above it, where the
    /// distance-from-zero branch binds rather than the separation cap. Each
    /// case names the branch it exercises and the value it expects; the
    /// non-finite cases pin that such a profile scores NaN, which fails every
    /// threshold comparison.
    #[test]
    fn the_deepest_shell_separation_reads_a_negative_deepest_shell() {
        let cases: [(&str, [f64; 3], f64); 4] = [
            // The cycle(15) profile at eta = 0.001, where |deepest| is 4.6x
            // below the separation.
            (
                "measured cycle",
                [-0.162_791, 0.347_921, -0.096_173],
                0.096_173,
            ),
            // Separation binding on the same sign pattern.
            ("separation binds", [0.0, -0.10, -0.40], 0.30),
            // A deepest shell above the one over it scores zero, not a
            // negative separation.
            ("inverted shells", [0.0, -0.40, -0.10], -0.30),
            // No structure at all.
            ("flat", [0.2, 0.2, 0.2], 0.0),
        ];

        for (label, means, expected) in cases {
            let separation = profile_separation(&means);
            assert!(
                (separation - expected).abs() < 1e-9,
                "{label}: shell means {means:?} score {separation:.9}, expected {expected:.9}"
            );
        }

        for (label, means) in [
            ("NaN deepest", [0.2, 0.4, f64::NAN]),
            ("NaN above", [0.2, f64::NAN, 0.4]),
            ("infinite deepest", [0.2, 0.4, f64::NEG_INFINITY]),
            ("infinite above", [0.2, f64::INFINITY, 0.4]),
        ] {
            let separation = profile_separation(&means);
            assert!(
                separation.is_nan(),
                "{label}: shell means {means:?} score {separation}, expected NaN"
            );
        }

        let single = profile_separation(&[0.4]);
        assert!(
            single.is_nan(),
            "a one-entry profile scores {single}, expected NaN"
        );
    }

    /// The associative reference the separation measures against: an embedding
    /// whose rows are the adjacency rows scores exactly zero on every D-graph,
    /// its deepest-shell pairs having no common neighbour. Its distance-2 mean
    /// is asserted positive alongside, so the zero is a property of the deepest
    /// shell rather than of a cosine matrix with no structure at all.
    #[test]
    fn an_adjacency_row_embedding_scores_zero_on_the_deepest_shell() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let means = system.shell_means(graph.adjacency());
            let separation = system.deepest_shell_separation(graph.adjacency());
            let deepest = means[system.shell_count() - 1];

            println!("{name}: adjacency-row shell means {means:?}, separation {separation:.6}");
            assert!(
                deepest.abs() < 1e-15,
                "{name}: the adjacency-row embedding's distance-{} mean is {deepest:.6e}, \
                 expected 0",
                system.shell_count()
            );
            assert!(
                separation.abs() < 1e-15,
                "{name}: the adjacency-row embedding scores {separation:.6e}, expected 0"
            );
            assert!(
                means[1] > 0.1,
                "{name}: the adjacency-row embedding's distance-2 mean is {:.6}, so the zero \
                 above would hold for a cosine matrix with no shell structure at all",
                means[1]
            );
        }
    }

    /// An embedding that left the finite range scores NaN on the shell
    /// separation and is a typed error on the Fiedler alignment — a diverged
    /// run cannot report a geometry through either instrument. The finite
    /// embedding beside it scores a number through both, so the guards are not
    /// firing on every input.
    #[test]
    fn a_non_finite_embedding_cannot_report_a_geometry() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&graph).expect("TinyNn::new");

        for (label, entry) in [("NaN", f64::NAN), ("infinite", f64::INFINITY)] {
            let broken = DMatrix::from_element(15, 4, entry);
            let separation = system.deepest_shell_separation(&broken);
            assert!(
                separation.is_nan(),
                "{label}: the shell separation is {separation}, expected NaN"
            );
            assert!(
                separation.partial_cmp(&0.0).is_none(),
                "{label}: a NaN separation orders against 0 as {:?}",
                separation.partial_cmp(&0.0)
            );
            match system.fiedler_alignment(&broken) {
                Err(Error::NonFinite { .. }) => {}
                other => panic!("{label}: expected NonFinite, got {other:?}"),
            }
        }

        let finite = cosine_similarity(graph.adjacency());
        assert!(
            system.deepest_shell_separation(&finite).is_finite(),
            "a finite embedding scores NaN on the shell separation, so the guard above fires \
             on every input"
        );
        assert!(
            system.fiedler_alignment(&finite).is_ok(),
            "a finite embedding is an error on the Fiedler alignment, so the guard above fires \
             on every input"
        );
    }

    /// The embedding whose rows are `sign(e)` for the leading Fiedler-like
    /// eigenvector `e` of `system`: rank one, and cosine ±1 on every pair.
    fn fiedler_sign_embedding(system: &TinyNn) -> DMatrix<f64> {
        let column = system.fiedler_like().start;
        let fiedler = system.spectrum().eigenvectors().column(column).into_owned();
        DMatrix::from_fn(system.order(), 1, |u, _| {
            if fiedler[u] >= 0.0 { 1.0 } else { -1.0 }
        })
    }

    /// Draws the calibration test's Gaussian references.
    const GAUSSIAN_DRAWS: u64 = 200;

    /// The references [`FIEDLER_ALIGNMENT`] is calibrated against, measured on
    /// every D-graph: the Fiedler-like eigenvectors of −L and a converged
    /// Tier-1 `Node2Vec` embedding are above the threshold, while a rank-1
    /// Fiedler-sign embedding, an all-rows-identical embedding, and 200 raw
    /// Gaussian draws are below it. Every measured value is printed.
    #[test]
    fn the_fiedler_alignment_calibration_separates_the_references() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let fiedler = system.fiedler_like();
            let reference = system
                .spectrum()
                .eigenvectors()
                .columns(fiedler.start, fiedler.len())
                .into_owned();

            let history = TempPath::new("calibration-tier1");
            let tier1 = node2vec::run_tied(
                &graph,
                &node2vec::Params::default(),
                SEED,
                history.path(),
                || false,
            )
            .expect("run_tied");

            let mut worst_draw = 0.0_f64;
            for seed in 0..GAUSSIAN_DRAWS {
                let draw = gaussian_matrix(system.order(), 8, 1.0, seed).expect("gaussian_matrix");
                let score = system.fiedler_alignment(&draw).expect("fiedler_alignment");
                worst_draw = worst_draw.max(score);
            }

            let identical = DMatrix::from_fn(system.order(), 8, |_, j| (j + 1) as f64);
            // The most negative eigendirections of −L assign adjacent vertices
            // opposite signs, so this embedding's neighbours are near-antipodal
            // — the structure the retracted shell criterion certified, and the
            // one the learnable runs loaded on.
            let order = system.order();
            let bottom = system
                .spectrum()
                .eigenvectors()
                .columns(order - fiedler.len(), fiedler.len())
                .into_owned();
            let measured = [
                ("Fiedler eigenvectors", &reference),
                ("Node2Vec (Tier 1)", tier1.embedding()),
                ("rank-1 Fiedler sign", &fiedler_sign_embedding(&system)),
                ("all rows identical", &identical),
                ("bottom eigenvectors (antipodal neighbours)", &bottom),
            ]
            .map(|(label, embedding)| {
                (
                    label,
                    system
                        .fiedler_alignment(embedding)
                        .expect("fiedler_alignment"),
                )
            });

            println!(
                "{name}: eigenvalues {:?}, Fiedler-like range {fiedler:?}",
                system
                    .spectrum()
                    .eigenvalues()
                    .iter()
                    .map(|value| format!("{value:.6}"))
                    .collect::<Vec<_>>()
            );
            for (label, score) in measured {
                println!("{name}: {label} scores {score:.6}");
            }
            println!(
                "{name}: {GAUSSIAN_DRAWS} Gaussian draws peak at {worst_draw:.6}, \
                 threshold {FIEDLER_ALIGNMENT}"
            );

            for (label, score) in &measured[..2] {
                assert!(
                    *score >= FIEDLER_ALIGNMENT,
                    "{name}: {label} scores {score:.6}, below the {FIEDLER_ALIGNMENT} criterion \
                     it calibrates"
                );
            }
            for (label, score) in &measured[2..] {
                assert!(
                    *score < FIEDLER_ALIGNMENT,
                    "{name}: {label} scores {score:.6}, at or above the {FIEDLER_ALIGNMENT} \
                     criterion"
                );
            }
            assert!(
                worst_draw < FIEDLER_ALIGNMENT,
                "{name}: the highest of {GAUSSIAN_DRAWS} Gaussian draws scores \
                 {worst_draw:.6}, at or above the {FIEDLER_ALIGNMENT} criterion"
            );
        }
    }

    /// The Fiedler-like set the alignment measures against agrees with Tier
    /// 1's `fiedler_like_range` on each connected D-graph, and on the
    /// disconnected one it is the two components' Fiedler vectors — indices
    /// 2..4 — where `fiedler_like_range` returns 1..2, the second component's
    /// own leading eigenvector.
    #[test]
    fn the_fiedler_like_set_agrees_with_tier1_on_a_connected_graph() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let tier1 = fiedler_like_range(system.spectrum(), fiedler_spread(system.spectrum()));
            println!(
                "{name}: trivial block {:?}, Fiedler-like set {:?}, Tier 1 range {tier1:?}",
                system.trivial_block(),
                system.fiedler_like()
            );

            if name == "irregular()" {
                assert_eq!(
                    system.fiedler_like(),
                    2..4,
                    "irregular(): the Fiedler-like set is {:?}, expected the two components' \
                     Fiedler vectors 2..4",
                    system.fiedler_like()
                );
                assert_eq!(
                    tier1,
                    1..2,
                    "irregular(): Tier 1's range is {tier1:?}, expected 1..2"
                );
            } else {
                assert_eq!(
                    system.fiedler_like(),
                    tier1,
                    "{name}: the Fiedler-like set is {:?}, Tier 1's range is {tier1:?}",
                    system.fiedler_like()
                );
                assert_eq!(
                    system.trivial_block(),
                    0..1,
                    "{name}: the trivial block is {:?}, expected the single leading eigenvector \
                     of a connected graph",
                    system.trivial_block()
                );
            }
        }
    }

    /// The alignment is unchanged by scaling the embedding over the range
    /// 1e-120 to 1e120, where the Gram matrix stays inside f64 — so a run
    /// whose parameters grew or shrank by many orders of magnitude is neither
    /// rewarded nor punished for the change of scale alone.
    #[test]
    fn the_fiedler_alignment_is_scale_invariant() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let draw = gaussian_matrix(system.order(), 8, 1.0, SEED).expect("gaussian_matrix");
            let base = system.fiedler_alignment(&draw).expect("fiedler_alignment");
            assert!(
                base > 0.0,
                "{name}: the draw scores {base:.6e}, so the equalities below would hold for a \
                 measure that is always zero"
            );

            for scale in [1e-120, 1e-6, 1e6, 1e120] {
                let scaled = &draw * scale;
                let score = system
                    .fiedler_alignment(&scaled)
                    .expect("fiedler_alignment");
                assert!(
                    (score - base).abs() < 1e-12,
                    "{name}: scaling the embedding by {scale:e} moved the alignment from \
                     {base:.15} to {score:.15}"
                );
            }
        }
    }

    /// Distance shell 1 is the edge set, and the 15-cycle has 15 pairs at each
    /// of its seven distances — its whole diameter, covering all 105 vertex
    /// pairs.
    #[test]
    fn distance_shells_match_the_edge_and_cycle_counts() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            assert_eq!(
                system.shell(1).len(),
                graph.edge_count(),
                "{name}: shell 1 holds {} pairs, the graph has {} edges",
                system.shell(1).len(),
                graph.edge_count()
            );
            println!(
                "{name}: {} distance shells, sizes {:?}",
                system.shell_count(),
                (1..=system.shell_count())
                    .map(|distance| system.shell(distance).len())
                    .collect::<Vec<_>>()
            );
        }

        let cycle = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&cycle).expect("TinyNn::new");
        assert_eq!(
            system.shell_count(),
            7,
            "cycle(15): {} distance shells, expected the diameter 7",
            system.shell_count()
        );
        for distance in 1..=system.shell_count() {
            assert_eq!(
                system.shell(distance).len(),
                15,
                "cycle(15): shell {distance} holds {} pairs, expected 15",
                system.shell(distance).len()
            );
        }
    }

    /// A graph whose vertices are all mutually adjacent has one distance
    /// shell, and building a system on it is a typed error rather than a
    /// division by an empty shell.
    #[test]
    fn a_graph_without_enough_distance_shells_is_a_typed_error() {
        let complete = Graph::complete(7).expect("complete(7)");
        match TinyNn::new(&complete) {
            Err(Error::InsufficientDistanceShells { available }) => {
                assert_eq!(
                    available, 1,
                    "reported {available} populated shells, expected 1"
                );
            }
            other => panic!("expected InsufficientDistanceShells, got {other:?}"),
        }
    }

    /// Degenerate run parameters come back as typed errors naming the field.
    #[test]
    fn params_reject_degenerate_values() {
        let base = Params::default();
        let zero_width = Params { width: 0, ..base };
        match zero_width.validate() {
            Err(Error::InvalidDimension { dimension }) => {
                assert_eq!(dimension, 0, "reported width {dimension}");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }
        let zero_steps = Params {
            max_steps: 0,
            ..base
        };
        match zero_steps.validate() {
            Err(Error::ZeroMaxSteps) => {}
            other => panic!("expected ZeroMaxSteps, got {other:?}"),
        }

        for (parameter, params) in [
            (
                "embedding_sigma",
                Params {
                    embedding_sigma: 0.0,
                    ..base
                },
            ),
            (
                "weight_sigma",
                Params {
                    weight_sigma: f64::INFINITY,
                    ..base
                },
            ),
            (
                "learning_rate",
                Params {
                    learning_rate: -1.0,
                    ..base
                },
            ),
            (
                "tolerance",
                Params {
                    tolerance: f64::NAN,
                    ..base
                },
            ),
        ] {
            match params.validate() {
                Err(Error::InvalidRunParameter {
                    parameter: observed,
                    ..
                }) => {
                    assert_eq!(
                        observed, parameter,
                        "rejected parameter {observed:?}, expected {parameter:?}"
                    );
                }
                other => panic!("expected InvalidRunParameter for {parameter}, got {other:?}"),
            }
        }
    }

    /// A weight matrix that is not square of the embedding's width, and an
    /// embedding with the wrong row count, are rejected before any linear
    /// algebra runs.
    #[test]
    fn shape_mismatches_are_typed_errors() {
        match Parameters::new(DMatrix::zeros(15, 6), DMatrix::zeros(6, 5)) {
            Err(Error::WeightShapeMismatch {
                rows,
                columns,
                width,
            }) => {
                assert_eq!(
                    (rows, columns, width),
                    (6, 5, 6),
                    "reported {rows}×{columns} against width {width}"
                );
            }
            other => panic!("expected WeightShapeMismatch, got {other:?}"),
        }

        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&graph).expect("TinyNn::new");
        let wrong =
            Parameters::new(DMatrix::zeros(4, 6), DMatrix::zeros(6, 6)).expect("Parameters::new");
        match system.loss(&wrong, Activation::Linear) {
            Err(Error::EmbeddingOrderMismatch { rows, order }) => {
                assert_eq!(
                    (rows, order),
                    (4, 15),
                    "reported {rows} rows against {order}"
                );
            }
            other => panic!("expected EmbeddingOrderMismatch, got {other:?}"),
        }
    }

    /// Applied updates the frozen-embedding pin allows, two orders of
    /// magnitude above the step at which the edges are memorized.
    const ASSOCIATIVE_BUDGET: usize = 200;

    /// Learning rates decision D10 sweeps, with the applied-update budget each
    /// gets: the 10²–10³ steps Figs. 8 and 22 plot.
    const GEOMETRIC_SWEEP: [(f64, usize); 3] = [(0.001, 1_200), (0.01, 200), (0.1, 50)];

    /// Movement in the Fiedler alignment over a learnable run below which the
    /// criterion is reading something other than the trained embedding. The
    /// smallest measured move over `GEOMETRIC_SWEEP` and the four D-graphs at
    /// seed 20260829 is 0.0308, on the 4×4 grid at η = 0.01.
    const ALIGNMENT_DRIFT: f64 = 1e-3;

    /// Runs `params` on `graph` at [`SEED`] into temp files, printing the
    /// measurement.
    fn measured_run(label: &str, graph: &Graph, params: &Params) -> Run {
        measured_run_at(label, graph, params, SEED)
    }

    /// Runs `params` on `graph` at `seed` into temp files, printing the
    /// measurement.
    fn measured_run_at(label: &str, graph: &Graph, params: &Params, seed: u64) -> Run {
        let (run, report) = measured_run_report(label, graph, params, seed);
        println!("{report}");
        run
    }

    /// Runs `params` on `graph` at `seed` into temp files, returning the run
    /// beside its one-line measurement rather than printing it.
    fn measured_run_report(
        label: &str,
        graph: &Graph,
        params: &Params,
        seed: u64,
    ) -> (Run, String) {
        let history = TempPath::new("history");
        let cosines = TempPath::new("cosines");
        let outputs = Outputs {
            history: history.path(),
            cosines: cosines.path(),
        };
        let started = Instant::now();
        let run = run(graph, params, seed, &outputs, || false).expect("run");
        let last = run.last().expect("a run records its initial state");
        let report = format!(
            "{label}: {:?}, outcome {:?}, {} steps, loss {:.6} (was {:.6}), \
             associative step {:?} (peak {:.6}, initial {:.6}), alignment step {:?} \
             (peak {:.6}, initial {:.6}, final {:.6}), peak shell separation {:.6}, \
             final shell means {:?}",
            started.elapsed(),
            run.outcome(),
            run.steps(),
            last.loss(),
            run.records()[0].loss(),
            run.associative_step(),
            run.peak_associative_score(),
            run.records()[0].associative_score(),
            run.alignment_step(FIEDLER_ALIGNMENT),
            run.peak_alignment(),
            run.records()[0].fiedler_alignment(),
            last.fiedler_alignment(),
            run.peak_deepest_shell_separation(),
            last.shell_means()
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
        );
        (run, report)
    }

    /// The second seed the frozen-run instruments measure over, beside
    /// [`SEED`]: the pair `tests/tier2_tinynn.rs`'s `TIMING_SEEDS` times.
    const SECOND_SEED: u64 = 42;

    /// The Pearson correlation of two square same-order matrices over their
    /// n(n − 1) off-diagonal entries (u, v), u ≠ v: with x and y those entries
    /// of the two matrices and x̄, ȳ their means,
    /// Σ(x − x̄)(y − ȳ) / √(Σ(x − x̄)² · Σ(y − ȳ)²). Panics on a shape
    /// mismatch or a constant off-diagonal on either side.
    fn off_diagonal_correlation(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
        assert_eq!(
            left.nrows(),
            left.ncols(),
            "off_diagonal_correlation takes square matrices"
        );
        assert_eq!(
            left.shape(),
            right.shape(),
            "off_diagonal_correlation takes same-order matrices"
        );
        let order = left.nrows();
        let pairs: Vec<(f64, f64)> = (0..order)
            .flat_map(|u| (0..order).map(move |v| (u, v)))
            .filter(|&(u, v)| u != v)
            .map(|(u, v)| (left[(u, v)], right[(u, v)]))
            .collect();
        let count = pairs.len() as f64;
        let left_mean = pairs.iter().map(|&(x, _)| x).sum::<f64>() / count;
        let right_mean = pairs.iter().map(|&(_, y)| y).sum::<f64>() / count;

        let mut covariance = 0.0;
        let mut left_spread = 0.0;
        let mut right_spread = 0.0;
        for &(x, y) in &pairs {
            let dx = x - left_mean;
            let dy = y - right_mean;
            covariance += dx * dy;
            left_spread += dx * dx;
            right_spread += dy * dy;
        }
        assert!(
            left_spread > 0.0 && right_spread > 0.0,
            "off_diagonal_correlation needs non-constant off-diagonal entries"
        );
        covariance / (left_spread * right_spread).sqrt()
    }

    /// The parameters after `steps` updates from the `seed` draw, replaying
    /// `run`'s loop through the public gradient API: each step subtracts η
    /// times the gradient of the blocks `params.regime` trains, read off the
    /// same pre-update parameters.
    fn parameters_after(system: &TinyNn, params: &Params, seed: u64, steps: usize) -> Parameters {
        let mut parameters = system
            .initial_parameters(params, seed)
            .expect("initial_parameters");
        for _ in 0..steps {
            let gradients = system
                .gradients(&parameters, params.activation, params.regime)
                .expect("gradients");
            let weight_update = gradients.weight() * params.learning_rate;
            let embedding_update = gradients
                .embedding()
                .map(|gradient| gradient * params.learning_rate);
            parameters.weight -= &weight_update;
            if let Some(update) = &embedding_update {
                parameters.embedding -= update;
            }
        }
        parameters
    }

    /// Reports [`off_diagonal_correlation`] between the model's distribution at
    /// `run`'s memorization step and the target D⁻¹A, recomputing that
    /// distribution from the same seed through [`parameters_after`]. Pins the
    /// replay to the run bit-for-bit through the recorded loss, and floors the
    /// correlation at 0.9 so the 0.9419–0.9756 the CHANGELOG quotes cannot
    /// drift silently.
    fn hit_step_correlation_report(
        label: &str,
        graph: &Graph,
        params: &Params,
        seed: u64,
        run: &Run,
    ) -> String {
        let system = TinyNn::new(graph).expect("TinyNn::new");
        let hit = run.associative_step().unwrap_or_else(|| {
            panic!(
                "{label}: the top-d score never reached 1 in {} steps; it peaked at {:.6}",
                params.max_steps,
                run.peak_associative_score()
            )
        });
        let target = transition(graph).expect("transition");
        let parameters = parameters_after(&system, params, seed, hit);
        let probabilities = system
            .probabilities(&parameters, params.activation)
            .expect("probabilities");
        let replayed_loss = system.loss(&parameters, params.activation).expect("loss");
        let recorded_loss = run
            .records()
            .iter()
            .find(|record| record.step() == hit)
            .expect("invariant: `associative_step` names a recorded step")
            .loss();
        assert!(
            replayed_loss.to_bits() == recorded_loss.to_bits(),
            "{label}: the replay diverges from the run at step {hit}: loss {replayed_loss:e} \
             against the recorded {recorded_loss:e}, so the correlation below reads a state \
             the run never visited"
        );
        let correlation = off_diagonal_correlation(&probabilities, &target);
        let report = format!(
            "{label}: at the hit step {hit}, the off-diagonal Pearson correlation between the \
             model's distribution and D⁻¹A is {correlation:.6}; the replayed loss matches the \
             recorded {recorded_loss:.6} bit-for-bit",
        );
        assert!(
            correlation > 0.9,
            "{label}: the off-diagonal correlation {correlation:.6} fell below 0.9, off the \
             0.9419–0.9756 the CHANGELOG quotes"
        );
        report
    }

    /// Fraction of the largest singular value below which a direction counts
    /// as absent for [`embedding_spread`]'s numeric rank.
    const NUMERIC_RANK_FLOOR: f64 = 1e-9;

    /// Three non-degeneracy readings of an embedding.
    struct EmbeddingSpread {
        /// The count of singular values above [`NUMERIC_RANK_FLOOR`] times the
        /// largest.
        rank: usize,
        /// The participation ratio (Σσ)²/Σσ² of the singular values, n for a
        /// flat spectrum and 1 for a rank-one one.
        participation: f64,
        /// The largest row ℓ2 norm over the smallest.
        row_norms: f64,
    }

    /// The numeric rank, participation ratio, and row-norm spread of
    /// `embedding`, from its singular values and its row ℓ2 norms.
    fn embedding_spread(embedding: &DMatrix<f64>) -> EmbeddingSpread {
        let singular = embedding.singular_values();
        let largest = singular.iter().copied().fold(0.0_f64, f64::max);
        let rank = singular
            .iter()
            .filter(|&&value| value > NUMERIC_RANK_FLOOR * largest)
            .count();
        let total: f64 = singular.iter().sum();
        let squares: f64 = singular.iter().map(|value| value * value).sum();

        let mut widest = 0.0_f64;
        let mut narrowest = f64::INFINITY;
        for row in embedding.row_iter() {
            let norm = row.norm();
            widest = widest.max(norm);
            narrowest = narrowest.min(norm);
        }

        EmbeddingSpread {
            rank,
            participation: total * total / squares,
            row_norms: widest / narrowest,
        }
    }

    /// Figures 8, 22 and 23 through the run API. On every D-graph and every
    /// swept learning rate the learnable run's embedding stays off the
    /// Fiedler-like eigenvectors of −L, so no step meets the §4.1 criterion;
    /// and its final shell profile over the whole diameter attains its
    /// maximum at distance 2, above the distance-1 mean that Figure 23 reads
    /// as the cosine matrix reproducing the adjacency. Both measurements are
    /// printed per graph and rate.
    ///
    /// The 12 configurations run on a [`sweep_pool`], each run internally
    /// sequential and drawing from its own seeded stream. Each returns its
    /// measurement lines, printed in configuration order once the pool is
    /// done; a panicking configuration is re-raised after the surviving lines
    /// print.
    #[test]
    fn the_learnable_run_misses_the_geometry_and_peaks_at_distance_two() {
        let graphs = d_graphs();
        let configurations: Vec<(&str, &Graph, f64, usize)> = graphs
            .iter()
            .flat_map(|(name, graph)| {
                GEOMETRIC_SWEEP
                    .into_iter()
                    .map(move |(learning_rate, budget)| (*name, graph, learning_rate, budget))
            })
            .collect();

        let reports: Vec<std::thread::Result<Vec<String>>> = sweep_pool().install(|| {
            configurations
                .par_iter()
                .map(|&(name, graph, learning_rate, budget)| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        learnable_sweep_reports(name, graph, learning_rate, budget)
                    }))
                })
                .collect()
        });

        print_surviving_reports_then_reraise(reports);
    }

    /// Prints every non-panicked configuration's measurement lines in
    /// configuration order, then re-raises the first panic if one occurred.
    fn print_surviving_reports_then_reraise(reports: Vec<std::thread::Result<Vec<String>>>) {
        let mut first_failure = None;
        for outcome in reports {
            match outcome {
                Ok(lines) => {
                    for line in lines {
                        println!("{line}");
                    }
                }
                Err(panic) => {
                    if first_failure.is_none() {
                        first_failure = Some(panic);
                    }
                }
            }
        }
        if let Some(panic) = first_failure {
            std::panic::resume_unwind(panic);
        }
    }

    /// One configuration of
    /// [`the_learnable_run_misses_the_geometry_and_peaks_at_distance_two`]: the
    /// learnable run on `graph` at `learning_rate` over `budget` applied
    /// updates, asserted and reported as three measurement lines.
    fn learnable_sweep_reports(
        name: &str,
        graph: &Graph,
        learning_rate: f64,
        budget: usize,
    ) -> Vec<String> {
        let mut reports = Vec::new();
        let params = Params {
            learning_rate,
            max_steps: budget,
            regime: Regime::LearnableEmbedding,
            ..Params::default()
        };
        let label = format!("{name} at eta = {learning_rate}");
        let (run, report) = measured_run_report(&label, graph, &params, SEED);
        reports.push(report);

        let spread = embedding_spread(run.parameters().embedding());
        reports.push(format!(
            "{label}: final embedding numeric rank {} of {} rows (singular values above \
             {NUMERIC_RANK_FLOOR:e} times the largest), participation ratio {:.6}, \
             row-norm spread {:.6}",
            spread.rank,
            graph.order(),
            spread.participation,
            spread.row_norms
        ));
        assert_eq!(
            spread.rank,
            graph.order(),
            "{label}: the final embedding's numeric rank fell below full, off the rank-n \
             reading the CHANGELOG quotes"
        );
        assert!(
            spread.participation > 10.0,
            "{label}: the participation ratio {:.6} fell to 10, far off the 12.6–15.2 the \
             sweep prints and the CHANGELOG quotes",
            spread.participation
        );
        assert!(
            spread.row_norms < 1.5,
            "{label}: the row-norm spread {:.6} reached 1.5, off the ≤ 1.22 the CHANGELOG \
             quotes at η = 0.1",
            spread.row_norms
        );

        let peak = run.peak_alignment();
        assert!(
            peak < FIEDLER_ALIGNMENT,
            "{label}: the Fiedler alignment peaked at {peak:.6} over {budget} steps, reaching \
             the {FIEDLER_ALIGNMENT} criterion at step {:?}",
            run.alignment_step(FIEDLER_ALIGNMENT)
        );
        assert!(
            run.peak_associative_score() >= 1.0 - FULL_MEMORIZATION_SLACK,
            "{label}: the top-d score peaked at {:.6}, so the alignment null above is a run \
             that learned nothing",
            run.peak_associative_score()
        );
        let last = run
            .last()
            .expect("a run records its initial state")
            .fiedler_alignment();
        let first = run.records()[0].fiedler_alignment();
        reports.push(format!(
            "{label}: the alignment moved {:.6} over the run, from {first:.6} to {last:.6}",
            (last - first).abs()
        ));
        assert!(
            (last - first).abs() > ALIGNMENT_DRIFT,
            "{label}: the alignment moved from {first:.6} to {last:.6}, less than \
             {ALIGNMENT_DRIFT}, so the null above would hold for a measurement that never \
             reads the trained embedding"
        );

        let means = run
            .last()
            .expect("a run records its initial state")
            .shell_means();
        let (highest, peak_mean) = means.iter().enumerate().fold(
            (0_usize, f64::NEG_INFINITY),
            |(best, value), (index, &mean)| {
                if mean > value {
                    (index, mean)
                } else {
                    (best, value)
                }
            },
        );
        assert_eq!(
            highest,
            1,
            "{label}: the shell profile {means:?} peaks at distance {}, expected 2",
            highest + 1
        );
        assert!(
            peak_mean > means[0],
            "{label}: the distance-2 mean {peak_mean:.6} does not exceed the distance-1 mean \
             {:.6}, so the profile is consistent with Figure 23",
            means[0]
        );
        reports
    }

    /// The frozen-embedding regime memorizes the edges while its embedding
    /// stays at the draw: the top-d score reaches its maximum and the geometry
    /// measurements are the ones the seeded draw started with. §B.2.2 rests
    /// its associative reading on the first half; the second is a property of
    /// the regime — a frozen embedding cannot move — so the test pins that the
    /// draw itself is not already geometric.
    ///
    /// The four graphs run on a [`sweep_pool`], each graph's pair of seeded
    /// runs internally sequential. Each returns its measurement lines, printed
    /// in graph order once the pool is done; a panicking graph is re-raised
    /// after the surviving lines print.
    #[test]
    fn the_frozen_run_memorizes_without_moving_its_embedding() {
        let reports: Vec<std::thread::Result<Vec<String>>> = sweep_pool().install(|| {
            d_graphs()
                .into_par_iter()
                .map(|(name, graph)| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        frozen_sweep_reports(name, &graph)
                    }))
                })
                .collect()
        });

        print_surviving_reports_then_reraise(reports);
    }

    /// One graph of [`the_frozen_run_memorizes_without_moving_its_embedding`]:
    /// its frozen runs at [`SEED`] and [`SECOND_SEED`], asserted and reported
    /// as four measurement lines.
    fn frozen_sweep_reports(name: &str, graph: &Graph) -> Vec<String> {
        let mut reports = Vec::new();
        let params = Params {
            max_steps: ASSOCIATIVE_BUDGET,
            ..Params::default()
        };
        let (run, report) = measured_run_report(&format!("{name} frozen"), graph, &params, SEED);
        reports.push(report);
        reports.push(hit_step_correlation_report(
            &format!("{name} frozen seed {SEED}"),
            graph,
            &params,
            SEED,
            &run,
        ));
        let (second, report) = measured_run_report(
            &format!("{name} frozen seed {SECOND_SEED}"),
            graph,
            &params,
            SECOND_SEED,
        );
        reports.push(report);
        reports.push(hit_step_correlation_report(
            &format!("{name} frozen seed {SECOND_SEED}"),
            graph,
            &params,
            SECOND_SEED,
            &second,
        ));

        for (seed, frozen) in [(SEED, &run), (SECOND_SEED, &second)] {
            assert_eq!(
                frozen.associative_step(),
                Some(1),
                "{name}: the frozen run at seed {seed} did not memorize at step 1, so the \
                 both-seeds step-1 sentence in the CHANGELOG does not hold"
            );
            assert!(
                frozen.records()[0].associative_score() < 1.0,
                "{name}: the seed-{seed} draw scores {:.6} before the first update, so the \
                 step-1 pin above measures nothing",
                frozen.records()[0].associative_score()
            );
        }

        assert!(
            run.peak_associative_score() >= 1.0 - FULL_MEMORIZATION_SLACK,
            "{name}: the frozen run peaked at a top-d score of {:.6} in {ASSOCIATIVE_BUDGET} \
             steps, so the geometry null below is a run that learned nothing",
            run.peak_associative_score()
        );

        let last = run.last().expect("a run records its initial state");
        for (measure, first, last) in [
            (
                "Fiedler alignment",
                run.records()[0].fiedler_alignment(),
                last.fiedler_alignment(),
            ),
            (
                "shell separation",
                run.records()[0].deepest_shell_separation(),
                last.deepest_shell_separation(),
            ),
        ] {
            assert!(
                (first - last).abs() < 1e-12,
                "{name}: the frozen run's {measure} moved from {first:.12} to {last:.12}; a \
                 frozen embedding cannot change either measurement"
            );
        }
        let first = run.records()[0].fiedler_alignment();
        assert!(
            first < FIEDLER_ALIGNMENT,
            "{name}: the seeded draw already scores {first:.6} against the {FIEDLER_ALIGNMENT} \
             criterion, so a learnable run reaching it would not be attributable to training"
        );
        reports
    }

    /// The GELU variant carries both Tier-2 results on the 15-cycle: the
    /// frozen run memorizes the edges within Refutation 3c's two steps, and
    /// the learnable run at η = 0.01 leaves the Fiedler alignment below the
    /// criterion, as the linear variant does. The measured values are printed.
    #[test]
    fn the_gelu_variant_carries_both_results_on_the_cycle() {
        let graph = Graph::cycle(15).expect("cycle(15)");

        let frozen = Params {
            max_steps: ASSOCIATIVE_BUDGET,
            activation: Activation::Gelu,
            ..Params::default()
        };
        let associative = measured_run("cycle(15) gelu frozen", &graph, &frozen);
        let associative_step = associative.associative_step().unwrap_or_else(|| {
            panic!(
                "cycle(15) gelu: the top-d score never reached 1 in {ASSOCIATIVE_BUDGET} steps; \
                 it peaked at {:.6}",
                associative.peak_associative_score()
            )
        });
        assert!(
            associative_step <= 2,
            "cycle(15) gelu: the top-d score first reached 1 at step {associative_step}, above \
             Refutation 3c's 2"
        );

        let learnable = Params {
            learning_rate: 0.01,
            max_steps: 200,
            activation: Activation::Gelu,
            regime: Regime::LearnableEmbedding,
            ..Params::default()
        };
        let geometric = measured_run("cycle(15) gelu learnable", &graph, &learnable);
        let peak = geometric.peak_alignment();
        assert!(
            peak < FIEDLER_ALIGNMENT,
            "cycle(15) gelu: the Fiedler alignment peaked at {peak:.6} over 200 steps, \
             reaching the {FIEDLER_ALIGNMENT} criterion at step {:?}",
            geometric.alignment_step(FIEDLER_ALIGNMENT)
        );
        assert!(
            geometric.peak_associative_score() >= 1.0 - FULL_MEMORIZATION_SLACK,
            "cycle(15) gelu: the learnable run's top-d score peaked at {:.6}, so the alignment \
             null above is a run that learned nothing",
            geometric.peak_associative_score()
        );
    }

    // The transition machinery of issue #5: the W initializer, the relative
    // weight rate ρ, the §B.3 optimizer, and the geometry stop. These pin the
    // machinery; no threshold here asserts an experiment outcome.

    /// One decoupled AdamW update by hand from the §B.3 formulas, on one
    /// scalar: the decayed moments, their bias correction at update `updates`,
    /// the adaptive step, then the decay of the stepped value. Returns the new
    /// parameter beside the new moments.
    fn adamw_reference_step(
        parameter: f64,
        first: f64,
        second: f64,
        gradient: f64,
        rate: f64,
        updates: i32,
        settings: AdamW,
    ) -> (f64, f64, f64) {
        let first = settings.beta1 * first + (1.0 - settings.beta1) * gradient;
        let second = settings.beta2 * second + (1.0 - settings.beta2) * gradient * gradient;
        let corrected_first = first / (1.0 - settings.beta1.powi(updates));
        let corrected_second = second / (1.0 - settings.beta2.powi(updates));
        let step = rate * corrected_first / (corrected_second.sqrt() + settings.epsilon);
        (
            (parameter - step) * (1.0 - rate * settings.weight_decay),
            first,
            second,
        )
    }

    /// `updates` steps of [`adamw_reference_step`] over a row of parameters and
    /// a per-step row of gradients, `coupled` folding the decay into the
    /// gradient and dropping the decay factor instead.
    fn adamw_reference_run(
        start: &[f64],
        gradients: &[Vec<f64>],
        rate: f64,
        settings: AdamW,
        coupled: bool,
    ) -> Vec<f64> {
        let mut parameters = start.to_vec();
        let mut first = vec![0.0; start.len()];
        let mut second = vec![0.0; start.len()];
        let applied = if coupled {
            AdamW {
                weight_decay: 0.0,
                ..settings
            }
        } else {
            settings
        };
        for (update, row) in gradients.iter().enumerate() {
            let index = i32::try_from(update + 1).expect("invariant: the reference runs few steps");
            for j in 0..start.len() {
                let gradient = if coupled {
                    row[j] + settings.weight_decay * parameters[j]
                } else {
                    row[j]
                };
                let (next, moment, square) = adamw_reference_step(
                    parameters[j],
                    first[j],
                    second[j],
                    gradient,
                    rate,
                    index,
                    applied,
                );
                parameters[j] = next;
                first[j] = moment;
                second[j] = square;
            }
        }
        parameters
    }

    /// Bound on the deviation between [`Moments::advance`] and the hand
    /// reference. The measured maximum over the cases of
    /// `the_adamw_step_matches_a_two_step_hand_reference` and
    /// `the_adamw_decay_is_decoupled_from_the_gradient` is 0.000000e0 — the
    /// two agree bit-for-bit there — so this leaves room for the rounding a
    /// different association of the same formulas would introduce.
    const ADAMW_REFERENCE_TOLERANCE: f64 = 1e-14;

    /// Parameters, per-step gradients and rate the AdamW hand references run
    /// on: three entries of differing sign and magnitude, so the adaptive
    /// normalization is not reading one scale.
    fn adamw_case() -> (Vec<f64>, Vec<Vec<f64>>, f64) {
        (
            vec![0.7, -0.2, 1.3],
            vec![
                vec![0.4, -1.1, 0.02],
                vec![-0.3, -0.9, 0.5],
                vec![0.15, -0.4, -0.25],
            ],
            0.05,
        )
    }

    /// Runs [`Moments::advance`] over `gradients` at `rate`, returning the
    /// parameter row it leaves.
    fn adamw_advance_run(
        start: &[f64],
        gradients: &[Vec<f64>],
        rate: f64,
        settings: AdamW,
    ) -> DMatrix<f64> {
        let width = start.len();
        let mut moments = Moments::zeros(1, width);
        let mut parameter = DMatrix::from_row_slice(1, width, start);
        for row in gradients {
            let gradient = DMatrix::from_row_slice(1, width, row);
            let delta = moments.advance(&parameter, &gradient, rate, settings);
            parameter -= delta;
        }
        parameter
    }

    /// Two decoupled AdamW updates land where the §B.3 formulas put them,
    /// computed in the test from the moments, their bias correction, the
    /// adaptive step and the decay. The parameters are asserted to have moved
    /// by more than the tolerance the agreement is asserted at, so the match
    /// is not one between two unmoved rows.
    #[test]
    fn the_adamw_step_matches_a_two_step_hand_reference() {
        let settings = AdamW::default();
        let (start, gradients, rate) = adamw_case();
        let steps = &gradients[..2];

        let observed = adamw_advance_run(&start, steps, rate, settings);
        let expected = adamw_reference_run(&start, steps, rate, settings, false);

        let mut worst = 0.0_f64;
        let mut smallest_move = f64::INFINITY;
        for (j, &reference) in expected.iter().enumerate() {
            let deviation = (observed[(0, j)] - reference).abs();
            worst = worst.max(deviation);
            smallest_move = smallest_move.min((observed[(0, j)] - start[j]).abs());
            assert!(
                deviation < ADAMW_REFERENCE_TOLERANCE,
                "entry {j} is {:.17e} after two updates, the hand reference puts it at \
                 {reference:.17e} (deviation {deviation:.6e}, tolerance \
                 {ADAMW_REFERENCE_TOLERANCE:e})",
                observed[(0, j)]
            );
        }
        println!(
            "the_adamw_step_matches_a_two_step_hand_reference: max deviation {worst:.6e}, \
             smallest entry movement {smallest_move:.6e}, reference {expected:?}"
        );
        assert!(
            smallest_move > ADAMW_REFERENCE_TOLERANCE * 1e6,
            "the least-moved entry travelled {smallest_move:.6e} over the two updates, so the \
             agreement above would hold for an optimizer that does nothing"
        );
    }

    /// The decay reaches the parameter without passing through the gradient:
    /// [`Moments::advance`] matches the decoupled reference, while the coupled
    /// alternative — the same optimizer with the decay folded into the gradient
    /// and the decay factor dropped — lands measurably elsewhere, at the §B.3
    /// decay 0.01 and at a larger one.
    #[test]
    fn the_adamw_decay_is_decoupled_from_the_gradient() {
        let (start, gradients, rate) = adamw_case();
        for weight_decay in [0.01, 0.25] {
            let settings = AdamW {
                weight_decay,
                ..AdamW::default()
            };
            let observed = adamw_advance_run(&start, &gradients, rate, settings);
            let decoupled = adamw_reference_run(&start, &gradients, rate, settings, false);
            let coupled = adamw_reference_run(&start, &gradients, rate, settings, true);

            let mut worst = 0.0_f64;
            let mut separation = f64::INFINITY;
            for (j, &reference) in decoupled.iter().enumerate() {
                worst = worst.max((observed[(0, j)] - reference).abs());
                separation = separation.min((reference - coupled[j]).abs());
            }
            println!(
                "the_adamw_decay_is_decoupled_from_the_gradient: wd = {weight_decay}, max \
                 deviation from the decoupled reference {worst:.6e}, least separation from the \
                 coupled one {separation:.6e}"
            );
            assert!(
                worst < ADAMW_REFERENCE_TOLERANCE,
                "wd = {weight_decay}: the implementation deviates from the decoupled reference \
                 by {worst:.6e}, tolerance {ADAMW_REFERENCE_TOLERANCE:e}"
            );
            assert!(
                separation > ADAMW_REFERENCE_TOLERANCE * 1e4,
                "wd = {weight_decay}: the coupled and decoupled references differ by only \
                 {separation:.6e}, so the agreement above would hold for either path"
            );
        }
    }

    /// The §B.3 schedule at its four named points, from the closed form: zero
    /// at step 0, the peak at the warm-up count (the first cosine step), half
    /// the peak halfway through the cosine phase, and zero at the budget. The
    /// rate is asserted non-decreasing over the warm-up and non-increasing
    /// over the cosine phase, so a swapped branch does not pass on the
    /// endpoints alone, and the warm-up lengths of the three budgets the §B.3
    /// sweep runs are named. Steps past the budget pin at zero, and two
    /// budgets whose warm-up consumes them whole pin the empty-cosine branch
    /// at the peak.
    #[test]
    fn the_warmup_cosine_schedule_matches_its_closed_form() {
        let fraction = AdamW::default().warmup_fraction;
        for (budget, expected_warmup) in [(1_200_usize, 60_usize), (200, 10), (50, 3)] {
            assert_eq!(
                warmup_steps(budget, fraction),
                expected_warmup,
                "a {fraction} warm-up of {budget} steps is {} steps, expected {expected_warmup}",
                warmup_steps(budget, fraction)
            );
        }

        let peak = 0.01;
        let budget = 200_usize;
        let warmup = warmup_steps(budget, fraction);
        let midpoint = warmup + (budget - warmup) / 2;
        let named: [(&str, usize, f64); 4] = [
            ("step 0", 0, 0.0),
            ("warm-up end", warmup, peak),
            ("cosine midpoint", midpoint, peak * 0.5),
            ("budget", budget, 0.0),
        ];
        for (label, step, expected) in named {
            let rate = scheduled_rate(step, budget, peak, fraction);
            println!(
                "the_warmup_cosine_schedule_matches_its_closed_form: {label} (step {step}) is \
                 {rate:.17e}, expected {expected:.17e}"
            );
            assert!(
                (rate - expected).abs() < 1e-15,
                "{label} (step {step} of {budget}) is {rate:.17e}, the closed form gives \
                 {expected:.17e}"
            );
        }

        let mut previous = scheduled_rate(0, budget, peak, fraction);
        for step in 1..=warmup {
            let rate = scheduled_rate(step, budget, peak, fraction);
            assert!(
                rate >= previous,
                "the warm-up fell from {previous:.9e} at step {} to {rate:.9e} at step {step}",
                step - 1
            );
            previous = rate;
        }
        for step in warmup + 1..=budget {
            let rate = scheduled_rate(step, budget, peak, fraction);
            assert!(
                rate <= previous,
                "the cosine phase rose from {previous:.9e} at step {} to {rate:.9e} at step \
                 {step}",
                step - 1
            );
            previous = rate;
        }
        let last = scheduled_rate(budget - 1, budget, peak, fraction);
        assert!(
            last > 0.0 && last < peak,
            "the last applied step's rate is {last:.9e}, outside (0, {peak})"
        );

        let beyond = scheduled_rate(budget + 7, budget, peak, fraction);
        assert!(
            beyond.abs() < 1e-15,
            "step {} past the budget gives {beyond:.9e}, expected 0",
            budget + 7
        );
        for (whole_budget, whole_fraction) in [(1_usize, fraction), (4, 0.9)] {
            let warmup = warmup_steps(whole_budget, whole_fraction);
            assert_eq!(
                warmup, whole_budget,
                "budget {whole_budget} at fraction {whole_fraction} leaves a cosine phase, so \
                 this case no longer probes the empty-cosine branch"
            );
            for step in [whole_budget, whole_budget + 3] {
                let rate = scheduled_rate(step, whole_budget, peak, whole_fraction);
                assert!(
                    (rate - peak).abs() < 1e-15,
                    "with the warm-up consuming the whole budget, step {step} of \
                     {whole_budget} gives {rate:.17e}, expected the peak {peak:.17e}"
                );
            }
        }
    }

    /// Applied updates the weight-rate pins run.
    const RATIO_STEPS: usize = 6;

    /// Transition parameters at the production width: `ratio` as ρ, `init` as
    /// W's initializer, and `optimizer` over [`RATIO_STEPS`] updates at
    /// η = 0.01 with the tolerance held below anything the runs reach.
    fn transition_params(init: WeightInit, ratio: f64, optimizer: Optimizer) -> Params {
        Params {
            learning_rate: 0.01,
            max_steps: RATIO_STEPS,
            tolerance: 1e-300,
            regime: Regime::LearnableEmbedding,
            weight_init: init,
            weight_rate_ratio: ratio,
            optimizer,
            ..Params::default()
        }
    }

    /// ρ = 0 leaves W bit-identical to its initialization over a whole run
    /// under both optimizers, while E moves; ρ = 1 moves W, so the identity is
    /// a property of the ratio rather than of a run that trains nothing.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity of a frozen block, not numeric closeness"
    )]
    fn a_zero_weight_rate_leaves_the_weight_at_its_initialization() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        for optimizer in [
            Optimizer::GradientDescent,
            Optimizer::AdamW(AdamW::default()),
        ] {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let frozen = transition_params(WeightInit::Gaussian, 0.0, optimizer);
            let initial = system
                .initial_parameters(&frozen, SEED)
                .expect("initial_parameters");

            let paths = (TempPath::new("rho0-history"), TempPath::new("rho0-cosines"));
            let outputs = Outputs {
                history: paths.0.path(),
                cosines: paths.1.path(),
            };
            let held = run(&graph, &frozen, SEED, &outputs, || false).expect("run");
            assert_eq!(
                held.steps(),
                RATIO_STEPS,
                "{optimizer:?}: the run applied {} updates, expected {RATIO_STEPS}",
                held.steps()
            );

            for (index, (left, right)) in held
                .parameters()
                .weight()
                .iter()
                .zip(initial.weight().iter())
                .enumerate()
            {
                assert!(
                    left == right,
                    "{optimizer:?}: entry {index} of W is {left:e} after {RATIO_STEPS} updates \
                     at rho = 0, {right:e} at the draw"
                );
            }
            let embedding_move = (held.parameters().embedding() - initial.embedding()).amax();
            println!(
                "a_zero_weight_rate_leaves_the_weight_at_its_initialization: {optimizer:?}, \
                 max |ΔE| {embedding_move:.6e}"
            );
            assert!(
                embedding_move > 1e-9,
                "{optimizer:?}: E moved by {embedding_move:.6e} at rho = 0, so the W identity \
                 above would hold for a run that trains nothing"
            );

            let moving = transition_params(WeightInit::Gaussian, 1.0, optimizer);
            let other = (TempPath::new("rho1-history"), TempPath::new("rho1-cosines"));
            let moved = run(
                &graph,
                &moving,
                SEED,
                &Outputs {
                    history: other.0.path(),
                    cosines: other.1.path(),
                },
                || false,
            )
            .expect("run");
            let weight_move = (moved.parameters().weight() - initial.weight()).amax();
            assert!(
                weight_move > 1e-9,
                "{optimizer:?}: W moved by only {weight_move:.6e} at rho = 1, so the rho = 0 \
                 identity above would hold at every ratio"
            );
        }
    }

    /// ρ = 1/2 puts W exactly where subtracting half of η times ∂L/∂W puts it,
    /// against a reference built from the public gradient API, and away from
    /// where ρ = 1 puts it.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity against a hand-scaled reference, not numeric closeness"
    )]
    fn a_half_weight_rate_matches_a_hand_scaled_reference_step() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&graph).expect("TinyNn::new");
        let params = Params {
            max_steps: 1,
            ..transition_params(WeightInit::Gaussian, 0.5, Optimizer::GradientDescent)
        };
        let initial = system
            .initial_parameters(&params, SEED)
            .expect("initial_parameters");
        let gradients = system
            .gradients(&initial, params.activation, params.regime)
            .expect("gradients");
        let expected = initial.weight()
            - gradients.weight() * (params.learning_rate * params.weight_rate_ratio);

        let paths = (TempPath::new("rho-half-h"), TempPath::new("rho-half-c"));
        let run = run(
            &graph,
            &params,
            SEED,
            &Outputs {
                history: paths.0.path(),
                cosines: paths.1.path(),
            },
            || false,
        )
        .expect("run");
        assert_eq!(run.steps(), 1, "the run applied {} updates", run.steps());

        for (index, (left, right)) in run
            .parameters()
            .weight()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                left == right,
                "entry {index} of W is {left:e} after the run's step, {right:e} in the \
                 half-rate reference"
            );
        }
        let full = initial.weight() - gradients.weight() * params.learning_rate;
        let separation = (&expected - &full).amax();
        println!(
            "a_half_weight_rate_matches_a_hand_scaled_reference_step: the half-rate and \
             full-rate references differ by {separation:.6e}"
        );
        assert!(
            separation > 1e-9,
            "the half-rate and full-rate references differ by only {separation:.6e}, so the \
             match above would hold at either ratio"
        );
    }

    /// Applied updates the AdamW ρ replay takes.
    const ADAMW_RATIO_STEPS: usize = 8;

    /// Under AdamW, ρ enters through W's rate: each step's W movement is
    /// [`Moments::advance`] at ρ times the scheduled rate, pinned at ρ = 1/2
    /// by a replay through the public gradient API and the moment state
    /// directly. A ρ applied to the returned movement instead would land the
    /// decoupled decay term measurably elsewhere at any ρ strictly between 0
    /// and 1.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity against the rate-scaled replay, not numeric closeness"
    )]
    fn the_adamw_weight_rate_enters_through_the_rate() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&graph).expect("TinyNn::new");
        let settings = AdamW::default();
        let params = Params {
            max_steps: ADAMW_RATIO_STEPS,
            ..transition_params(WeightInit::Gaussian, 0.5, Optimizer::AdamW(settings))
        };

        let paths = (TempPath::new("adamw-rho-h"), TempPath::new("adamw-rho-c"));
        let run = run(
            &graph,
            &params,
            SEED,
            &Outputs {
                history: paths.0.path(),
                cosines: paths.1.path(),
            },
            || false,
        )
        .expect("run");
        assert_eq!(
            run.steps(),
            ADAMW_RATIO_STEPS,
            "the run applied {} updates",
            run.steps()
        );

        let mut replay = system
            .initial_parameters(&params, SEED)
            .expect("initial_parameters");
        let mut weight_moments = Moments::zeros(params.width, params.width);
        let mut embedding_moments = Moments::zeros(graph.order(), params.width);
        for step in 0..ADAMW_RATIO_STEPS {
            let gradients = system
                .gradients(&replay, params.activation, params.regime)
                .expect("gradients");
            let rate = scheduled_rate(
                step,
                params.max_steps,
                params.learning_rate,
                settings.warmup_fraction,
            );
            let weight_move = weight_moments.advance(
                &replay.weight,
                gradients.weight(),
                rate * params.weight_rate_ratio,
                settings,
            );
            let embedding_move = embedding_moments.advance(
                &replay.embedding,
                gradients
                    .embedding()
                    .expect("a learnable-embedding run carries an embedding gradient"),
                rate,
                settings,
            );
            replay.weight -= weight_move;
            replay.embedding -= embedding_move;
        }

        for (index, (observed, expected)) in run
            .parameters()
            .weight()
            .iter()
            .zip(replay.weight().iter())
            .enumerate()
        {
            assert!(
                observed == expected,
                "entry {index} of W is {observed:e} after the run, {expected:e} in the \
                 rate-scaled replay"
            );
        }
        assert!(
            run.parameters().embedding() == replay.embedding(),
            "the replayed embedding diverges from the run's"
        );
    }

    /// Applied updates the identity corner replays.
    const IDENTITY_CORNER_STEPS: usize = 6;

    /// The (identity W, ρ = 0) corner is dynamically Tier 1 for a whole short
    /// run, not only at the draw: W stays bit-identical to the identity at
    /// every step, and at each of them −∂L/∂E is Lemma 6's ascent direction CV
    /// — the premise
    /// `the_identity_weight_reproduces_the_tier1_ascent_direction` pins at
    /// step 0. The hand replay's final parameters are asserted bit-identical to
    /// [`run`]'s, so the steps checked are the ones the run took.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claims are bit-identity of a frozen block and of a replay, not numeric closeness"
    )]
    fn the_identity_corner_keeps_the_tier1_ascent_direction_along_the_run() {
        let width = 12;
        let identity = DMatrix::<f64>::identity(width, width);
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let tier1 = Node2Vec::new(&graph).expect("Node2Vec::new");
            let params = Params {
                width,
                embedding_sigma: 0.5,
                max_steps: IDENTITY_CORNER_STEPS,
                ..transition_params(WeightInit::Identity, 0.0, Optimizer::GradientDescent)
            };

            let mut parameters = system
                .initial_parameters(&params, SEED)
                .expect("initial_parameters");
            let mut worst = 0.0_f64;
            let mut weakest = f64::INFINITY;
            for step in 0..=IDENTITY_CORNER_STEPS {
                for (index, (left, right)) in
                    parameters.weight().iter().zip(identity.iter()).enumerate()
                {
                    assert!(
                        left == right,
                        "{name}: entry {index} of W is {left:e} at step {step}, {right:e} in the \
                         identity it was initialized to"
                    );
                }
                let gradients = system
                    .gradients(&parameters, params.activation, params.regime)
                    .expect("gradients");
                let gradient = gradients
                    .embedding()
                    .expect("a learnable-embedding run carries an embedding gradient");
                let ascent = tier1.gradient(parameters.embedding()).expect("gradient");
                worst = worst.max((-gradient - &ascent).amax());
                weakest = weakest.min(ascent.amax());
                assert!(
                    (-gradient - &ascent).amax() < 1e-12,
                    "{name}: at step {step} the descent direction −∂L/∂E deviates from Lemma 6's \
                     CV by {:.6e}",
                    (-gradient - &ascent).amax()
                );
                if step < IDENTITY_CORNER_STEPS {
                    parameters.embedding -= gradient * params.learning_rate;
                }
            }
            println!(
                "the_identity_corner_keeps_the_tier1_ascent_direction_along_the_run: {name}, \
                 max deviation {worst:.6e} over {IDENTITY_CORNER_STEPS} steps, weakest \
                 max |CV| {weakest:.6e}"
            );
            assert!(
                weakest > 1e-3,
                "{name}: Lemma 6's CV falls to {weakest:.6e} along the run, so the agreement \
                 above would hold for a pair of near-zero matrices"
            );

            let paths = (TempPath::new("corner-h"), TempPath::new("corner-c"));
            let run = run(
                &graph,
                &params,
                SEED,
                &Outputs {
                    history: paths.0.path(),
                    cosines: paths.1.path(),
                },
                || false,
            )
            .expect("run");
            assert_eq!(
                run.steps(),
                IDENTITY_CORNER_STEPS,
                "{name}: the run applied {} updates, expected {IDENTITY_CORNER_STEPS}",
                run.steps()
            );
            for (block, left, right) in [
                ("E", run.parameters().embedding(), parameters.embedding()),
                ("W", run.parameters().weight(), parameters.weight()),
            ] {
                for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                    assert!(
                        left == right,
                        "{name}: entry {index} of {block} is {left:e} in the run and {right:e} \
                         in the replay the directions above were checked along"
                    );
                }
            }
        }
    }

    /// Applied updates the AdamW replay runs.
    const ADAMW_REPLAY_STEPS: usize = 12;

    /// Replays `run` through the public [`TinyNn::gradients`], the schedule and
    /// the moment state, asserting every step's loss matches the recorded one
    /// bit-for-bit and returning the parameters the replay ends at.
    fn adamw_public_gradient_replay(
        system: &TinyNn,
        params: &Params,
        seed: u64,
        settings: AdamW,
        run: &Run,
    ) -> Parameters {
        let mut parameters = system
            .initial_parameters(params, seed)
            .expect("initial_parameters");
        let mut weight = Moments::zeros(params.width, params.width);
        let mut embedding = Moments::zeros(system.order(), params.width);
        for step in 0..=params.max_steps {
            let replayed = system.loss(&parameters, params.activation).expect("loss");
            let recorded = run.records()[step].loss();
            assert!(
                replayed.to_bits() == recorded.to_bits(),
                "the replay diverges at step {step}: loss {replayed:e} against the recorded \
                 {recorded:e}, so the run consumed a gradient this path does not"
            );
            if step == params.max_steps {
                break;
            }
            let gradients = system
                .gradients(&parameters, params.activation, params.regime)
                .expect("gradients");
            let rate = scheduled_rate(
                step,
                params.max_steps,
                params.learning_rate,
                settings.warmup_fraction,
            );
            let weight_delta = weight.advance(
                parameters.weight(),
                gradients.weight(),
                rate * params.weight_rate_ratio,
                settings,
            );
            let embedding_delta = embedding.advance(
                parameters.embedding(),
                gradients
                    .embedding()
                    .expect("a learnable-embedding run carries an embedding gradient"),
                rate,
                settings,
            );
            parameters.weight -= weight_delta;
            parameters.embedding -= embedding_delta;
        }
        parameters
    }

    /// The AdamW path consumes the gradients the finite-difference pins cover
    /// and no others: replaying a run through the public
    /// [`TinyNn::gradients`] — the entry point
    /// `gradients_match_central_differences_on_every_d_graph` probes at
    /// m = 16 — plus [`scheduled_rate`] and [`Moments::advance`] reproduces
    /// every recorded loss and the final parameters bit-for-bit. The run is
    /// asserted to reach its step limit, so the schedule's zero-rate first step
    /// is not read as convergence.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity of a replayed trajectory, not numeric closeness"
    )]
    fn the_adamw_path_consumes_the_finite_difference_checked_gradients() {
        let settings = AdamW::default();
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let params = Params {
                learning_rate: 0.01,
                max_steps: ADAMW_REPLAY_STEPS,
                optimizer: Optimizer::AdamW(settings),
                ..fd_params(16, 0.35, Activation::Linear)
            };

            let paths = (TempPath::new("adamw-h"), TempPath::new("adamw-c"));
            let run = run(
                &graph,
                &params,
                SEED,
                &Outputs {
                    history: paths.0.path(),
                    cosines: paths.1.path(),
                },
                || false,
            )
            .expect("run");
            assert_eq!(
                run.steps(),
                ADAMW_REPLAY_STEPS,
                "{name}: the run applied {} updates, expected the {ADAMW_REPLAY_STEPS}-step \
                 limit; a zero-rate warm-up step read as convergence would stop it at 0",
                run.steps()
            );
            assert_eq!(
                run.stop_reason(),
                StopReason::StepLimit,
                "{name}: the run stopped for {:?}, expected the step limit",
                run.stop_reason()
            );

            let parameters = adamw_public_gradient_replay(&system, &params, SEED, settings, &run);

            for (block, left, right) in [
                ("E", run.parameters().embedding(), parameters.embedding()),
                ("W", run.parameters().weight(), parameters.weight()),
            ] {
                for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                    assert!(
                        left == right,
                        "{name}: entry {index} of {block} is {left:e} in the run and {right:e} \
                         in the public-gradient replay"
                    );
                }
            }
            let moved = (run.parameters().embedding()
                - system
                    .initial_parameters(&params, SEED)
                    .expect("initial_parameters")
                    .embedding())
            .amax();
            println!(
                "the_adamw_path_consumes_the_finite_difference_checked_gradients: {name}, \
                 max |ΔE| over {ADAMW_REPLAY_STEPS} AdamW steps {moved:.6e}"
            );
            assert!(
                moved > 1e-6,
                "{name}: E moved {moved:.6e} over the run, so the replay agreement above would \
                 hold for a trajectory that never left its draw"
            );
        }
    }

    /// The geometry stop ends the run at the first recorded step meeting its
    /// threshold. `TinyNn::fiedler_alignment` documents a range of [0, 1], so a
    /// threshold of 0 is met by the initial record and one of 2 by none: the
    /// first run stops at step 0 with [`StopReason::Aligned`], the second
    /// spends its whole budget and stops at the step limit.
    #[test]
    fn the_alignment_stop_ends_the_run_at_the_first_step_that_meets_it() {
        let budget = 4;
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let base = transition_params(WeightInit::Gaussian, 1.0, Optimizer::GradientDescent);
            let draw = system
                .initial_parameters(&base, SEED)
                .expect("initial_parameters");
            let initial_alignment = system
                .fiedler_alignment(draw.embedding())
                .expect("fiedler_alignment");

            for (label, threshold, steps, expected, expected_outcome) in [
                (
                    "met at once",
                    0.0,
                    0_usize,
                    StopReason::Aligned,
                    Outcome::Stopped,
                ),
                (
                    "met exactly at the draw's own alignment",
                    initial_alignment,
                    0,
                    StopReason::Aligned,
                    Outcome::Stopped,
                ),
                (
                    "never met",
                    1.0,
                    budget,
                    StopReason::StepLimit,
                    Outcome::StepLimit,
                ),
            ] {
                let params = Params {
                    alignment_stop: Some(threshold),
                    max_steps: budget,
                    ..transition_params(WeightInit::Gaussian, 1.0, Optimizer::GradientDescent)
                };
                let paths = (TempPath::new("stop-h"), TempPath::new("stop-c"));
                let run = run(
                    &graph,
                    &params,
                    SEED,
                    &Outputs {
                        history: paths.0.path(),
                        cosines: paths.1.path(),
                    },
                    || false,
                )
                .expect("run");
                assert_eq!(
                    run.stop_reason(),
                    expected,
                    "{name} {label}: the run stopped for {:?} after {} steps, expected \
                     {expected:?}",
                    run.stop_reason(),
                    run.steps()
                );
                assert_eq!(
                    run.outcome(),
                    expected_outcome,
                    "{name} {label}: the run's outcome is {:?}, expected {expected_outcome:?}",
                    run.outcome()
                );
                assert_eq!(
                    run.steps(),
                    steps,
                    "{name} {label}: the run applied {} updates, expected {steps}",
                    run.steps()
                );
                assert_eq!(
                    run.records().len(),
                    steps + 1,
                    "{name} {label}: the run recorded {} steps, expected {}",
                    run.records().len(),
                    steps + 1
                );
            }
        }
    }

    /// Applied updates the determinism pins compare.
    const TRANSITION_TRAJECTORY_STEPS: usize = 10;

    /// Runs `params` at `seed` and at `seed + 1`, asserting the same seed
    /// reproduces the parameters, the records and both CSVs bit for bit and
    /// that the other seed does not.
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity of a deterministic trajectory, not numeric closeness"
    )]
    fn assert_trajectory_is_reproducible(label: &str, graph: &Graph, params: &Params, seed: u64) {
        let paths: Vec<(TempPath, TempPath)> = (0..3)
            .map(|index| {
                (
                    TempPath::new(&format!("{label}-{index}-h")),
                    TempPath::new(&format!("{label}-{index}-c")),
                )
            })
            .collect();
        let runs: Vec<Run> = [seed, seed, seed + 1]
            .into_iter()
            .zip(&paths)
            .map(|(seed, paths)| {
                run(
                    graph,
                    params,
                    seed,
                    &Outputs {
                        history: paths.0.path(),
                        cosines: paths.1.path(),
                    },
                    || false,
                )
                .expect("run")
            })
            .collect();

        assert_eq!(
            runs[0].steps(),
            params.max_steps,
            "{label}: ran {} steps, expected the {}-step limit",
            runs[0].steps(),
            params.max_steps
        );
        for (block, left, right) in [
            (
                "E",
                runs[0].parameters().embedding(),
                runs[1].parameters().embedding(),
            ),
            (
                "W",
                runs[0].parameters().weight(),
                runs[1].parameters().weight(),
            ),
        ] {
            for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                assert!(
                    left == right,
                    "{label}: entry {index} of {block} is {left:e} on the first run and \
                     {right:e} on the second, at seed {seed}"
                );
            }
        }
        assert_eq!(
            runs[0].records(),
            runs[1].records(),
            "{label}: the two same-seed runs recorded different histories"
        );
        for (file, first, second) in [
            ("history", paths[0].0.path(), paths[1].0.path()),
            ("cosine", paths[0].1.path(), paths[1].1.path()),
        ] {
            assert_eq!(
                std::fs::read(first).expect("read CSV"),
                std::fs::read(second).expect("read CSV"),
                "{label}: the two same-seed runs wrote different {file} CSVs"
            );
        }

        let deviation =
            (runs[0].parameters().embedding() - runs[2].parameters().embedding()).amax();
        assert!(
            deviation > 0.0,
            "{label}: seeds {seed} and {} produced an identical E (max |Δ| = {deviation:e}), so \
             the comparison above would hold for any seed",
            seed + 1
        );
    }

    /// A W-sweep configuration at the same seed reproduces its trajectory bit
    /// for bit, and a different seed does not.
    #[test]
    fn the_same_seed_reproduces_a_w_sweep_trajectory_bit_for_bit() {
        let params = Params {
            max_steps: TRANSITION_TRAJECTORY_STEPS,
            ..transition_params(WeightInit::Identity, 0.5, Optimizer::GradientDescent)
        };
        for (name, graph) in d_graphs() {
            assert_trajectory_is_reproducible(name, &graph, &params, SEED);
        }
    }

    /// An AdamW configuration at the same seed reproduces its trajectory bit
    /// for bit, and a different seed does not.
    #[test]
    fn the_same_seed_reproduces_an_adamw_trajectory_bit_for_bit() {
        let params = Params {
            max_steps: TRANSITION_TRAJECTORY_STEPS,
            ..transition_params(
                WeightInit::Gaussian,
                1.0,
                Optimizer::AdamW(AdamW::default()),
            )
        };
        for (name, graph) in d_graphs() {
            assert_trajectory_is_reproducible(name, &graph, &params, SEED);
        }
    }

    /// A `Some` geometry-stop threshold outside [0, 1] — the range
    /// [`TinyNn::fiedler_alignment`] takes values in — is rejected as
    /// [`Error::RunParameterNotAFraction`]; the boundary values 0, 0.75 and 1
    /// pass.
    #[test]
    fn the_alignment_stop_threshold_is_validated() {
        let base = Params::default();
        for value in [-1.0, f64::NAN, f64::INFINITY, 2.0] {
            let params = Params {
                alignment_stop: Some(value),
                ..base
            };
            match params.validate() {
                Err(Error::RunParameterNotAFraction { parameter, .. }) => {
                    assert_eq!(
                        parameter, "alignment_stop",
                        "rejected parameter {parameter:?} for alignment_stop {value}"
                    );
                }
                other => panic!(
                    "expected RunParameterNotAFraction for alignment_stop {value}, got {other:?}"
                ),
            }
        }
        for value in [0.0, 0.75, 1.0] {
            let params = Params {
                alignment_stop: Some(value),
                ..base
            };
            assert!(
                params.validate().is_ok(),
                "alignment_stop {value} is inside [0, 1] and was rejected"
            );
        }
    }

    /// The transition knobs come back as typed errors naming the field.
    #[test]
    fn the_transition_parameters_reject_degenerate_values() {
        let base = Params::default();
        for (parameter, value) in [("weight_rate_ratio", -0.5), ("weight_rate_ratio", f64::NAN)] {
            let params = Params {
                weight_rate_ratio: value,
                ..base
            };
            match params.validate() {
                Err(Error::NegativeRunParameter {
                    parameter: observed,
                    ..
                }) => {
                    assert_eq!(
                        observed, parameter,
                        "rejected parameter {observed:?}, expected {parameter:?}"
                    );
                }
                other => panic!("expected NegativeRunParameter for {parameter}, got {other:?}"),
            }
        }

        let decayed = Params {
            optimizer: Optimizer::AdamW(AdamW {
                weight_decay: -1.0,
                ..AdamW::default()
            }),
            ..base
        };
        match decayed.validate() {
            Err(Error::NegativeRunParameter { parameter, .. }) => {
                assert_eq!(
                    parameter, "weight_decay",
                    "rejected parameter {parameter:?}"
                );
            }
            other => panic!("expected NegativeRunParameter for weight_decay, got {other:?}"),
        }

        for (parameter, settings) in [
            (
                "beta1",
                AdamW {
                    beta1: 1.0,
                    ..AdamW::default()
                },
            ),
            (
                "beta2",
                AdamW {
                    beta2: -0.1,
                    ..AdamW::default()
                },
            ),
            (
                "warmup_fraction",
                AdamW {
                    warmup_fraction: f64::INFINITY,
                    ..AdamW::default()
                },
            ),
        ] {
            let params = Params {
                optimizer: Optimizer::AdamW(settings),
                ..base
            };
            match params.validate() {
                Err(Error::RunParameterNotAFraction {
                    parameter: observed,
                    ..
                }) => {
                    assert_eq!(
                        observed, parameter,
                        "rejected parameter {observed:?}, expected {parameter:?}"
                    );
                }
                other => panic!("expected RunParameterNotAFraction for {parameter}, got {other:?}"),
            }
        }

        let epsilon = Params {
            optimizer: Optimizer::AdamW(AdamW {
                epsilon: 0.0,
                ..AdamW::default()
            }),
            ..base
        };
        match epsilon.validate() {
            Err(Error::InvalidRunParameter { parameter, .. }) => {
                assert_eq!(parameter, "epsilon", "rejected parameter {parameter:?}");
            }
            other => panic!("expected InvalidRunParameter for epsilon, got {other:?}"),
        }

        assert!(
            base.validate().is_ok(),
            "the default parameters are rejected, so the errors above would fire on every input"
        );
    }
}
