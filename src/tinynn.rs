//! Tier 2: the TinyNN of §B.2.2 and its associative-vs-geometric competition.
//!
//! [`TinyNn`] holds the graph-derived quantities the passes reuse — the target
//! distribution D⁻¹A, the adjacency matrix, the degrees, and the vertex pairs
//! at each graph distance up to [`GEOMETRY_SHELLS`]. The model of decision D9
//! is one wide trainable W ∈ ℝ^{m×m} between a tied embedding/unembedding
//! E ∈ ℝ^{n×m}: the logit of v given u is E[u] W E[v]ᵀ, optionally with a GELU
//! on the hidden state E W. [`run`] drives full-batch gradient descent on the
//! degree-normalized cross-entropy of the bidirectional edge bigrams,
//! recording one [`StepRecord`] per step, streaming it to a CSV, and writing
//! the final node-node cosine matrix; [`Regime`] selects whether E is frozen
//! (§B.2.2's associative setting) or trained alongside W (the geometric one).
//! [`run`] takes an explicit output path and seed (decision D8,
//! `docs/2510.26745v2-poc-analysis.md` §8).

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — E[u] W E[v]ᵀ, W_ij, ‖ΔW‖_F — that the lint reads as unbackticked identifiers"
)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::DMatrix;

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::node2vec::{Outcome, cosine_similarity, second_factor_seed};
use crate::numerics::{gaussian_matrix, row_softmax, weighted_log_likelihood};
use crate::output::write_matrix_csv;
use crate::spectral::transition;

/// Number of graph-distance shells [`TinyNn::shell_means`] reports and
/// [`TinyNn::new`] requires non-empty. Shell k holds the vertex pairs at
/// distance k; the criterion itself compares the last two, so raising this
/// deepens the pair of shells [`GEOMETRY_MARGIN`] is calibrated against.
pub const GEOMETRY_SHELLS: usize = 3;

// Two vertices at distance 3 or more have no common neighbour, so an embedding
// whose rows are the adjacency rows gives the deepest shell cosine 0 — the
// reference `shell_margin` measures against
// (`an_adjacency_row_embedding_scores_zero_on_the_deepest_shell`).
const _: () = assert!(GEOMETRY_SHELLS >= 3);

/// Geometry margin at or above which an embedding counts as geometric for
/// [`Run::geometry_step`].
///
/// Measured on the four D-graphs at [`Params::default`]'s width: an
/// adjacency-row embedding scores 0
/// (`an_adjacency_row_embedding_scores_zero_on_the_deepest_shell`) and the
/// learnable runs of `the_learnable_run_forms_a_geometry_at_every_swept_rate`
/// peak between 0.096 and 0.183.
pub const GEOMETRY_MARGIN: f64 = 0.05;

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

/// Run parameters for gradient descent on the TinyNN cross-entropy.
///
/// [`Params::default`] carries decision D10's associative setting: the
/// §B.2.2 width m = 512, η = 0.1, a frozen embedding, and a linear hidden
/// layer.
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
        }
    }

    /// Rejects a zero width or step budget and a non-positive or non-finite
    /// σ, η, or tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimension`] for `width == 0`,
    /// [`Error::ZeroMaxSteps`] for `max_steps == 0`, and
    /// [`Error::InvalidRunParameter`] naming the first of `embedding_sigma`,
    /// `weight_sigma`, `learning_rate`, `tolerance` that is not positive and
    /// finite.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 {
            return Err(Error::InvalidDimension {
                dimension: self.width,
            });
        }
        if self.max_steps == 0 {
            return Err(Error::ZeroMaxSteps);
        }
        for (parameter, value) in [
            ("embedding_sigma", self.embedding_sigma),
            ("weight_sigma", self.weight_sigma),
            ("learning_rate", self.learning_rate),
            ("tolerance", self.tolerance),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidRunParameter { parameter, value });
            }
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
/// Construction computes the target distribution D⁻¹A, the degrees, and the
/// vertex pairs at each of the first [`GEOMETRY_SHELLS`] graph distances once;
/// every forward, backward, and metric evaluation reuses them.
#[derive(Debug, Clone)]
pub struct TinyNn {
    order: usize,
    walk: DMatrix<f64>,
    adjacency: DMatrix<f64>,
    degrees: Vec<usize>,
    shells: Vec<Vec<(usize, usize)>>,
}

impl TinyNn {
    /// Builds the system for `graph`.
    ///
    /// # Errors
    ///
    /// Propagates [`transition`]'s [`Error::IsolatedVertex`] and returns
    /// [`Error::InsufficientDistanceShells`] when `graph` has no vertex pair
    /// at some distance below [`GEOMETRY_SHELLS`].
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
        let shells = distance_shells(&adjacency, order)?;
        Ok(Self {
            order,
            walk,
            adjacency,
            degrees,
            shells,
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

    /// The unordered vertex pairs at graph distance `distance`, for
    /// `distance` in `1..=GEOMETRY_SHELLS`; an empty slice outside that range.
    #[must_use]
    pub fn shell(&self, distance: usize) -> &[(usize, usize)] {
        self.shells
            .get(distance.wrapping_sub(1))
            .map_or(&[], Vec::as_slice)
    }

    /// Draws E from `seed` and W from [`second_factor_seed`] of it, both with
    /// `params`-scaled standard normals.
    ///
    /// # Errors
    ///
    /// Propagates [`Params::validate`]'s errors and
    /// [`gaussian_matrix`]'s [`Error::MatrixTooLarge`].
    pub fn initial_parameters(&self, params: &Params, seed: u64) -> Result<Parameters> {
        params.validate()?;
        let embedding = gaussian_matrix(self.order, params.width, params.embedding_sigma, seed)?;
        let weight = gaussian_matrix(
            params.width,
            params.width,
            params.weight_sigma,
            second_factor_seed(seed),
        )?;
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

        let weight = parameters.embedding.tr_mul(&pre_gradient);
        let embedding = match regime {
            Regime::FrozenEmbedding => None,
            Regime::LearnableEmbedding => Some(
                residual.tr_mul(&forward.hidden) + &pre_gradient * parameters.weight.transpose(),
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

    /// The mean cosine similarity over the vertex pairs at each distance
    /// `1..=GEOMETRY_SHELLS`, in that order.
    ///
    /// # Panics
    ///
    /// Panics if a shell is empty. [`TinyNn::new`] returns
    /// [`Error::InsufficientDistanceShells`] rather than build a system with
    /// one, so a `TinyNn` this crate constructed has none.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "pair counts are bounded by order², exact in f64 at this scale"
    )]
    pub fn shell_means(&self, cosines: &DMatrix<f64>) -> Vec<f64> {
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

    /// The geometry criterion: the deepest shell's mean cosine measured by its
    /// distance from zero, capped by its drop below the shell above it. An
    /// adjacency-row embedding scores zero
    /// (`an_adjacency_row_embedding_scores_zero_on_the_deepest_shell`).
    #[must_use]
    pub fn geometry_margin(&self, cosines: &DMatrix<f64>) -> f64 {
        shell_margin(&self.shell_means(cosines))
    }

    /// Rejects a parameter set whose embedding row count is not the vertex
    /// count.
    fn check_order(&self, parameters: &Parameters) -> Result<()> {
        if parameters.embedding.nrows() == self.order {
            Ok(())
        } else {
            Err(Error::EmbeddingOrderMismatch {
                rows: parameters.embedding.nrows(),
                order: self.order,
            })
        }
    }
}

/// The geometry margin of a shell-mean profile: the deepest shell's distance
/// from zero, capped by its drop below the shell above it. NaN for a profile
/// carrying a NaN or shorter than two entries, so such a profile fails every
/// threshold comparison.
fn shell_margin(shell_means: &[f64]) -> f64 {
    let [.., above, deepest] = shell_means else {
        return f64::NAN;
    };
    let separation = above - deepest;
    if deepest.is_nan() || separation.is_nan() {
        return f64::NAN;
    }
    deepest.abs().min(separation)
}

/// The unordered vertex pairs at each distance `1..=GEOMETRY_SHELLS`, by
/// breadth-first search from every vertex.
///
/// # Errors
///
/// Returns [`Error::InsufficientDistanceShells`] when one of those shells is
/// empty, reporting how many leading shells are populated.
fn distance_shells(adjacency: &DMatrix<f64>, order: usize) -> Result<Vec<Vec<(usize, usize)>>> {
    let mut shells: Vec<Vec<(usize, usize)>> = vec![Vec::new(); GEOMETRY_SHELLS];
    let mut distance = vec![usize::MAX; order];
    let mut queue = VecDeque::with_capacity(order);

    for source in 0..order {
        distance.fill(usize::MAX);
        distance[source] = 0;
        queue.clear();
        queue.push_back(source);
        while let Some(vertex) = queue.pop_front() {
            let next = distance[vertex] + 1;
            if next > GEOMETRY_SHELLS {
                continue;
            }
            for other in 0..order {
                if adjacency[(vertex, other)] > 0.0 && distance[other] == usize::MAX {
                    distance[other] = next;
                    queue.push_back(other);
                }
            }
        }
        for (other, &reached) in distance.iter().enumerate().skip(source + 1) {
            if (1..=GEOMETRY_SHELLS).contains(&reached) {
                shells[reached - 1].push((source, other));
            }
        }
    }

    let available = shells.iter().take_while(|pairs| !pairs.is_empty()).count();
    if available < GEOMETRY_SHELLS {
        return Err(Error::InsufficientDistanceShells { available });
    }
    Ok(shells)
}

/// One recorded step: the state after `step` applied updates, together with
/// the relative size of the update pending from it.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    step: usize,
    loss: f64,
    associative_score: f64,
    geometry_margin: f64,
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

    /// [`TinyNn::geometry_margin`] at this state.
    #[must_use]
    pub fn geometry_margin(&self) -> f64 {
        self.geometry_margin
    }

    /// (‖ΔW‖_F + ‖ΔE‖_F)/(‖W‖_F + ‖E‖_F) for the update pending from this
    /// state, the ΔE term being zero under [`Regime::FrozenEmbedding`].
    #[must_use]
    pub fn relative_update(&self) -> f64 {
        self.relative_update
    }

    /// The mean cosine similarity at each distance `1..=GEOMETRY_SHELLS`.
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
    outcome: Outcome,
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

    /// Why the step loop ended.
    #[must_use]
    pub fn outcome(&self) -> Outcome {
        self.outcome
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

    /// The first step whose geometry margin reaches `threshold`.
    #[must_use]
    pub fn geometry_step(&self, threshold: f64) -> Option<usize> {
        self.records
            .iter()
            .find(|record| record.geometry_margin >= threshold)
            .map(StepRecord::step)
    }

    /// The largest geometry margin over the recorded steps.
    #[must_use]
    pub fn peak_geometry_margin(&self) -> f64 {
        self.records
            .iter()
            .map(StepRecord::geometry_margin)
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

/// Runs full-batch gradient descent on `graph`, streaming one CSV row per
/// recorded step to `outputs.history` and writing the final node-node
/// cosine-similarity matrix to `outputs.cosines`.
///
/// Each step subtracts η times the gradient from every block
/// [`Params::regime`] trains, both updates read off the same pre-update
/// parameters, polling `should_stop` once per applied update. It stops on
/// convergence (relative update ≤ [`Params::tolerance`]), on reaching
/// [`Params::max_steps`], or on `should_stop` returning `true`; the CSV holds
/// a header row followed by one complete row per recorded step in each case.
///
/// # Errors
///
/// Propagates [`TinyNn::new`]'s and [`Params::validate`]'s errors and
/// [`Error::Io`] from creating or writing either file.
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
    // Recomputed where the embedding moves; a frozen-embedding run holds one
    // cosine matrix because its embedding never changes.
    let mut cosines = cosine_similarity(&parameters.embedding);

    let mut sink = BufWriter::new(File::create(outputs.history)?);
    write_header(&mut sink)?;

    let mut records = Vec::new();
    let mut steps = 0_usize;
    let outcome = loop {
        let forward = system.forward(&parameters, params.activation)?;
        let gradients =
            system.gradients_of(&parameters, params.activation, params.regime, &forward);
        let weight_update = gradients.weight() * params.learning_rate;
        let embedding_update = gradients
            .embedding()
            .map(|gradient| gradient * params.learning_rate);

        let moved = weight_update.norm() + embedding_update.as_ref().map_or(0.0, DMatrix::norm);
        let relative_update = moved / (parameters.weight.norm() + parameters.embedding.norm());
        let shell_means = system.shell_means(&cosines);

        let record = StepRecord {
            step: steps,
            loss: system.loss_of(&forward),
            associative_score: system.associative_score(&forward.probabilities),
            geometry_margin: shell_margin(&shell_means),
            relative_update,
            shell_means,
        };
        write_row(&mut sink, &record)?;
        records.push(record);

        if relative_update <= params.tolerance {
            break Outcome::Converged;
        }
        if steps >= params.max_steps {
            break Outcome::StepLimit;
        }
        // Polled before the update so that on every outcome the returned
        // state is the one the last record and CSV row describe.
        if should_stop() {
            break Outcome::Stopped;
        }

        parameters.weight -= &weight_update;
        if let Some(update) = &embedding_update {
            parameters.embedding -= update;
            cosines = cosine_similarity(&parameters.embedding);
        }
        steps += 1;
    };
    sink.flush()?;

    write_matrix_csv(outputs.cosines, &cosines)?;

    Ok(Run {
        parameters,
        records,
        outcome,
        steps,
    })
}

/// Writes the history header: the fixed columns followed by one
/// `shell_mean_k` per distance shell.
fn write_header<W: Write>(sink: &mut W) -> Result<()> {
    write!(
        sink,
        "step,loss,associative_score,geometry_margin,relative_update"
    )?;
    for distance in 1..=GEOMETRY_SHELLS {
        write!(sink, ",shell_mean_{distance}")?;
    }
    writeln!(sink)?;
    Ok(())
}

/// Writes one history row, each float in Rust's shortest round-tripping form.
fn write_row<W: Write>(sink: &mut W, record: &StepRecord) -> Result<()> {
    write!(
        sink,
        "{},{},{},{},{}",
        record.step,
        record.loss,
        record.associative_score,
        record.geometry_margin,
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
    use crate::node2vec::{self, Node2Vec};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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

    /// The geometry margin is zero on the adjacency matrix and positive on a
    /// cosine matrix that decays with distance, so the criterion reads the
    /// multi-hop structure rather than the edge set.
    #[test]
    fn the_geometry_margin_separates_multi_hop_structure_from_adjacency() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let order = graph.order();

            let adjacency_margin = system.geometry_margin(graph.adjacency());
            assert!(
                adjacency_margin.abs() < 1e-15,
                "{name}: the adjacency matrix scores {adjacency_margin:.6e}, expected 0"
            );

            let mut decaying = DMatrix::<f64>::identity(order, order);
            for distance in 1..=GEOMETRY_SHELLS {
                let value = 1.0 - 0.2 * distance as f64;
                for &(u, v) in system.shell(distance) {
                    decaying[(u, v)] = value;
                    decaying[(v, u)] = value;
                }
            }
            let decaying_margin = system.geometry_margin(&decaying);
            let expected = (1.0 - 0.2 * GEOMETRY_SHELLS as f64).abs().min(0.2);
            assert!(
                (decaying_margin - expected).abs() < 1e-12,
                "{name}: a cosine matrix decaying by 0.2 per shell scores \
                 {decaying_margin:.15}, expected {expected}"
            );
        }
    }

    /// The criterion on the profile shape the learnable runs actually reach: a
    /// negative deepest shell below a positive one above it, where the
    /// distance-from-zero branch binds rather than the separation cap. Each
    /// case names the branch it exercises and the value it expects.
    #[test]
    fn the_geometry_margin_reads_a_negative_deepest_shell() {
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
            // negative margin.
            ("inverted shells", [0.0, -0.40, -0.10], -0.30),
            // No structure at all.
            ("flat", [0.2, 0.2, 0.2], 0.0),
        ];

        for (label, means, expected) in cases {
            let margin = shell_margin(&means);
            assert!(
                (margin - expected).abs() < 1e-9,
                "{label}: shell means {means:?} score {margin:.9}, expected {expected:.9}"
            );
        }
    }

    /// The associative reference the criterion measures against: an embedding
    /// whose rows are the adjacency rows scores exactly zero on every D-graph,
    /// its deepest-shell pairs having no common neighbour. Its distance-2 mean
    /// is printed alongside, so the zero is visibly a property of the deepest
    /// shell rather than of a cosine matrix with no structure at all.
    #[test]
    fn an_adjacency_row_embedding_scores_zero_on_the_deepest_shell() {
        for (name, graph) in d_graphs() {
            let system = TinyNn::new(&graph).expect("TinyNn::new");
            let cosines = cosine_similarity(graph.adjacency());
            let means = system.shell_means(&cosines);
            let margin = system.geometry_margin(&cosines);

            println!("{name}: adjacency-row shell means {means:?}, margin {margin:.6}");
            assert!(
                means[GEOMETRY_SHELLS - 1].abs() < 1e-15,
                "{name}: the adjacency-row embedding's distance-{GEOMETRY_SHELLS} mean is \
                 {:.6e}, expected 0",
                means[GEOMETRY_SHELLS - 1]
            );
            assert!(
                margin.abs() < 1e-15,
                "{name}: the adjacency-row embedding scores {margin:.6e}, expected 0"
            );
            assert!(
                means[GEOMETRY_SHELLS - 2] > 0.1,
                "{name}: the adjacency-row embedding's distance-{} mean is {:.6}, so the zero \
                 above would hold for a cosine matrix with no shell structure at all",
                GEOMETRY_SHELLS - 1,
                means[GEOMETRY_SHELLS - 2]
            );
        }
    }

    /// A cosine matrix that left the finite range scores NaN, which fails
    /// every threshold comparison — a diverged run cannot report a geometry.
    #[test]
    fn a_non_finite_cosine_matrix_scores_nan() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&graph).expect("TinyNn::new");
        let broken = DMatrix::from_element(15, 15, f64::NAN);

        let margin = system.geometry_margin(&broken);
        assert!(
            margin.is_nan(),
            "a NaN cosine matrix scores {margin}, expected NaN"
        );
        assert!(
            margin.partial_cmp(&GEOMETRY_MARGIN).is_none(),
            "a NaN margin orders against the {GEOMETRY_MARGIN} threshold as {:?}",
            margin.partial_cmp(&GEOMETRY_MARGIN)
        );
    }

    /// Distance shell 1 is the edge set, and the 15-cycle has 15 pairs at each
    /// of the first three distances.
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
        }

        let cycle = Graph::cycle(15).expect("cycle(15)");
        let system = TinyNn::new(&cycle).expect("TinyNn::new");
        for distance in 1..=GEOMETRY_SHELLS {
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
    /// gets: above the step at which the criterion is met at that rate on the
    /// four D-graphs.
    const GEOMETRIC_SWEEP: [(f64, usize); 3] = [(0.001, 1_200), (0.01, 200), (0.1, 50)];

    /// Runs `params` on `graph` into temp files, printing the measurement.
    fn measured_run(label: &str, graph: &Graph, params: &Params) -> Run {
        let history = TempPath::new("history");
        let cosines = TempPath::new("cosines");
        let outputs = Outputs {
            history: history.path(),
            cosines: cosines.path(),
        };
        let started = Instant::now();
        let run = run(graph, params, SEED, &outputs, || false).expect("run");
        let last = run.last().expect("a run records its initial state");
        println!(
            "{label}: {:?}, outcome {:?}, {} steps, loss {:.6} (was {:.6}), \
             associative step {:?} (peak {:.6}, initial {:.6}), geometry step {:?} \
             (peak margin {:.6}), final shell means {:?}",
            started.elapsed(),
            run.outcome(),
            run.steps(),
            last.loss(),
            run.records()[0].loss(),
            run.associative_step(),
            run.peak_associative_score(),
            run.records()[0].associative_score(),
            run.geometry_step(GEOMETRY_MARGIN),
            run.peak_geometry_margin(),
            last.shell_means()
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
        );
        run
    }

    /// Figures 8 and 22 through the run API: on every D-graph and every swept
    /// learning rate the geometry criterion is met, and not at the draw. The
    /// step at which it is first met is printed per graph and rate.
    #[test]
    fn the_learnable_run_forms_a_geometry_at_every_swept_rate() {
        for (name, graph) in d_graphs() {
            for (learning_rate, budget) in GEOMETRIC_SWEEP {
                let params = Params {
                    learning_rate,
                    max_steps: budget,
                    regime: Regime::LearnableEmbedding,
                    ..Params::default()
                };
                let label = format!("{name} at eta = {learning_rate}");
                let run = measured_run(&label, &graph, &params);

                let step = run.geometry_step(GEOMETRY_MARGIN).unwrap_or_else(|| {
                    panic!(
                        "{label}: the geometry margin never reached {GEOMETRY_MARGIN} in \
                         {budget} steps; it peaked at {:.6}",
                        run.peak_geometry_margin()
                    )
                });
                assert!(
                    step > 0,
                    "{label}: the draw already meets the geometry criterion at margin {:.6}, \
                     so the step above measures nothing",
                    run.records()[0].geometry_margin()
                );
            }
        }
    }

    /// The frozen-embedding regime memorizes the edges while its embedding
    /// stays at the draw: the top-d score reaches its maximum and the shell
    /// profile is the one the seeded draw started with. §B.2.2 rests its
    /// associative reading on the first half; the second is a property of the
    /// regime — a frozen embedding cannot move — so the test pins that the
    /// draw itself is not already geometric, which is what makes the
    /// learnable runs' margins attributable to training.
    #[test]
    fn the_frozen_run_memorizes_without_moving_its_embedding() {
        for (name, graph) in d_graphs() {
            let params = Params {
                max_steps: ASSOCIATIVE_BUDGET,
                ..Params::default()
            };
            let run = measured_run(&format!("{name} frozen"), &graph, &params);

            assert!(
                run.peak_associative_score() >= 1.0 - FULL_MEMORIZATION_SLACK,
                "{name}: the frozen run peaked at a top-d score of {:.6} in \
                 {ASSOCIATIVE_BUDGET} steps, so the geometry null below is a run that learned \
                 nothing",
                run.peak_associative_score()
            );

            let first = run.records()[0].geometry_margin();
            let last = run
                .last()
                .expect("a run records its initial state")
                .geometry_margin();
            assert!(
                (first - last).abs() < 1e-12,
                "{name}: the frozen run's margin moved from {first:.12} to {last:.12}; a frozen \
                 embedding cannot change the cosine matrix"
            );
            assert!(
                first < GEOMETRY_MARGIN,
                "{name}: the seeded draw already scores {first:.6} against the \
                 {GEOMETRY_MARGIN} criterion, so the learnable runs' margins are not \
                 attributable to training"
            );
        }
    }

    /// The GELU variant carries the same two results on the 15-cycle: the
    /// frozen run memorizes the edges within Refutation 3c's two steps, and
    /// the learnable run at η = 0.01 forms a geometry after it. The measured
    /// steps are printed.
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
        let geometric_step = geometric.geometry_step(GEOMETRY_MARGIN).unwrap_or_else(|| {
            panic!(
                "cycle(15) gelu: the geometry margin never reached {GEOMETRY_MARGIN} in 200 \
                 steps; it peaked at {:.6}",
                geometric.peak_geometry_margin()
            )
        });
        assert!(
            geometric_step > associative_step,
            "cycle(15) gelu: the geometry criterion is met at step {geometric_step}, at or \
             before the associative step {associative_step}"
        );
    }
}
