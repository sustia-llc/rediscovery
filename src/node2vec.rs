//! Tier 1: the 1-hop, weight-tied `Node2Vec` system of Appendix F.
//!
//! [`Node2Vec`] holds the graph-derived quantities that Lemma 6's update
//! reuses at every step — W = D⁻¹A and W + Wᵀ — and exposes the Eq. 1
//! objective, the Eq. 2 probability matrix P = row_softmax(VVᵀ), and the
//! coefficient matrix C = (W − P) + (W − P)ᵀ. [`run_tied`] drives full-batch
//! gradient ascent ΔV = ηCV, recording one [`StepRecord`] per step and
//! streaming it to a CSV file; [`run_untied`] does the same for the
//! weight-untied variant V₁, V₂ of §4.4 and additionally writes each factor's
//! node-node cosine-similarity matrix. [`run_tied`] and [`run_untied`] take
//! an explicit output path and seed (decision D8,
//! `docs/2510.26745v2-poc-analysis.md` §8).

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — Σ_i, W_ij, ‖Vᵀe_i‖₂, V₁V₂ᵀ — that the lint reads as unbackticked identifiers"
)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

use nalgebra::DMatrix;

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::numerics::{gaussian_matrix, row_softmax, weighted_log_likelihood};
use crate::output::write_matrix_csv;
use crate::spectral::{Spectrum, symmetrize, transition};

/// Eigenvalue gap below which two eigenvalues of −L count as one degenerate
/// group, matching decision D6's closed-form eigenvalue tolerance.
pub const DEGENERACY_TOLERANCE: f64 = 1e-9;

/// Run parameters for gradient ascent on the Eq. 1 objective.
///
/// [`Params::default`] carries decision D7: m = 100, σ = 1.0, η = 0.01,
/// 10 000 steps, and a relative-update stop at 1e-8.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Embedding dimensionality m, the column count of V.
    pub dimension: usize,
    /// Standard deviation σ of the N(0, σ²) entries of V(0).
    pub sigma: f64,
    /// Ascent step size η.
    pub learning_rate: f64,
    /// Upper bound on applied updates.
    pub max_steps: usize,
    /// Value of ‖ΔV‖_F/‖V‖_F at or below which the run stops as converged.
    pub tolerance: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            dimension: 100,
            sigma: 1.0,
            learning_rate: 0.01,
            max_steps: 10_000,
            tolerance: 1e-8,
        }
    }
}

impl Params {
    /// Rejects a zero dimension and a non-positive or non-finite σ, η, or
    /// tolerance.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimension`] for `dimension == 0`,
    /// [`Error::ZeroMaxSteps`] for `max_steps == 0`, and
    /// [`Error::InvalidRunParameter`] naming the first of `sigma`,
    /// `learning_rate`, `tolerance` that is not positive and finite.
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 {
            return Err(Error::InvalidDimension {
                dimension: self.dimension,
            });
        }
        if self.max_steps == 0 {
            return Err(Error::ZeroMaxSteps);
        }
        for (parameter, value) in [
            ("sigma", self.sigma),
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

/// The weight-tied `Node2Vec` system for one graph.
///
/// Construction computes W = D⁻¹A and W + Wᵀ once; every objective,
/// probability, and coefficient evaluation reuses them.
#[derive(Debug, Clone)]
pub struct Node2Vec {
    order: usize,
    walk: DMatrix<f64>,
    walk_symmetric: DMatrix<f64>,
}

impl Node2Vec {
    /// Builds the system for `graph`.
    ///
    /// # Errors
    ///
    /// Propagates [`transition`]'s [`Error::IsolatedVertex`].
    pub fn new(graph: &Graph) -> Result<Self> {
        let walk = transition(graph)?;
        let walk_symmetric = symmetrize(&walk);
        Ok(Self {
            order: graph.order(),
            walk,
            walk_symmetric,
        })
    }

    /// The vertex count n, equal to the row count of a valid embedding.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// The row-normalized transition matrix W = D⁻¹A.
    #[must_use]
    pub fn walk(&self) -> &DMatrix<f64> {
        &self.walk
    }

    /// Draws V(0) ∈ ℝ^{n×m} with entries `sigma`-scaled standard normals from
    /// a `ChaCha20` stream keyed by `seed`.
    ///
    /// Entries are filled row by row from a fixed number of draws each, so a
    /// given `(seed, order, dimension, sigma)` yields bit-identical values.
    ///
    /// # Errors
    ///
    /// Propagates [`Params::validate`]'s errors and returns
    /// [`Error::EmbeddingTooLarge`] when `order * dimension` overflows
    /// `usize`.
    pub fn initial_embedding(&self, params: &Params, seed: u64) -> Result<DMatrix<f64>> {
        params.validate()?;
        self.order
            .checked_mul(params.dimension)
            .ok_or(Error::EmbeddingTooLarge {
                rows: self.order,
                columns: params.dimension,
            })?;

        gaussian_matrix(self.order, params.dimension, params.sigma, seed)
    }

    /// Computes P = row_softmax(VVᵀ) of Eq. 2, each row shifted by its
    /// maximum before exponentiation.
    ///
    /// The softmax denominator runs over every k, the self term k = i
    /// included, as §F.1 requires.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `embedding` does not
    /// have one row per vertex.
    pub fn probabilities(&self, embedding: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.check_order(embedding)?;
        Ok(row_softmax(&(embedding * embedding.transpose())))
    }

    /// Evaluates the Eq. 1 objective
    /// J = Σ_i (1/|nbr(i)|) Σ_{j∈nbr(i)} log p(i, j) at `embedding`, each
    /// row's log-partition computed after shifting by that row's maximum.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when `embedding` does not
    /// have one row per vertex.
    pub fn objective(&self, embedding: &DMatrix<f64>) -> Result<f64> {
        self.check_order(embedding)?;
        Ok(weighted_log_likelihood(
            &self.walk,
            &(embedding * embedding.transpose()),
        ))
    }

    /// Computes Lemma 6's coefficient matrix C = (W − P) + (W − P)ᵀ from a
    /// probability matrix.
    fn coefficient_from(&self, probabilities: &DMatrix<f64>) -> DMatrix<f64> {
        &self.walk_symmetric - symmetrize(probabilities)
    }

    /// Computes Lemma 6's coefficient matrix C = (W − P) + (W − P)ᵀ at
    /// `embedding`.
    ///
    /// # Errors
    ///
    /// Propagates [`Node2Vec::probabilities`]'s
    /// [`Error::EmbeddingOrderMismatch`].
    pub fn coefficient(&self, embedding: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let probabilities = self.probabilities(embedding)?;
        Ok(self.coefficient_from(&probabilities))
    }

    /// Computes the ascent direction CV of Lemma 6 at `embedding`.
    ///
    /// # Errors
    ///
    /// Propagates [`Node2Vec::coefficient`]'s
    /// [`Error::EmbeddingOrderMismatch`].
    pub fn gradient(&self, embedding: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Ok(self.coefficient(embedding)? * embedding)
    }

    /// Evaluates the weight-untied objective: the same degree-normalized
    /// cross-entropy of Eq. 1 with the logits VVᵀ replaced by V₁V₂ᵀ.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbeddingOrderMismatch`] when either factor lacks one
    /// row per vertex and [`Error::EmbeddingShapeMismatch`] when the two
    /// factors differ in shape.
    pub fn untied_objective(&self, first: &DMatrix<f64>, second: &DMatrix<f64>) -> Result<f64> {
        self.check_pair(first, second)?;
        Ok(weighted_log_likelihood(
            &self.walk,
            &(first * second.transpose()),
        ))
    }

    /// Computes P = row_softmax(V₁V₂ᵀ) for the weight-untied variant.
    ///
    /// # Errors
    ///
    /// Propagates [`Node2Vec::untied_objective`]'s shape errors.
    pub fn untied_probabilities(
        &self,
        first: &DMatrix<f64>,
        second: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>> {
        self.check_pair(first, second)?;
        Ok(row_softmax(&(first * second.transpose())))
    }

    /// Computes the weight-untied ascent directions
    /// ((W − P)V₂, (W − P)ᵀV₁), the gradients of
    /// [`Node2Vec::untied_objective`] in each factor at the given pair.
    ///
    /// # Errors
    ///
    /// Propagates [`Node2Vec::untied_probabilities`]'s shape errors.
    pub fn untied_gradients(
        &self,
        first: &DMatrix<f64>,
        second: &DMatrix<f64>,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>)> {
        let probabilities = self.untied_probabilities(first, second)?;
        let residual = &self.walk - &probabilities;
        Ok((&residual * second, residual.transpose() * first))
    }

    /// Rejects an embedding whose row count is not the vertex count.
    fn check_order(&self, embedding: &DMatrix<f64>) -> Result<()> {
        if embedding.nrows() == self.order {
            Ok(())
        } else {
            Err(Error::EmbeddingOrderMismatch {
                rows: embedding.nrows(),
                order: self.order,
            })
        }
    }

    /// Rejects a factor pair that is not two equally shaped embeddings.
    fn check_pair(&self, first: &DMatrix<f64>, second: &DMatrix<f64>) -> Result<()> {
        self.check_order(first)?;
        self.check_order(second)?;
        if first.ncols() == second.ncols() {
            Ok(())
        } else {
            Err(Error::EmbeddingShapeMismatch {
                columns: first.ncols(),
                other_columns: second.ncols(),
            })
        }
    }
}

/// The node-node cosine-similarity matrix of `embedding`, entry (i, j) being
/// the cosine of the angle between rows i and j.
///
/// Rows of zero norm have no direction; their entries are written as 0.
#[must_use]
pub fn cosine_similarity(embedding: &DMatrix<f64>) -> DMatrix<f64> {
    let norms: Vec<f64> = embedding.row_iter().map(|row| row.norm()).collect();
    let gram = embedding * embedding.transpose();
    DMatrix::from_fn(embedding.nrows(), embedding.nrows(), |i, j| {
        let scale = norms[i] * norms[j];
        if scale > 0.0 {
            gram[(i, j)] / scale
        } else {
            0.0
        }
    })
}

/// The Fiedler-like index range of `spectrum`: whole degenerate groups
/// starting below the top one, extending while the group's leading eigenvalue
/// stays within `spread` of the first one outside the top group.
/// [`DEGENERACY_TOLERANCE`] sets the grouping itself.
///
/// Footnote 9 of the paper takes Fiedler-like to mean the second-top
/// eigenvectors together with some of the subsequent ones without fixing how
/// many, so `spread` is the caller's choice; `spread = 0.0` gives exactly the
/// degenerate group below the top. On a spectrum with a single group the
/// range is that group.
#[must_use]
pub fn fiedler_like_range(spectrum: &Spectrum, spread: f64) -> Range<usize> {
    let groups = spectrum.degenerate_groups(DEGENERACY_TOLERANCE);
    if groups.len() < 2 {
        return groups[0].clone();
    }

    let start = groups[1].start;
    let leading = spectrum.eigenvalues()[start];
    let mut end = groups[1].end;
    for group in groups.iter().skip(2) {
        if (leading - spectrum.eigenvalues()[group.start]).abs() > spread {
            break;
        }
        end = group.end;
    }
    start..end
}

/// Fraction of the spectral range within which successive degenerate groups
/// still count as Fiedler-like. Zero — the degenerate group below the top,
/// and no more — is what [`run_tied`] and [`run_untied`] instrument with; a
/// caller studying a graph whose Fiedler-like structure spans several groups
/// passes its own spread to [`fiedler_like_range`].
pub const FIEDLER_SPREAD_FRACTION: f64 = 0.0;

/// The seed [`run_untied`] draws its second factor from, offset by the
/// golden-ratio constant so that neighbouring `seed` values do not share
/// initialization data.
#[must_use]
pub fn second_factor_seed(seed: u64) -> u64 {
    seed.wrapping_add(0x9E37_79B9_7F4A_7C15)
}

/// [`FIEDLER_SPREAD_FRACTION`] of `spectrum`'s eigenvalue range.
#[must_use]
pub fn fiedler_spread(spectrum: &Spectrum) -> f64 {
    let values = spectrum.eigenvalues();
    let high = values[0];
    let low = values[spectrum.order() - 1];
    (high - low).abs() * FIEDLER_SPREAD_FRACTION
}

/// ‖Vᵀe_i‖₂ for every eigenvector column of `spectrum`.
fn eigenvector_projections(spectrum: &Spectrum, embedding: &DMatrix<f64>) -> Vec<f64> {
    let projected = embedding.transpose() * spectrum.eigenvectors();
    projected
        .column_iter()
        .map(|column| column.norm())
        .collect()
}

/// ‖Ce_i‖₂ for every eigenvector column of `spectrum`.
fn coefficient_norms(spectrum: &Spectrum, coefficient: &DMatrix<f64>) -> Vec<f64> {
    let applied = coefficient * spectrum.eigenvectors();
    applied.column_iter().map(|column| column.norm()).collect()
}

/// The Observation-8 residual ‖Q_v − Q_p Q_pᵀ Q_v‖_F between the top-`k`
/// eigenspaces of P + Pᵀ and of VVᵀ, which is √(Σ sin²θ_i) over the principal
/// angles θ_i between those two subspaces. Zero means the subspaces coincide.
fn observation8_residual(
    probabilities: &DMatrix<f64>,
    embedding: &DMatrix<f64>,
    k: usize,
) -> Result<f64> {
    let probability_basis = Spectrum::new(symmetrize(probabilities))?;
    let gram = embedding * embedding.transpose();
    let gram_basis = Spectrum::new(symmetrize(&gram))?;

    let from_probabilities = probability_basis.eigenvectors().columns(0, k).into_owned();
    let from_gram = gram_basis.eigenvectors().columns(0, k).into_owned();
    let residual = &from_gram - &from_probabilities * (from_probabilities.transpose() * &from_gram);
    Ok(residual.norm())
}

/// One recorded step of a weight-tied run: the state after `step` applied
/// updates, together with the relative size of the update pending from it.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    step: usize,
    objective: f64,
    relative_update: f64,
    observation8_residual: f64,
    projections: Vec<f64>,
    coefficient_norms: Vec<f64>,
}

impl StepRecord {
    /// The number of updates applied before this state.
    #[must_use]
    pub fn step(&self) -> usize {
        self.step
    }

    /// The Eq. 1 objective at this state.
    #[must_use]
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// ‖ΔV‖_F/‖V‖_F for the update pending from this state.
    #[must_use]
    pub fn relative_update(&self) -> f64 {
        self.relative_update
    }

    /// The Observation-8 residual between the top-k eigenspaces of P + Pᵀ and
    /// VVᵀ at this state, k being [`fiedler_like_range`]'s end.
    #[must_use]
    pub fn observation8_residual(&self) -> f64 {
        self.observation8_residual
    }

    /// ‖Vᵀe_i‖₂ for every eigenvector of −L, indexed as the spectrum is.
    #[must_use]
    pub fn projections(&self) -> &[f64] {
        &self.projections
    }

    /// ‖Ce_i‖₂ for every eigenvector of −L, indexed as the spectrum is.
    #[must_use]
    pub fn coefficient_norms(&self) -> &[f64] {
        &self.coefficient_norms
    }

    /// ‖Vᵀe_0‖₂, the degenerate-vector projection Remark 5 monitors. It is
    /// [`StepRecord::projections`] entry 0, named for the trend it tracks.
    ///
    /// # Panics
    ///
    /// Panics if the record carries no projections. A record holds one per
    /// vertex, and `Graph` rejects an order of zero
    /// (`Error::InvalidGraphParameter`), so a `StepRecord` this crate built
    /// has at least one.
    #[must_use]
    pub fn degenerate_projection(&self) -> f64 {
        *self.projections.first().expect(
            "invariant: a record carries one projection per vertex and `Graph` rejects order 0",
        )
    }
}

/// One recorded step of a weight-untied run.
#[derive(Debug, Clone, PartialEq)]
pub struct UntiedStepRecord {
    step: usize,
    objective: f64,
    relative_update: f64,
    first_projections: Vec<f64>,
    second_projections: Vec<f64>,
}

impl UntiedStepRecord {
    /// The number of updates applied before this state.
    #[must_use]
    pub fn step(&self) -> usize {
        self.step
    }

    /// The weight-untied objective at this state.
    #[must_use]
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// (‖ΔV₁‖_F + ‖ΔV₂‖_F)/(‖V₁‖_F + ‖V₂‖_F) for the update pending from this
    /// state.
    #[must_use]
    pub fn relative_update(&self) -> f64 {
        self.relative_update
    }

    /// ‖V₁ᵀe_i‖₂ for every eigenvector of −L.
    #[must_use]
    pub fn first_projections(&self) -> &[f64] {
        &self.first_projections
    }

    /// ‖V₂ᵀe_i‖₂ for every eigenvector of −L.
    #[must_use]
    pub fn second_projections(&self) -> &[f64] {
        &self.second_projections
    }
}

/// Why a run's step loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The relative update fell to or below [`Params::tolerance`].
    Converged,
    /// [`Params::max_steps`] updates were applied.
    StepLimit,
    /// The `should_stop` predicate returned `true`.
    Stopped,
}

/// The result of a weight-tied run.
#[derive(Debug, Clone)]
pub struct TiedRun {
    embedding: DMatrix<f64>,
    spectrum: Spectrum,
    records: Vec<StepRecord>,
    outcome: Outcome,
    steps: usize,
}

impl TiedRun {
    /// The final embedding V.
    #[must_use]
    pub fn embedding(&self) -> &DMatrix<f64> {
        &self.embedding
    }

    /// The spectrum of −L the run measured against.
    #[must_use]
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
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
}

/// The result of a weight-untied run.
#[derive(Debug, Clone)]
pub struct UntiedRun {
    first: DMatrix<f64>,
    second: DMatrix<f64>,
    spectrum: Spectrum,
    records: Vec<UntiedStepRecord>,
    outcome: Outcome,
    steps: usize,
}

impl UntiedRun {
    /// The final first factor V₁.
    #[must_use]
    pub fn first(&self) -> &DMatrix<f64> {
        &self.first
    }

    /// The final second factor V₂.
    #[must_use]
    pub fn second(&self) -> &DMatrix<f64> {
        &self.second
    }

    /// The spectrum of −L the run measured against.
    #[must_use]
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }

    /// The recorded steps, in the order written to the CSV.
    #[must_use]
    pub fn records(&self) -> &[UntiedStepRecord] {
        &self.records
    }

    /// The last recorded step.
    #[must_use]
    pub fn last(&self) -> Option<&UntiedStepRecord> {
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
}

/// The paths a weight-untied run writes.
#[derive(Debug, Clone, Copy)]
pub struct UntiedOutputs<'a> {
    /// Per-step instrumentation.
    pub history: &'a Path,
    /// The node-node cosine-similarity matrix of the final V₁.
    pub first_cosines: &'a Path,
    /// The node-node cosine-similarity matrix of the final V₂.
    pub second_cosines: &'a Path,
}

/// Runs weight-tied gradient ascent on `graph`, streaming one CSV row per
/// recorded step to `history_path`.
///
/// The loop applies ΔV = ηCV of Lemma 6 in f64 full batch, polling
/// `should_stop` once per applied update. It stops on convergence
/// (‖ΔV‖_F/‖V‖_F ≤ [`Params::tolerance`]), on reaching
/// [`Params::max_steps`], or on `should_stop` returning `true`; the CSV holds
/// a header row followed by one complete row per recorded step in each case.
///
/// # Errors
///
/// Propagates [`Node2Vec::new`]'s and [`Params::validate`]'s errors,
/// [`Spectrum::new`]'s errors from the instrumentation, and [`Error::Io`]
/// from creating or writing `history_path`.
pub fn run_tied<S: Fn() -> bool>(
    graph: &Graph,
    params: &Params,
    seed: u64,
    history_path: &Path,
    should_stop: S,
) -> Result<TiedRun> {
    params.validate()?;
    let system = Node2Vec::new(graph)?;
    let spectrum = Spectrum::of_negative_laplacian(graph)?;
    let subspace_rank = fiedler_like_range(&spectrum, fiedler_spread(&spectrum)).end;
    let mut embedding = system.initial_embedding(params, seed)?;

    let mut sink = BufWriter::new(File::create(history_path)?);
    write_tied_header(&mut sink, system.order())?;

    let mut records = Vec::new();
    let mut steps = 0_usize;
    let outcome = loop {
        let probabilities = system.probabilities(&embedding)?;
        let coefficient = system.coefficient_from(&probabilities);
        let update = (&coefficient * &embedding) * params.learning_rate;
        let relative_update = update.norm() / embedding.norm();

        let record = StepRecord {
            step: steps,
            objective: system.objective(&embedding)?,
            relative_update,
            observation8_residual: observation8_residual(
                &probabilities,
                &embedding,
                subspace_rank,
            )?,
            projections: eigenvector_projections(&spectrum, &embedding),
            coefficient_norms: coefficient_norms(&spectrum, &coefficient),
        };
        write_tied_row(&mut sink, &record)?;
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

        embedding += &update;
        steps += 1;
    };
    sink.flush()?;

    Ok(TiedRun {
        embedding,
        spectrum,
        records,
        outcome,
        steps,
    })
}

/// Runs weight-untied gradient ascent on `graph`, streaming one CSV row per
/// recorded step to `outputs.history` and writing each final factor's
/// node-node cosine-similarity matrix to the other two paths.
///
/// The two factors are seeded from `seed` and [`second_factor_seed`], and
/// each step applies V₁ ← V₁ + η(W − P)V₂ and V₂ ← V₂ + η(W − P)ᵀV₁ from the
/// same pre-update pair. Stop conditions match [`run_tied`]'s.
///
/// # Errors
///
/// Propagates [`Node2Vec::new`]'s and [`Params::validate`]'s errors,
/// [`Spectrum::of_negative_laplacian`]'s errors, and [`Error::Io`] from
/// creating or writing any of the three files.
pub fn run_untied<S: Fn() -> bool>(
    graph: &Graph,
    params: &Params,
    seed: u64,
    outputs: &UntiedOutputs<'_>,
    should_stop: S,
) -> Result<UntiedRun> {
    params.validate()?;
    let system = Node2Vec::new(graph)?;
    let spectrum = Spectrum::of_negative_laplacian(graph)?;
    let mut first = system.initial_embedding(params, seed)?;
    let mut second = system.initial_embedding(params, second_factor_seed(seed))?;

    let mut sink = BufWriter::new(File::create(outputs.history)?);
    write_untied_header(&mut sink, system.order())?;

    let mut records = Vec::new();
    let mut steps = 0_usize;
    let outcome = loop {
        let (first_update, second_update) = system.untied_gradients(&first, &second)?;
        let first_update = first_update * params.learning_rate;
        let second_update = second_update * params.learning_rate;
        let relative_update =
            (first_update.norm() + second_update.norm()) / (first.norm() + second.norm());

        let record = UntiedStepRecord {
            step: steps,
            objective: system.untied_objective(&first, &second)?,
            relative_update,
            first_projections: eigenvector_projections(&spectrum, &first),
            second_projections: eigenvector_projections(&spectrum, &second),
        };
        write_untied_row(&mut sink, &record)?;
        records.push(record);

        if relative_update <= params.tolerance {
            break Outcome::Converged;
        }
        if steps >= params.max_steps {
            break Outcome::StepLimit;
        }
        // Polled before the update so that on every outcome the returned
        // factors are the ones the last record and both cosine dumps describe.
        if should_stop() {
            break Outcome::Stopped;
        }

        first += &first_update;
        second += &second_update;
        steps += 1;
    };
    sink.flush()?;

    write_matrix_csv(outputs.first_cosines, &cosine_similarity(&first))?;
    write_matrix_csv(outputs.second_cosines, &cosine_similarity(&second))?;

    Ok(UntiedRun {
        first,
        second,
        spectrum,
        records,
        outcome,
        steps,
    })
}

/// Writes the weight-tied header: the fixed columns followed by one
/// `projection_i` and one `coefficient_norm_i` per vertex.
fn write_tied_header<W: Write>(sink: &mut W, order: usize) -> Result<()> {
    write!(sink, "step,objective,relative_update,observation8_residual")?;
    for i in 0..order {
        write!(sink, ",projection_{i}")?;
    }
    for i in 0..order {
        write!(sink, ",coefficient_norm_{i}")?;
    }
    writeln!(sink)?;
    Ok(())
}

/// Writes one weight-tied row, each float in Rust's shortest round-tripping
/// form.
fn write_tied_row<W: Write>(sink: &mut W, record: &StepRecord) -> Result<()> {
    write!(
        sink,
        "{},{},{},{}",
        record.step, record.objective, record.relative_update, record.observation8_residual
    )?;
    for value in record.projections.iter().chain(&record.coefficient_norms) {
        write!(sink, ",{value}")?;
    }
    writeln!(sink)?;
    Ok(())
}

/// Writes the weight-untied header: the fixed columns followed by one
/// `first_projection_i` and one `second_projection_i` per vertex.
fn write_untied_header<W: Write>(sink: &mut W, order: usize) -> Result<()> {
    write!(sink, "step,objective,relative_update")?;
    for i in 0..order {
        write!(sink, ",first_projection_{i}")?;
    }
    for i in 0..order {
        write!(sink, ",second_projection_{i}")?;
    }
    writeln!(sink)?;
    Ok(())
}

/// Writes one weight-untied row, each float in Rust's shortest round-tripping
/// form.
fn write_untied_row<W: Write>(sink: &mut W, record: &UntiedStepRecord) -> Result<()> {
    write!(
        sink,
        "{},{},{}",
        record.step, record.objective, record.relative_update
    )?;
    for value in record
        .first_projections
        .iter()
        .chain(&record.second_projections)
    {
        write!(sink, ",{value}")?;
    }
    writeln!(sink)?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "vertex counts and step indices are small and exact in f64"
)]
mod tests {
    use super::*;
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
                "rediscovery-node2vec-{label}-{}-{nanos}-{counter}.csv",
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

    /// Settings the finite-difference pins sample, chosen so the logits VVᵀ
    /// stay in a range where a central difference resolves the derivative:
    /// `m * sigma^2` is the scale of a logit entry.
    const FD_SETTINGS: [(usize, f64); 2] = [(8, 0.5), (24, 0.35)];

    /// Central-difference probe size.
    const FD_STEP: f64 = 1e-5;

    /// Bound on the entrywise deviation between an analytic gradient and its
    /// central difference. The measured maxima over `FD_SETTINGS` and the four
    /// D-graphs are 2.363e-9 (tied) and 2.217e-9 (untied), so this leaves two
    /// orders of magnitude of headroom over f64 rounding at `FD_STEP`.
    const FD_TOLERANCE: f64 = 1e-7;

    /// Lemma 6's `CV` is the gradient of the Eq. 1 objective: entrywise
    /// central differences of `objective` agree with `gradient` on every
    /// D-graph. This adjudicates the sign of the probability term (analysis
    /// doc §1.3.1): the paper's Prop.-7 induction writes
    /// `C = W + Wᵀ + (P + Pᵀ)`, Lemma 6 writes the minus.
    #[test]
    fn gradient_matches_central_differences_on_every_d_graph() {
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();

        for (name, graph) in d_graphs() {
            let system = Node2Vec::new(&graph).expect("Node2Vec::new");
            for (dimension, sigma) in FD_SETTINGS {
                let params = Params {
                    dimension,
                    sigma,
                    ..Params::default()
                };
                let embedding = system
                    .initial_embedding(&params, 20_260_829)
                    .expect("initial_embedding");

                let analytic = system.gradient(&embedding).expect("gradient");
                let numeric = central_differences(&embedding, FD_STEP, |probe| {
                    system.objective(probe).expect("objective")
                });

                let deviation = (&analytic - &numeric).amax();
                if deviation > worst {
                    worst = deviation;
                    worst_label = format!("{name} at m = {dimension}, sigma = {sigma}");
                }
                assert!(
                    deviation < FD_TOLERANCE,
                    "{name} at m = {dimension}, sigma = {sigma}: max |CV − central difference| \
                     = {deviation:.6e}, tolerance {FD_TOLERANCE:e}; max |CV| = {:.6e}, \
                     probe step {FD_STEP:e}",
                    analytic.amax()
                );
            }
        }

        println!(
            "gradient_matches_central_differences_on_every_d_graph: \
             max deviation {worst:.6e} at {worst_label}"
        );
    }

    /// The weight-untied ascent directions `(W − P)V₂` and `(W − P)ᵀV₁` are
    /// the gradients of the untied objective in each factor.
    #[test]
    fn untied_gradients_match_central_differences_on_every_d_graph() {
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();

        for (name, graph) in d_graphs() {
            let system = Node2Vec::new(&graph).expect("Node2Vec::new");
            for (dimension, sigma) in FD_SETTINGS {
                let params = Params {
                    dimension,
                    sigma,
                    ..Params::default()
                };
                let first = system
                    .initial_embedding(&params, 20_260_829)
                    .expect("initial_embedding");
                let second = system
                    .initial_embedding(&params, 20_260_830)
                    .expect("initial_embedding");

                let (analytic_first, analytic_second) = system
                    .untied_gradients(&first, &second)
                    .expect("untied_gradients");
                let numeric_first = central_differences(&first, FD_STEP, |probe| {
                    system
                        .untied_objective(probe, &second)
                        .expect("untied_objective")
                });
                let numeric_second = central_differences(&second, FD_STEP, |probe| {
                    system
                        .untied_objective(&first, probe)
                        .expect("untied_objective")
                });

                for (factor, analytic, numeric) in [
                    ("V1", &analytic_first, &numeric_first),
                    ("V2", &analytic_second, &numeric_second),
                ] {
                    let deviation = (analytic - numeric).amax();
                    if deviation > worst {
                        worst = deviation;
                        worst_label = format!("{name} {factor} at m = {dimension}");
                    }
                    assert!(
                        deviation < FD_TOLERANCE,
                        "{name} {factor} at m = {dimension}, sigma = {sigma}: \
                         max |analytic − central difference| = {deviation:.6e}, tolerance \
                         {FD_TOLERANCE:e}; max |analytic| = {:.6e}, probe step {FD_STEP:e}",
                        analytic.amax()
                    );
                }
            }
        }

        println!(
            "untied_gradients_match_central_differences_on_every_d_graph: \
             max deviation {worst:.6e} at {worst_label}"
        );
    }

    /// Every row of P sums to one and keeps a positive self term p(i, i) —
    /// the denominator of Eq. 1 ranges over all k, as §F.1 states.
    #[test]
    fn probabilities_are_row_stochastic_and_keep_the_self_term() {
        let params = Params {
            dimension: 12,
            sigma: 0.7,
            ..Params::default()
        };
        for (name, graph) in d_graphs() {
            let system = Node2Vec::new(&graph).expect("Node2Vec::new");
            let embedding = system
                .initial_embedding(&params, 7)
                .expect("initial_embedding");
            let probabilities = system.probabilities(&embedding).expect("probabilities");

            for i in 0..system.order() {
                let sum = probabilities.row(i).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "{name}: row {i} of P sums to {sum:.15}, expected 1"
                );
                assert!(
                    probabilities[(i, i)] > 0.0,
                    "{name}: p({i}, {i}) is {}, but Eq. 1's denominator includes k = i",
                    probabilities[(i, i)]
                );
            }
        }
    }

    /// The objective's log-sum-exp path agrees with the logarithm of the
    /// separately computed probability matrix.
    #[test]
    fn objective_agrees_with_the_logarithm_of_the_probability_matrix() {
        let params = Params {
            dimension: 12,
            sigma: 0.7,
            ..Params::default()
        };
        for (name, graph) in d_graphs() {
            let system = Node2Vec::new(&graph).expect("Node2Vec::new");
            let embedding = system
                .initial_embedding(&params, 11)
                .expect("initial_embedding");
            let probabilities = system.probabilities(&embedding).expect("probabilities");

            let mut expected = 0.0;
            for i in 0..system.order() {
                for j in 0..system.order() {
                    let weight = system.walk()[(i, j)];
                    if weight > 0.0 {
                        expected += weight * probabilities[(i, j)].ln();
                    }
                }
            }

            let observed = system.objective(&embedding).expect("objective");
            assert!(
                (observed - expected).abs() < 1e-9,
                "{name}: objective is {observed:.15}, Σ W_ij ln P_ij is {expected:.15} \
                 (|Δ| = {:.3e})",
                (observed - expected).abs()
            );
        }
    }

    /// `fiedler_like_range` starts at the group below the top one and extends
    /// over the subsequent groups within `spread`, per footnote 9. At a zero
    /// spread the 15-cycle gives its closed-form k = ±1 pair alone.
    #[test]
    fn fiedler_like_range_starts_below_the_top_group() {
        let cycle = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&cycle).expect("spectrum");
        let range = fiedler_like_range(&spectrum, 0.0);
        assert_eq!(
            range,
            1..3,
            "cycle(15): Fiedler-like range at zero spread is {range:?}, the degenerate \
             k = ±1 pair is 1..3"
        );

        let complete = Graph::complete(7).expect("complete(7)");
        let spectrum = Spectrum::of_negative_laplacian(&complete).expect("spectrum");
        let range = fiedler_like_range(&spectrum, 0.0);
        assert_eq!(
            range,
            1..7,
            "complete(7): Fiedler-like range is {range:?}, the −2n/(n−1) eigenvalue has \
             multiplicity 6"
        );
    }

    /// A widening spread pulls whole degenerate groups into the Fiedler-like
    /// range: on the 15-cycle a spread above the 0.489 gap between the k = ±1
    /// and k = ±2 pairs extends 1..3 to 1..5.
    #[test]
    fn fiedler_like_range_widens_with_the_spread() {
        let cycle = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&cycle).expect("spectrum");

        let gap = (spectrum.eigenvalues()[1] - spectrum.eigenvalues()[3]).abs();
        let narrow = fiedler_like_range(&spectrum, gap * 0.5);
        assert_eq!(
            narrow,
            1..3,
            "cycle(15): below the {gap:.6} gap the range is {narrow:?}, expected 1..3"
        );

        let wide = fiedler_like_range(&spectrum, gap * 1.5);
        assert_eq!(
            wide,
            1..5,
            "cycle(15): above the {gap:.6} gap the range is {wide:?}, expected 1..5"
        );
    }

    /// Degenerate run parameters come back as typed errors naming the field.
    #[test]
    fn params_reject_degenerate_values() {
        let base = Params::default();
        let zero_dimension = Params {
            dimension: 0,
            ..base
        };
        match zero_dimension.validate() {
            Err(Error::InvalidDimension { dimension }) => {
                assert_eq!(dimension, 0, "reported dimension {dimension}");
            }
            other => panic!("expected InvalidDimension, got {other:?}"),
        }

        for (parameter, params) in [
            ("sigma", Params { sigma: 0.0, ..base }),
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

    /// An embedding with the wrong row count, and a mismatched factor pair,
    /// are rejected before any linear algebra runs.
    #[test]
    fn shape_mismatches_are_typed_errors() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let system = Node2Vec::new(&graph).expect("Node2Vec::new");

        match system.objective(&DMatrix::<f64>::zeros(4, 3)) {
            Err(Error::EmbeddingOrderMismatch { rows, order }) => {
                assert_eq!(
                    (rows, order),
                    (4, 15),
                    "reported {rows} rows against {order}"
                );
            }
            other => panic!("expected EmbeddingOrderMismatch, got {other:?}"),
        }

        let first = DMatrix::<f64>::zeros(15, 3);
        let second = DMatrix::<f64>::zeros(15, 4);
        match system.untied_objective(&first, &second) {
            Err(Error::EmbeddingShapeMismatch {
                columns,
                other_columns,
                ..
            }) => {
                assert_eq!(
                    (columns, other_columns),
                    (3, 4),
                    "reported {columns} against {other_columns} columns"
                );
            }
            other => panic!("expected EmbeddingShapeMismatch, got {other:?}"),
        }
    }

    /// Seed shared by the signature pins.
    const SIGNATURE_SEED: u64 = 20_260_829;

    /// Runs the weight-tied system on `graph` at `params`, into a temp file.
    fn tied_run(label: &str, graph: &Graph, params: &Params) -> TiedRun {
        let temp = TempPath::new(label);
        let started = Instant::now();
        let run = run_tied(graph, params, SIGNATURE_SEED, temp.path(), || false).expect("run_tied");
        println!(
            "{label}: {:?}, outcome {:?}, {} steps, {} rows, final objective {:.9}, \
             final relative update {:.3e}, degenerate projection {:.6} (was {:.6})",
            started.elapsed(),
            run.outcome(),
            run.steps(),
            run.records().len(),
            run.last()
                .expect("a run records its initial state")
                .objective(),
            run.last()
                .expect("a run records its initial state")
                .relative_update(),
            run.last()
                .expect("a run records its initial state")
                .degenerate_projection(),
            run.records()[0].degenerate_projection(),
        );
        run
    }

    /// Prints a run's Fig-9 measurements and returns the last record together
    /// with the Fiedler-like range it was measured against.
    fn report(label: &str, run: &TiedRun) -> (Range<usize>, StepRecord) {
        let fiedler = fiedler_like_range(run.spectrum(), fiedler_spread(run.spectrum()));
        let record = run.last().expect("a run records its initial state");
        println!(
            "{label}: Fiedler-like range {fiedler:?}, eigenvalues {:?}",
            run.spectrum()
                .eigenvalues()
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
        );
        println!(
            "{label}: projections {:?}",
            record
                .projections()
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
        );
        println!(
            "{label}: coefficient norms {:?}",
            record
                .coefficient_norms()
                .iter()
                .map(|value| format!("{value:.3e}"))
                .collect::<Vec<_>>()
        );
        println!(
            "{label}: Observation-8 residual {:.6e} (was {:.6e})",
            record.observation8_residual(),
            run.records()[0].observation8_residual()
        );

        (fiedler, record.clone())
    }

    /// The smallest and largest of `values` over `range`.
    ///
    /// Panics on an empty range: an empty slice would yield (+∞, −∞) and pass
    /// either signature assertion with no data behind it.
    fn extremes(values: &[f64], range: Range<usize>) -> (f64, f64) {
        assert!(
            !range.is_empty(),
            "extremes over an empty range {range:?} would make its assertion vacuous"
        );
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for &value in &values[range] {
            low = low.min(value);
            high = high.max(value);
        }
        (low, high)
    }

    /// The largest off-diagonal magnitude of `embedding`'s cosine matrix.
    fn peak_off_diagonal_cosine(embedding: &DMatrix<f64>) -> f64 {
        let cosines = cosine_similarity(embedding);
        let mut peak = 0.0_f64;
        for i in 0..cosines.nrows() {
            for j in 0..cosines.ncols() {
                if i != j {
                    peak = peak.max(cosines[(i, j)].abs());
                }
            }
        }
        peak
    }

    /// Asserts the Fig-9 projection signature at `threshold`: ‖Vᵀe_i‖₂ above
    /// it on the Fiedler-like index set and below it for every later
    /// eigenvector. The messages carry the measured extremes on both sides.
    fn assert_projection_signature(label: &str, run: &TiedRun, threshold: f64) {
        let (fiedler, record) = report(label, run);
        let (on_set_low, _) = extremes(record.projections(), fiedler.clone());
        let (_, beyond_high) = extremes(record.projections(), fiedler.end..run.spectrum().order());

        assert!(
            on_set_low > threshold,
            "{label}: the smallest ‖Vᵀe_i‖₂ on the Fiedler-like set {fiedler:?} is \
             {on_set_low:.6}, threshold {threshold} (largest beyond the set: {beyond_high:.6})"
        );
        assert!(
            beyond_high < threshold,
            "{label}: the largest ‖Vᵀe_i‖₂ beyond the Fiedler-like set {fiedler:?} is \
             {beyond_high:.6}, threshold {threshold} (smallest on the set: {on_set_low:.6})"
        );
    }

    /// Asserts Observation 7's null-space condition at `tolerance`: ‖Ce_i‖₂
    /// below it on the Fiedler-like set, while the smallest ‖Ce_i‖₂ beyond
    /// that set stays above `floor` — so the pin cannot be met by C
    /// collapsing everywhere.
    fn assert_null_space_signature(label: &str, run: &TiedRun, tolerance: f64, floor: f64) {
        let (fiedler, record) = report(label, run);
        let (_, on_set_high) = extremes(record.coefficient_norms(), fiedler.clone());
        let (beyond_low, _) = extremes(
            record.coefficient_norms(),
            fiedler.end..run.spectrum().order(),
        );

        assert!(
            on_set_high < tolerance,
            "{label}: the largest ‖Ce_i‖₂ on the Fiedler-like set {fiedler:?} is \
             {on_set_high:.6e}, tolerance {tolerance:e} (smallest beyond the set: \
             {beyond_low:.6e})"
        );
        assert!(
            beyond_low > floor,
            "{label}: the smallest ‖Ce_i‖₂ beyond the Fiedler-like set {fiedler:?} is \
             {beyond_low:.6e}, floor {floor:e}; without a gap the null-space claim is empty"
        );
    }

    /// Largest off-diagonal cosine magnitude below which the weight-untied
    /// factors count as showing no multi-hop structure (Fig. 33), and above
    /// which the weight-tied embedding counts as showing it. Measured at D7
    /// defaults on the four D-graphs: untied V₁ 0.328–0.353, tied
    /// 0.955–1.000.
    const COSINE_CEILING: f64 = 0.6;
    const COSINE_FLOOR: f64 = 0.9;

    /// The weight-tied run on the path-star contracts onto the Fiedler-like
    /// eigenvectors of −L (Observation 3a / Observation 7's second condition)
    /// and leaves multi-hop cosine structure behind. Observation 7's
    /// null-space condition is *not* reached here; see
    /// `path_star_null_space_signature`.
    #[test]
    fn path_star_reproduces_the_projection_signature() {
        let graph = Graph::path_star(4, 4).expect("path_star(4,4)");
        let run = tied_run("path_star(4,4)", &graph, &Params::default());
        assert_projection_signature("path_star(4,4)", &run, 3.0);

        let peak = peak_off_diagonal_cosine(run.embedding());
        assert!(
            peak > COSINE_FLOOR,
            "path_star(4,4): largest off-diagonal cosine of the tied embedding is {peak:.6}, \
             floor {COSINE_FLOOR}"
        );
    }

    /// The weight-tied run on the 4×4 grid contracts onto the Fiedler-like
    /// eigenvectors of −L. Observation 7's null-space condition is not
    /// reached here; see `grid_null_space_signature`.
    #[test]
    fn grid_reproduces_the_projection_signature() {
        let graph = Graph::grid(4, 4).expect("grid(4,4)");
        let run = tied_run("grid(4,4)", &graph, &Params::default());
        assert_projection_signature("grid(4,4)", &run, 2.0);

        let peak = peak_off_diagonal_cosine(run.embedding());
        assert!(
            peak > COSINE_FLOOR,
            "grid(4,4): largest off-diagonal cosine of the tied embedding is {peak:.6}, \
             floor {COSINE_FLOOR}"
        );
    }

    /// The 15-cycle — the one degree-regular D-graph, where Assumption 2
    /// holds exactly — reproduces the whole of Fig. 9: the projection
    /// signature, Observation 7's null-space condition, and Observation 8's
    /// eigenspace agreement. Its `max_steps` is raised to 20 000 because the
    /// D7 cap of 10 000 stops short of the 1e-8 relative-update criterion,
    /// which this graph reaches at step 15 855.
    #[test]
    fn cycle_reproduces_the_full_fig9_signature() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let params = Params {
            max_steps: 20_000,
            ..Params::default()
        };
        let run = tied_run("cycle(15)", &graph, &params);
        assert_eq!(
            run.outcome(),
            Outcome::Converged,
            "cycle(15): outcome is {:?} after {} steps, expected Converged",
            run.outcome(),
            run.steps()
        );

        assert_projection_signature("cycle(15)", &run, 1.0);
        assert_null_space_signature("cycle(15)", &run, 1e-3, 1e-2);

        let residual = run
            .last()
            .expect("a run records its initial state")
            .observation8_residual();
        assert!(
            residual < 1e-3,
            "cycle(15): Observation-8 residual between the top-k eigenspaces of P + Pᵀ and \
             VVᵀ is {residual:.6e}, tolerance 1e-3 (it was {:.6e} at initialization)",
            run.records()[0].observation8_residual()
        );

        let peak = peak_off_diagonal_cosine(run.embedding());
        assert!(
            peak > COSINE_FLOOR,
            "cycle(15): largest off-diagonal cosine of the tied embedding is {peak:.6}, \
             floor {COSINE_FLOOR}"
        );
    }

    /// On the disconnected D4 graph the projection signature holds per
    /// component rather than over one degenerate group. Its two components
    /// give −L two near-null eigenvalues (0.033007, 0.025475, 7.5e-3 apart,
    /// so `degenerate_groups` splits them) whose eigenvectors are the two
    /// component indicators, followed by the two components' Fiedler vectors
    /// at −0.226732 and −0.505536. Measured at D7 defaults, seed 20260829:
    /// projections 10.402 and 11.285 on the indicators, 4.495 and 3.692 on
    /// the Fiedler vectors, then ≤ 0.904 — so indices 0..4 carry the
    /// embedding and everything beyond is suppressed, which is the Fig-9
    /// projection claim read one component at a time.
    #[test]
    fn irregular_reproduces_the_projection_signature_per_component() {
        let graph = Graph::irregular().expect("irregular()");
        let run = tied_run("irregular()", &graph, &Params::default());
        let (_, record) = report("irregular()", &run);

        let (carried_low, _) = extremes(record.projections(), 0..4);
        let (_, suppressed_high) = extremes(record.projections(), 4..run.spectrum().order());

        assert!(
            carried_low > 3.0,
            "irregular(): the smallest ‖Vᵀe_i‖₂ over the two components' indicator and \
             Fiedler vectors (0..4) is {carried_low:.6}, threshold 3.0"
        );
        assert!(
            suppressed_high < 3.0,
            "irregular(): the largest ‖Vᵀe_i‖₂ beyond index 3 is {suppressed_high:.6}, \
             threshold 3.0 (smallest carried: {carried_low:.6})"
        );
    }

    /// Reportable deviation, not a defect (plan Phase 2 acceptance): on the
    /// path-star, ‖Ce_i‖₂ does not reach zero on the Fiedler-like set.
    /// Measured at D7 defaults, seed 20260829: 1.587e-1, 1.586e-1, 1.586e-1
    /// on set 1..4 against 1.701e-1 beyond it, a ratio of 1.07. The value
    /// grows with training — 1.841e-1 at 400 000 steps, 1.790e-1 at 1e6,
    /// byte-stable at 2e6 — so this is a fixed point, not an unfinished run,
    /// and it is unchanged across η ∈ {0.001, 0.01, 0.1}, σ ∈ {1, 4},
    /// m ∈ {100, 400}, and a different initialization draw. The paper's own
    /// Figure 9 path-star panel plots the same plateau (orange band peaking
    /// near 0.3 around epoch 100, settling at ≈0.15), so the implementation
    /// reproduces the figure and the caption's "converges to 0" over-claims
    /// relative to it.
    #[test]
    #[ignore = "reproduces the paper's figure; its caption's 'converges to 0' does not hold"]
    fn path_star_null_space_signature() {
        let graph = Graph::path_star(4, 4).expect("path_star(4,4)");
        let run = tied_run("path_star(4,4)", &graph, &Params::default());
        assert_null_space_signature("path_star(4,4)", &run, 1e-3, 1e-2);
    }

    /// Reportable deviation, not a defect: the 4×4 grid behaves as the
    /// path-star does. Measured at D7 defaults, seed 20260829: ‖Ce_i‖₂ is
    /// 6.406e-2 on set 1..3 against 6.549e-2 beyond it at 10 000 steps, a
    /// ratio of 1.02, rising to 7.409e-2 on the set at 1e6 steps. The paper's
    /// own Figure 9 grid panel plots an orange plateau at ≈0.07 with the
    /// nearest grey curve at ≈0.17.
    #[test]
    #[ignore = "reproduces the paper's figure; its caption's 'converges to 0' does not hold"]
    fn grid_null_space_signature() {
        let graph = Graph::grid(4, 4).expect("grid(4,4)");
        let run = tied_run("grid(4,4)", &graph, &Params::default());
        assert_null_space_signature("grid(4,4)", &run, 1e-3, 1e-2);
    }

    /// Weight-untying destroys the tied projection signature (C8): on every
    /// D-graph the largest ‖V₁ᵀe_i‖₂ beyond the Fiedler-like set exceeds the
    /// smallest one on it, so no threshold separates the two — the ratio
    /// stays near 1 where the tied runs reach 4.5 (path-star), 7.8 (grid) and
    /// 1180 (cycle). The dumped cosine matrices carry the Fig. 33 half of the
    /// contrast: off-diagonal magnitudes stay under `COSINE_CEILING` where
    /// the tied runs exceed `COSINE_FLOOR`.
    #[test]
    fn untied_run_breaks_the_tied_projection_signature() {
        for (name, graph) in d_graphs() {
            let history = TempPath::new("untied-history");
            let first_cosines = TempPath::new("untied-first");
            let second_cosines = TempPath::new("untied-second");
            let outputs = UntiedOutputs {
                history: history.path(),
                first_cosines: first_cosines.path(),
                second_cosines: second_cosines.path(),
            };

            let started = Instant::now();
            let run = run_untied(&graph, &Params::default(), SIGNATURE_SEED, &outputs, || {
                false
            })
            .expect("run_untied");
            let record = run.last().expect("a run records its initial state");
            let fiedler = fiedler_like_range(run.spectrum(), fiedler_spread(run.spectrum()));
            let (on_set_low, _) = extremes(record.first_projections(), fiedler.clone());
            let (_, beyond_high) = extremes(
                record.first_projections(),
                fiedler.end..run.spectrum().order(),
            );
            let separation = on_set_low / beyond_high;
            println!(
                "{name} untied: {:?}, outcome {:?}, {} steps, objective {:.6}, Fiedler-like set \
                 {fiedler:?}, smallest on set {on_set_low:.6}, largest beyond {beyond_high:.6}, \
                 ratio {separation:.4}",
                started.elapsed(),
                run.outcome(),
                run.steps(),
                record.objective(),
            );

            assert!(
                separation < UNTIED_SEPARATION_CEILING,
                "{name}: the weight-untied V₁ separates the Fiedler-like set {fiedler:?} by a \
                 factor of {separation:.4} (smallest on set {on_set_low:.6}, largest beyond \
                 {beyond_high:.6}), at or above the ceiling {UNTIED_SEPARATION_CEILING} that \
                 marks the tied signature"
            );

            for (factor, embedding) in [("V1", run.first()), ("V2", run.second())] {
                let peak = peak_off_diagonal_cosine(embedding);
                assert!(
                    peak < COSINE_CEILING,
                    "{name} untied {factor}: largest off-diagonal cosine is {peak:.6}, \
                     ceiling {COSINE_CEILING}"
                );
            }

            for path in [outputs.first_cosines, outputs.second_cosines] {
                let text = std::fs::read_to_string(path).expect("read cosine matrix");
                let lines: Vec<&str> = text.lines().collect();
                assert_eq!(
                    lines.len(),
                    graph.order() + 1,
                    "{name}: {} has {} lines, expected a header plus {} rows",
                    path.display(),
                    lines.len(),
                    graph.order()
                );
            }
        }
    }

    /// Separation ratio at or above which a run counts as showing the tied
    /// signature. Measured at D7 defaults, seed 20260829: weight-untied runs
    /// reach 0.809–0.942, weight-tied ones 4.52 (path-star), 7.83 (grid) and
    /// 1180 (cycle).
    const UNTIED_SEPARATION_CEILING: f64 = 1.5;

    /// The cosine matrix has a unit diagonal and is symmetric.
    #[test]
    fn cosine_similarity_has_a_unit_diagonal() {
        let graph = Graph::grid(4, 4).expect("grid(4,4)");
        let system = Node2Vec::new(&graph).expect("Node2Vec::new");
        let params = Params {
            dimension: 6,
            ..Params::default()
        };
        let embedding = system
            .initial_embedding(&params, 3)
            .expect("initial_embedding");

        let cosines = cosine_similarity(&embedding);
        for i in 0..graph.order() {
            assert!(
                (cosines[(i, i)] - 1.0).abs() < 1e-12,
                "diagonal entry {i} is {}, expected 1",
                cosines[(i, i)]
            );
            for j in 0..graph.order() {
                assert!(
                    (cosines[(i, j)] - cosines[(j, i)]).abs() < 1e-12,
                    "entries ({i}, {j}) and ({j}, {i}) differ: {} against {}",
                    cosines[(i, j)],
                    cosines[(j, i)]
                );
            }
        }
    }
}
