"""The Tier-2 spectral geometry turns on and off with W's initialization alone.

This is the Python companion to `examples/w_init_flip.rs`: one file, standard
library plus numpy, reading top to bottom as the experiment runs. Two runs of
the learnable-embedding regime on the 15-cycle, at eta = 0.01 and a relative
weight rate rho = eta_W/eta_E of 1 -- W training as freely as the embedding --
differ in one thing: W(0). From the identity W the embedding reaches the
Fiedler alignment FIEDLER_ALIGNMENT that the Rust crate reads as geometric, and
the run ends there; from a Gaussian draw the alignment stays below that
threshold for the whole 2000-step budget.

Run it with `python3 examples/w_init_flip.py [seed]`.

VERIFICATION CONTRACT
---------------------
This file is an independent reimplementation, not a translation. The Rust
crate is the pinned reference: `src/tinynn.rs` defines the model, the
gradients and `fiedler_alignment`, `src/spectral.rs` the -L spectrum, and
`examples/w_init_flip.rs` the experiment. Where the two disagree, the Rust
crate is right by definition.

The two languages do not share an RNG. The reference draws N(0, sigma^2)
entries from a ChaCha20 stream by the Box-Muller transform; this file draws
them from numpy's PCG64 through `Generator.standard_normal`. Same
distribution, different numbers -- so E(0) and the Gaussian W(0) differ
between the two implementations at a given seed, and no bit-level or
trajectory-level agreement is claimed.

What is claimed, and what was measured against the reference at commit cb3aeac:

  1. Instrument level. On inputs both languages construct without an RNG, the
     two `fiedler_alignment` implementations agree to 9.4e-16, the largest
     absolute deviation over three shared 15-row embeddings: the Fiedler-like
     eigenvectors of -L, which score exactly 1.0 on both; the two most negative
     eigendirections, which score 1.5e-32 there and 2.7e-32 here, zero to
     within rounding; and the 15x512 matrix with entry (i, j) = sin(1+i+15j),
     which scores 0.05641626822998210 there and 0.05641626822998304 here.
     That sin matrix is bit-identical across the two languages in all 7680
     entries. One gradient-descent step from it with W(0) = I agrees to
     1.0e-17 entrywise on Delta_E and 2.1e-17 on Delta_W.

  2. Phenomenon level. The flip reproduces across seeds rather than at one
     lucky draw. At numpy seeds 0, 1, 2 and 20260829 the identity run crosses
     0.75 at steps 11, 14, 9 and 15, and the Gaussian run's peak over the
     2000-step budget is 0.5011, 0.4915, 0.4942 and 0.1001 -- climbing, on
     three of the four seeds, to about two thirds of the threshold and
     flattening there.

The threshold itself is the reference's, calibrated there and quoted here.
"""

import sys

import numpy as np

# --- The experiment's constants, from `examples/w_init_flip.rs`. -------------

#: Vertices of the cycle both runs train on, decision D3's graph.
CYCLE_ORDER = 15

#: Hidden width m, the column count of E and the order of W.
WIDTH = 512

#: Descent step size eta.
LEARNING_RATE = 0.01

#: The relative rate rho = eta_W/eta_E: E moves at eta and W at rho times it.
WEIGHT_RATE_RATIO = 1.0

#: Applied updates either run is allowed.
MAX_STEPS = 2000

#: Relative update at or below which a run stops as converged.
TOLERANCE = 1e-10

#: Seed both runs draw from, the first of the transition sweep's two.
SEED = 20260829

#: Steps between the Gaussian run's printed lines.
GAUSSIAN_STRIDE = 100

# --- The measure's constants, from `src/tinynn.rs` and `src/spectral.rs`. ----

#: Fiedler alignment at or above which an embedding counts as geometric.
#: Calibrated in the Rust crate against six reference embeddings on four
#: graphs; this file quotes the number rather than rederiving it.
FIEDLER_ALIGNMENT = 0.75

#: Fraction of the squared Frobenius norm of E below which a principal
#: direction counts as absent, contributing nothing to the alignment.
PRINCIPAL_DIRECTION_FLOOR = 1e-12

#: Gap below which two adjacent eigenvalues count as one degenerate group.
DEGENERACY_TOLERANCE = 1e-9

#: Magnitude above which an eigenvector component may fix that eigenvector's
#: sign.
SIGN_PIVOT_TOLERANCE = 1e-9

# --- Initialization scales, from `Params::for_width`. ------------------------

#: Standard deviation of the N(0, sigma^2) entries of E: 1/sqrt(m).
EMBEDDING_SIGMA = (1.0 / WIDTH) ** 0.5

#: Standard deviation of the N(0, sigma^2) entries of a Gaussian W: 1/m.
WEIGHT_SIGMA = 1.0 / WIDTH


# --- The graph and the distribution the model is trained to match. ----------


def cycle_adjacency(order):
    """The {0, 1} adjacency matrix of the cycle on `order` vertices."""
    adjacency = np.zeros((order, order))
    for vertex in range(order):
        following = (vertex + 1) % order
        adjacency[vertex, following] = 1.0
        adjacency[following, vertex] = 1.0
    return adjacency


def transition(adjacency):
    """The row-normalized target distribution D^-1 A.

    Row u is uniform over the neighbours of u. Every vertex of a cycle has
    degree 2, so this is A/2 there.
    """
    degrees = adjacency.sum(axis=1, keepdims=True)
    return adjacency / degrees


def negative_laplacian(adjacency):
    """-L for L = (I - D^-1 A) + (I - D^-1 A)^T, the Appendix F Laplacian.

    The summand is not symmetric on an irregular graph; L is, being of the
    form X + X^T.
    """
    deviation = np.eye(len(adjacency)) - transition(adjacency)
    return -(deviation + deviation.T)


def connected_components(adjacency):
    """The number of connected components of `adjacency`, by breadth-first
    search from every unvisited vertex."""
    order = len(adjacency)
    seen = [False] * order
    components = 0
    for source in range(order):
        if seen[source]:
            continue
        components += 1
        frontier = [source]
        seen[source] = True
        while frontier:
            vertex = frontier.pop()
            for other in range(order):
                if adjacency[vertex, other] > 0.0 and not seen[other]:
                    seen[other] = True
                    frontier.append(other)
    return components


# --- The spectrum the geometry is measured against. -------------------------


def spectrum(symmetric):
    """The eigendecomposition of a symmetric matrix, ordered and sign-fixed.

    Eigenvalues descend; eigenvector j is column j of the returned matrix, is
    unit-norm, and has a positive first component of magnitude above
    SIGN_PIVOT_TOLERANCE. Within a group of equal eigenvalues only the span is
    determined, so the columns there are one orthonormal basis among many.
    Every use of this spectrum below projects onto a whole group rather than
    comparing individual columns.
    """
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(-values, kind="stable")
    values = values[order]
    vectors = vectors[:, order]
    for column in range(vectors.shape[1]):
        significant = np.flatnonzero(np.abs(vectors[:, column]) > SIGN_PIVOT_TOLERANCE)
        if significant.size > 0 and vectors[significant[0], column] < 0.0:
            vectors[:, column] = -vectors[:, column]
    return values, vectors


def fiedler_like_set(eigenvalues, components):
    """The Fiedler-like eigenvector index range as a half-open (start, end).

    The `components` indices below the leading `components` of the spectrum,
    extended forward while the next eigenvalue is within DEGENERACY_TOLERANCE
    of the last one taken. On the 15-cycle -- one component, eigenvalues
    2cos(2 pi k/15) - 2 -- this is (1, 3): the k = +-1 degenerate pair, one
    index below the simple 0 at the top.
    """
    order = len(eigenvalues)
    start = min(components, max(order - 1, 0))
    end = min(max(start + components, start + 1), order)
    while (
        end < order
        and abs(eigenvalues[end - 1] - eigenvalues[end]) <= DEGENERACY_TOLERANCE
    ):
        end += 1
    return start, end


class System:
    """The graph-derived quantities every step reuses.

    `walk` is the target distribution D^-1 A, `trivial` the eigenvectors of -L
    projected out of an embedding before it is measured -- one per connected
    component -- and `fiedler` the eigenvectors of the Fiedler-like eigenspace
    the remainder is measured against.
    """

    def __init__(self, adjacency):
        self.walk = transition(adjacency)
        eigenvalues, eigenvectors = spectrum(negative_laplacian(adjacency))
        start, end = fiedler_like_set(eigenvalues, connected_components(adjacency))
        self.trivial = eigenvectors[:, :start]
        self.fiedler = eigenvectors[:, start:end]

    def fiedler_alignment(self, embedding):
        """The fraction of `embedding`'s leading principal directions that lie
        in the Fiedler-like eigenspace of -L.

        The trivial block is projected out of the embedding first. Of the
        remainder's principal directions -- the eigenvectors of its Gram
        matrix in descending order -- the leading k are taken, k being the
        width of the Fiedler-like block; each contributes its squared
        projection onto that eigenspace, or nothing when its share of the
        squared Frobenius norm of E falls below PRINCIPAL_DIRECTION_FLOOR, and
        the total is divided by k. Each term is at most 1, so the value lies
        in [0, 1], and it is unchanged by scaling the embedding.
        """
        deflated = embedding - self.trivial @ (self.trivial.T @ embedding)
        gram = deflated @ deflated.T
        # Adding the transpose makes the Gram exactly symmetric for the
        # eigensolver and doubles it, so eigenvalue j is twice the squared
        # singular value the floor below is stated against.
        values, directions = spectrum(gram + gram.T)
        floor = 2.0 * PRINCIPAL_DIRECTION_FLOOR * float(np.sum(embedding * embedding))

        projected = self.fiedler.T @ directions
        width = self.fiedler.shape[1]
        carried = sum(
            float(projected[:, j] @ projected[:, j])
            for j in range(width)
            if values[j] > floor
        )
        return carried / width


# --- The model: logits E W E^T, tied embedding and unembedding. -------------


def initial_parameters(seed, weight_init):
    """E drawn from N(0, EMBEDDING_SIGMA^2) and W as `weight_init` asks.

    E and W come off two streams split from one seed, so the identity run and
    the Gaussian run share E(0) exactly and differ only in W(0).
    """
    embedding_seed, weight_seed = np.random.SeedSequence(seed).spawn(2)
    embedding = (
        np.random.default_rng(embedding_seed).standard_normal((CYCLE_ORDER, WIDTH))
        * EMBEDDING_SIGMA
    )
    if weight_init == "identity":
        weight = np.eye(WIDTH)
    else:
        weight = (
            np.random.default_rng(weight_seed).standard_normal((WIDTH, WIDTH))
            * WEIGHT_SIGMA
        )
    return embedding, weight


def forward(embedding, weight):
    """The hidden state E W, the logits E W E^T, and their row softmax.

    Entry (u, v) of the logits is the logit of v given u. The softmax runs
    over all vertices, the self term included.
    """
    hidden = embedding @ weight
    logits = hidden @ embedding.T
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    return hidden, logits, probabilities


def cross_entropy(walk, logits):
    """L = -sum_uv (D^-1 A)_uv log softmax(Z)_uv, the full-batch loss.

    Each row's log-partition is shifted by that row's maximum before
    exponentiating.
    """
    peak = logits.max(axis=1, keepdims=True)
    log_partition = peak + np.log(np.exp(logits - peak).sum(axis=1, keepdims=True))
    return -float(np.sum(walk * (logits - log_partition)))


def gradients(walk, embedding, weight, hidden, probabilities):
    """dL/dE and dL/dW, by hand.

    With G = P - D^-1 A the derivative in the logits, the hidden gradient is
    G E; the hidden layer is linear, so that is also the pre-activation
    gradient. dL/dW is E^T times it, and dL/dE is G^T H + (dL/dpre) W^T.
    """
    residual = probabilities - walk
    pre_gradient = residual @ embedding
    weight_gradient = embedding.T @ pre_gradient
    embedding_gradient = residual.T @ hidden + pre_gradient @ weight.T
    return embedding_gradient, weight_gradient


# --- The run. ---------------------------------------------------------------


class Run:
    """One trained pair of blocks, with the per-step trace behind it."""

    def __init__(self, steps, alignments, losses, stop):
        self.steps = steps
        self.alignments = alignments
        self.losses = losses
        self.stop = stop

    def peak_alignment(self):
        return max(self.alignments)


def train(system, seed, weight_init, alignment_stop):
    """Full-batch gradient descent on the cross-entropy, both blocks moving.

    Each step subtracts eta times the gradient from E and rho eta times it
    from W, both blocks reading the same pre-update parameters. The run stops
    on the geometry criterion, on convergence, or on the step budget, checked
    in that order. The alignment is recorded before the update it triggers, so
    the reported step count is the number of applied updates.
    """
    embedding, weight = initial_parameters(seed, weight_init)
    alignments = []
    losses = []
    steps = 0
    while True:
        hidden, logits, probabilities = forward(embedding, weight)
        embedding_gradient, weight_gradient = gradients(
            system.walk, embedding, weight, hidden, probabilities
        )
        embedding_delta = embedding_gradient * LEARNING_RATE
        weight_delta = weight_gradient * (LEARNING_RATE * WEIGHT_RATE_RATIO)

        moved = np.linalg.norm(weight_delta) + np.linalg.norm(embedding_delta)
        relative_update = moved / (
            np.linalg.norm(weight) + np.linalg.norm(embedding)
        )
        alignments.append(system.fiedler_alignment(embedding))
        losses.append(cross_entropy(system.walk, logits))

        if alignment_stop is not None and alignments[-1] >= alignment_stop:
            stop = "the geometry criterion"
            break
        # The rate is a positive constant here, so an update this small is the
        # descent having converged rather than a schedule holding it back.
        if relative_update <= TOLERANCE:
            stop = "convergence"
            break
        if steps >= MAX_STEPS:
            stop = "the step limit"
            break

        weight -= weight_delta
        embedding -= embedding_delta
        steps += 1
    return Run(steps, alignments, losses, stop)


def report(header, run, stride):
    """Prints `header`, the alignment at every `stride`-th recorded step and
    at the last one, then the run's step count, stop reason and peak."""
    print(header)
    last = len(run.alignments) - 1
    for step, alignment in enumerate(run.alignments):
        if step % stride == 0 or step == last:
            print(f"  step {step:>4}   fiedler_alignment {alignment}")
    print(
        f"  {run.steps} steps, ended on {run.stop}, "
        f"peak alignment {run.peak_alignment()}"
    )


def main(argv):
    seed = int(argv[1]) if len(argv) > 1 else SEED
    system = System(cycle_adjacency(CYCLE_ORDER))

    print(
        f"{CYCLE_ORDER}-cycle, learnable embedding, seed {seed}, "
        f"eta {LEARNING_RATE:g}, rho {WEIGHT_RATE_RATIO:g}, gradient descent, "
        f"budget {MAX_STEPS} steps."
    )
    print("The two runs differ only in W's initialization.")
    print(f"Geometric at fiedler_alignment >= {FIEDLER_ALIGNMENT}.")
    print()

    identity = train(system, seed, "identity", FIEDLER_ALIGNMENT)
    report("W = I, geometry stop armed:", identity, 1)
    print()

    gaussian = train(system, seed, "gaussian", None)
    report("W ~ N(0, weight_sigma^2), no geometry stop:", gaussian, GAUSSIAN_STRIDE)
    print()

    print(
        f"Identity: {identity.stop} at step {identity.steps}. "
        f"Gaussian: peak {gaussian.peak_alignment()} over {gaussian.steps} steps."
    )


if __name__ == "__main__":
    main(sys.argv)
