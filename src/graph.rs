//! Graph construction for Tier 0 of the POC.
//!
//! Builds adjacency and degree representations for the five topologies used
//! throughout later tiers: path-star, grid, cycle, irregular, and tree-star.
//! Everything downstream — the `spectral` module, and the `Node2Vec`
//! dynamics of later tiers — consumes the graphs constructed here.
