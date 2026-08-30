//! Library-first core for the arXiv 2510.26745v2 geometric-memory POC.
//!
//! See `docs/2510.26745v2-poc-analysis.md` for the design.

pub mod error;
pub mod graph;
pub mod logger;
pub mod node2vec;
pub mod numerics;
pub mod output;
pub mod settings;
pub mod spectral;
pub mod tinynn;

pub mod subsystems {
    pub mod runner;
}
