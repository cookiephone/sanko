//! Depth-first proof-number search for Go under positional superko, with sound
//! proof transfer between paths, PSK-safe static knowledge and shared-table
//! parallelism.
//!
//! The API serves the `sanko` solver and is not stable: it may change in any
//! 0.x release.

pub mod checkpoint;
pub mod dfpn;
pub mod pntt;
pub mod prior;
mod static_cache;

pub use checkpoint::{CheckpointError, CheckpointHeader, CheckpointOptions, read_checkpoint_header};
pub use dfpn::{
    DfpnConfig, DfpnResult, DfpnStats, PRIOR_FEATURES, Progress, ResumableResult, enable_proof_dump, flush_proof_dump,
    solve_dfpn, solve_dfpn_resumable,
};
pub use pntt::TableFill;
pub use static_cache::StaticCache;
