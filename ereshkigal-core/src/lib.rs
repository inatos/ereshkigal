//! Ereshkigal — semantic ifs from open GGUF models via llama.cpp.
//!
//! Inspired by [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev).
//! Independent; not affiliated with Jev or TypeSafe.

pub use ereshkigal_lang::*;
pub use ereshkigal_lang::{
    DecisionRow, EncodedDecision, Error, OptionSpec, Result, ScoreResult, GgufMeta, ModelMeta,
};

pub mod cache;
pub mod cascade;
pub mod engine;
pub mod gbnf;
pub mod outline;
pub mod runner;
pub mod score;
pub mod store;
pub mod tokenizer;

pub use cascade::{cascade_select, cascade_select_conformal, top2_margin, CascadeOutcome};
pub use engine::{EngineConfig, EngineOwned};
pub use gbnf::{letter_gbnf, slots_match_letters};
pub use outline::outline_state;
pub use runner::Runtime;
pub use score::{decision_argmax, probs_within_tol, Scorer, SharedTiming};
pub use store::ReplayStore;
