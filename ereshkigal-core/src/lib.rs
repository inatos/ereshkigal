//! Ereshkigal — semantic ifs from open GGUF models via llama.cpp.
//!
//! Inspired by [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev).
//! Independent; not affiliated with Jev or TypeSafe.

pub use ereshkigal_lang::*;
pub use ereshkigal_lang::{
    DecisionRow, EncodedDecision, Error, OptionSpec, Result, ScoreResult, GgufMeta, ModelMeta,
};

pub mod autothink;
pub mod cache;
pub mod cascade;
pub mod cosine_head;
pub mod engine;
pub mod gbnf;
pub mod outline;
pub mod runner;
pub mod score;
pub mod store;
pub mod tokenizer;
pub mod tournament;
#[cfg(feature = "vision")]
pub mod vision;

pub use autothink::{
    apply_temperature_to_probs, build_autothink_outcome, fuse_probs, generate_thought,
    is_stop_token, should_think, AutoThinkOutcome, ThinkConfig, ThinkMode,
};
pub use cascade::{
    cascade_select, cascade_select_conformal, draft_agrees_argmax, speculative_tree_cascade,
    speculative_tree_verify, top2_margin, CascadeOutcome, SpeculativeTreeOutcome,
};
pub use cosine_head::CosineHead;
pub use engine::{required_n_ctx, truncate_tokens_head_tail, EngineConfig, EngineOwned};
pub use gbnf::{letter_gbnf, slots_match_letters};
pub use outline::outline_state;
pub use runner::Runtime;
pub use score::{decision_argmax, probs_within_tol, Scorer, SharedTiming};
pub use store::ReplayStore;
pub use tokenizer::locate_token_positions;
pub use tournament::{run_tournament, tournament_buckets};
