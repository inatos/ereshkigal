//! Ereshkigal — semantic ifs from open GGUF models via llama.cpp.
//!
//! Inspired by [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev).
//! Independent; not affiliated with Jev or TypeSafe.

pub mod cache;
pub mod calib;
pub mod cascade;
pub mod engine;
pub mod error;
pub mod gbnf;
pub mod outline;
pub mod prompt;
pub mod score;
pub mod softmax;
pub mod tokenizer;
pub mod types;
pub mod validate;

pub use calib::{
    apply_temperature, balanced_accuracy, ece, fit_temperature, fit_temperature_oof,
    mean_family_balanced_accuracy, OofCalib,
};
pub use cascade::{cascade_select, top2_margin, CascadeOutcome};
pub use engine::{EngineConfig, EngineOwned};
pub use error::{Error, Result};
pub use gbnf::{letter_gbnf, slots_match_letters};
pub use outline::outline_state;
pub use prompt::{digest, direct_messages, render_prompt, render_prompt_version};
pub use score::{decision_argmax, probs_within_tol, Scorer, SharedTiming};
pub use softmax::{argmax, softmax};
pub use tokenizer::ReferenceTokenizer;
pub use types::*;
pub use validate::validate_row;
