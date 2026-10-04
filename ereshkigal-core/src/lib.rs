//! Ereshkigal — semantic ifs from open GGUF models via llama.cpp.
//!
//! Inspired by [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev).
//! Independent; not affiliated with Jev or TypeSafe.

pub mod engine;
pub mod error;
pub mod prompt;
pub mod score;
pub mod softmax;
pub mod tokenizer;
pub mod types;
pub mod validate;

pub use engine::{EngineConfig, EngineOwned};
pub use error::{Error, Result};
pub use prompt::{digest, direct_messages, render_prompt};
pub use score::{decision_argmax, probs_within_tol, Scorer, SharedTiming};
pub use softmax::{argmax, softmax};
pub use tokenizer::ReferenceTokenizer;
pub use types::*;
pub use validate::validate_row;
