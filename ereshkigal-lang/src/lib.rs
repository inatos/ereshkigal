//! Pure Ereshkigal language front end (no llama.cpp).

pub mod calib;
pub mod conformal;
pub mod debias;
pub mod decided;
pub mod decree;
pub mod error;
pub mod lint;
pub mod lockfile;
pub mod metrics;
pub mod optimize;
pub mod pkg;
pub mod probe;
pub mod prompt;
pub mod ranking;
pub mod schema;
pub mod semantics;
pub mod softmax;
pub mod syntax;
pub mod template;
pub mod types;
pub mod validate;

pub use calib::{
    apply_temperature, balanced_accuracy, ece, fit_temperature, fit_temperature_oof,
    fit_temperature_oof_grouped, mean_family_balanced_accuracy, OofCalib,
};
pub use conformal::{coverage as conformal_coverage, fit_qhat, mean_set_size, option_set};
pub use optimize::{overlap_nll, pick_lowest_nll, rank_variants_overlap};
pub use debias::{
    apply_content_free, apply_pride, cycle_options, permute_debias, pride_frac_count,
    pride_log_prior, prior_from_content_free,
};
pub use decided::{DecideStatus, Decided, ProgramResult};
pub use decree::{Decree, Library, Program};
pub use error::{Error, Result};
pub use prompt::{digest, direct_messages, render_prompt, render_prompt_version};
pub use softmax::{argmax, softmax};
pub use schema::library_schema_json;
pub use syntax::{parse_library, print_library};
pub use types::*;
pub use validate::validate_row;
