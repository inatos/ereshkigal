//! GBNF letter grammar as a slot-id check (no decode).
//!
//! Gathered option-slot token IDs must be exactly the single-token letters
//! A.. that the grammar would accept.

use crate::error::{Error, Result};
use crate::types::LETTERS;

/// Grammar that accepts exactly one of the first `n` option letters.
pub fn letter_gbnf(n: usize) -> Result<String> {
    if n < 2 || n > LETTERS.len() {
        return Err(Error::Validation(format!(
            "GBNF letter count must be 2..={}, got {n}",
            LETTERS.len()
        )));
    }
    let alts: Vec<String> = LETTERS
        .chars()
        .take(n)
        .map(|c| format!("\"{c}\""))
        .collect();
    Ok(format!("root ::= {}\n", alts.join(" | ")))
}

/// Fail if `slots` is not one unique token per letter A.. (caller supplies encodings).
pub fn slots_match_letters(slots: &[i32], letter_token_ids: &[i32]) -> Result<()> {
    if slots.len() != letter_token_ids.len() {
        return Err(Error::Validation(
            "GBNF slot count does not match letter terminals".into(),
        ));
    }
    if slots != letter_token_ids {
        return Err(Error::Validation(
            "Gathered answer slots do not match GBNF letter terminals".into(),
        ));
    }
    let mut uniq = slots.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != slots.len() {
        return Err(Error::Validation("GBNF letter terminals collide".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_letter_grammar() {
        let g = letter_gbnf(3).unwrap();
        assert_eq!(g, "root ::= \"A\" | \"B\" | \"C\"\n");
        slots_match_letters(&[32, 33, 34], &[32, 33, 34]).unwrap();
        assert!(slots_match_letters(&[32, 33], &[32, 33, 34]).is_err());
        assert!(slots_match_letters(&[32, 32, 34], &[32, 32, 34]).is_err());
    }
}
