//! >16-option tournament bucketing (up to 255) for letter-logit readout.

use crate::error::{Error, Result};
use crate::types::{DecisionRow, OptionSpec, LETTERS};

const LETTER_CAP: usize = 16;

/// Bucket options into ≤16-way rounds; winners advance until one remains.
pub fn tournament_buckets(options: &[OptionSpec]) -> Vec<Vec<usize>> {
    if options.len() <= LETTER_CAP {
        return vec![(0..options.len()).collect()];
    }
    let mut rounds = Vec::new();
    let mut alive: Vec<usize> = (0..options.len()).collect();
    while alive.len() > LETTER_CAP {
        let mut next = Vec::new();
        for chunk in alive.chunks(LETTER_CAP) {
            rounds.push(chunk.to_vec());
            // Placeholder: caller scores each chunk and picks a winner index into `next`.
            next.push(chunk[0]);
        }
        if next.len() == alive.len() {
            // Safety: ensure progress.
            next.truncate(alive.len().saturating_sub(1).max(1));
        }
        alive = next;
    }
    rounds.push(alive);
    rounds
}

/// Build a DecisionRow restricted to `indices` (re-lettered A..).
pub fn row_for_bucket(base: &DecisionRow, indices: &[usize]) -> Result<DecisionRow> {
    if indices.is_empty() || indices.len() > LETTER_CAP {
        return Err(Error::Validation(format!(
            "bucket size {} invalid (1..{LETTER_CAP})",
            indices.len()
        )));
    }
    let mut options = Vec::with_capacity(indices.len());
    for &i in indices {
        let o = base
            .options
            .get(i)
            .ok_or_else(|| Error::Validation(format!("option index {i} OOB")))?;
        options.push(OptionSpec {
            id: o.id.clone(),
            description: o.description.clone(),
        });
    }
    let _ = LETTERS; // letters assigned by prompt recipe from option order
    Ok(DecisionRow {
        id: format!("{}-bucket", base.id),
        state: base.state.clone(),
        question: base.question.clone(),
        options,
    })
}

/// Run a tournament by invoking `score_bucket(row) -> winning local index`.
pub fn run_tournament<F>(base: &DecisionRow, mut score_bucket: F) -> Result<usize>
where
    F: FnMut(&DecisionRow) -> Result<usize>,
{
    if base.options.is_empty() {
        return Err(Error::Validation("no options".into()));
    }
    if base.options.len() <= LETTER_CAP {
        let row = row_for_bucket(base, &(0..base.options.len()).collect::<Vec<_>>())?;
        return score_bucket(&row);
    }
    let mut alive: Vec<usize> = (0..base.options.len()).collect();
    while alive.len() > 1 {
        let mut next = Vec::new();
        for chunk in alive.chunks(LETTER_CAP) {
            if chunk.len() == 1 {
                next.push(chunk[0]);
                continue;
            }
            let row = row_for_bucket(base, chunk)?;
            let local = score_bucket(&row)?;
            let winner = chunk
                .get(local)
                .copied()
                .ok_or_else(|| Error::Validation("tournament winner OOB".into()))?;
            next.push(winner);
        }
        if next.len() == alive.len() && alive.len() > LETTER_CAP {
            // Force shrink if scorer always returns same path.
            next.pop();
        }
        alive = next;
    }
    Ok(alive[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts(n: usize) -> Vec<OptionSpec> {
        (0..n)
            .map(|i| OptionSpec {
                id: format!("o{i}"),
                description: format!("option {i}"),
            })
            .collect()
    }

    #[test]
    fn tournament_reduces_32_to_one() {
        let base = DecisionRow {
            id: "t".into(),
            state: json!("s"),
            question: "q".into(),
            options: opts(32),
        };
        let winner = run_tournament(&base, |_row| Ok(0)).expect("tournament");
        assert!(winner < 32);
    }

    #[test]
    fn small_set_single_bucket() {
        let o = opts(4);
        let b = tournament_buckets(&o);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].len(), 4);
    }
}
