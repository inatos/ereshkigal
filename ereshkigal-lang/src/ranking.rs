//! Pairwise ranking: both-order averaging, Bradley–Terry, merge/tournament schedules.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// Average P(left better) from both comparison directions.
pub fn both_order_p(p_ab: f64, p_ba: f64) -> f64 {
    (p_ab + (1.0 - p_ba)) / 2.0
}

pub fn merge_sort_schedule(n: usize) -> Vec<(usize, usize)> {
    let mut items: Vec<usize> = (0..n).collect();
    let mut comps = Vec::new();
    fn sort(items: &mut [usize], comps: &mut Vec<(usize, usize)>) {
        if items.len() < 2 {
            return;
        }
        let mid = items.len() / 2;
        let mut left = items[..mid].to_vec();
        let mut right = items[mid..].to_vec();
        sort(&mut left, comps);
        sort(&mut right, comps);
        let mut i = 0;
        let mut j = 0;
        let mut out = Vec::new();
        while i < left.len() && j < right.len() {
            comps.push((left[i], right[j]));
            // placeholder order: stable by index until scores applied
            if left[i] < right[j] {
                out.push(left[i]);
                i += 1;
            } else {
                out.push(right[j]);
                j += 1;
            }
        }
        out.extend_from_slice(&left[i..]);
        out.extend_from_slice(&right[j..]);
        items.copy_from_slice(&out);
    }
    sort(&mut items, &mut comps);
    comps
}

pub fn tournament_schedule(n: usize) -> Vec<(usize, usize)> {
    let mut comps = Vec::new();
    let mut live: Vec<usize> = (0..n).collect();
    while live.len() > 1 {
        let mut next = Vec::new();
        let mut i = 0;
        while i + 1 < live.len() {
            comps.push((live[i], live[i + 1]));
            next.push(live[i]); // winner decided later
            i += 2;
        }
        if i < live.len() {
            next.push(live[i]);
        }
        live = next;
    }
    comps
}

/// Fit Bradley–Terry strengths from pairwise P(i beats j).
pub fn bradley_terry(n: usize, pairs: &[((usize, usize), f64)], iters: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::Validation("empty ranking".into()));
    }
    let mut s = vec![1.0; n];
    for _ in 0..iters.max(1) {
        let mut num = vec![0.0; n];
        let mut den = vec![0.0; n];
        for &((i, j), p) in pairs {
            if i >= n || j >= n {
                continue;
            }
            num[i] += p;
            num[j] += 1.0 - p;
            let d = s[i] + s[j];
            if d > 0.0 {
                den[i] += 1.0 / d;
                den[j] += 1.0 / d;
            }
        }
        for i in 0..n {
            if den[i] > 0.0 {
                s[i] = (num[i] / den[i]).max(1e-8);
            }
        }
        let mean = s.iter().sum::<f64>() / n as f64;
        if mean > 0.0 {
            for x in &mut s {
                *x /= mean;
            }
        }
    }
    Ok(s)
}

pub fn order_from_scores(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    idx
}

pub fn memo_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub fn apply_comparisons(
    n: usize,
    decide: impl Fn(usize, usize) -> f64,
) -> HashMap<(usize, usize), f64> {
    let mut memo = HashMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let p_ij = decide(i, j);
            let p_ji = decide(j, i);
            memo.insert((i, j), both_order_p(p_ij, p_ji));
        }
    }
    memo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_order_symmetric() {
        assert!((both_order_p(0.8, 0.3) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn bt_ranks_winner() {
        let pairs = [((0, 1), 0.9), ((0, 2), 0.8), ((1, 2), 0.6)];
        let s = bradley_terry(3, &pairs, 40).unwrap();
        let ord = order_from_scores(&s);
        assert_eq!(ord[0], 0);
    }
}
