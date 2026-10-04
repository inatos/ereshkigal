//! Evaluation metrics for decrees and programs.

use crate::softmax::argmax;

pub fn group_consistency(pred: &[usize], gold: &[usize], groups: &[String]) -> f64 {
    assert_eq!(pred.len(), gold.len());
    assert_eq!(pred.len(), groups.len());
    if pred.is_empty() {
        return 0.0;
    }
    use std::collections::BTreeMap;
    let mut by: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for i in 0..pred.len() {
        by.entry(&groups[i]).or_default().push((pred[i], gold[i]));
    }
    let mut ok = 0.0;
    for rows in by.values() {
        if rows.iter().all(|(p, g)| p == g) {
            ok += 1.0;
        }
    }
    ok / by.len() as f64
}

/// Accuracy among the highest-margin `coverage` fraction of rows.
pub fn selective_accuracy(margins: &[f64], correct: &[bool], coverage: f64) -> f64 {
    assert_eq!(margins.len(), correct.len());
    if margins.is_empty() {
        return 0.0;
    }
    let cov = coverage.clamp(0.0, 1.0);
    let mut idx: Vec<usize> = (0..margins.len()).collect();
    idx.sort_by(|a, b| {
        margins[*b]
            .partial_cmp(&margins[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = ((cov * idx.len() as f64).round() as usize).clamp(1, idx.len());
    let hits = idx[..k].iter().filter(|i| correct[**i]).count();
    hits as f64 / k as f64
}

/// Area under the risk-coverage curve (lower is better). Coverage from 1/n to 1.
pub fn aurc(margins: &[f64], correct: &[bool]) -> f64 {
    assert_eq!(margins.len(), correct.len());
    if margins.len() < 2 {
        return 1.0 - (correct.iter().filter(|c| **c).count() as f64 / margins.len().max(1) as f64);
    }
    let mut idx: Vec<usize> = (0..margins.len()).collect();
    idx.sort_by(|a, b| {
        margins[*b]
            .partial_cmp(&margins[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut wrong = 0.0;
    let mut area = 0.0;
    for (k, &i) in idx.iter().enumerate() {
        if !correct[i] {
            wrong += 1.0;
        }
        let risk = wrong / (k + 1) as f64;
        area += risk;
    }
    area / idx.len() as f64
}

pub fn top2_margin_from_probs(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 0.0;
    }
    let mut s = probs.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    s[0] - s[1]
}

pub fn mae(pred: &[usize], gold: &[usize]) -> f64 {
    assert_eq!(pred.len(), gold.len());
    if pred.is_empty() {
        return 0.0;
    }
    let sum: f64 = pred
        .iter()
        .zip(gold)
        .map(|(p, g)| (*p as f64 - *g as f64).abs())
        .sum();
    sum / pred.len() as f64
}

pub fn kendall_tau(pred_order: &[usize], gold_order: &[usize]) -> f64 {
    assert_eq!(pred_order.len(), gold_order.len());
    let n = pred_order.len();
    if n < 2 {
        return 1.0;
    }
    let mut pos_gold = vec![0usize; n];
    for (rank, &id) in gold_order.iter().enumerate() {
        if id < n {
            pos_gold[id] = rank;
        }
    }
    let mut conc = 0.0;
    let mut disc = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = pos_gold[pred_order[i]];
            let b = pos_gold[pred_order[j]];
            if a < b {
                conc += 1.0;
            } else {
                disc += 1.0;
            }
        }
    }
    (conc - disc) / (conc + disc)
}

pub fn ndcg_at_k(pred_order: &[usize], gold_order: &[usize], k: usize) -> f64 {
    let k = k.min(pred_order.len()).min(gold_order.len());
    if k == 0 {
        return 0.0;
    }
    let n = gold_order.len();
    let mut rel = vec![0.0; n];
    for (rank, &id) in gold_order.iter().enumerate() {
        if id < n {
            rel[id] = (n - rank) as f64;
        }
    }
    let mut dcg = 0.0;
    for i in 0..k {
        let id = pred_order[i];
        let r = if id < n { rel[id] } else { 0.0 };
        dcg += r / (i as f64 + 2.0).log2();
    }
    let mut ideal: Vec<f64> = rel.clone();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut idcg = 0.0;
    for i in 0..k {
        idcg += ideal[i] / (i as f64 + 2.0).log2();
    }
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

pub fn argmax_correct(probs: &[f64], gold: usize) -> bool {
    argmax(probs) == Some(gold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_all_or_nothing() {
        let pred = vec![0, 0, 1, 1];
        let gold = vec![0, 0, 1, 0];
        let g = vec!["a".into(), "a".into(), "b".into(), "b".into()];
        let v = group_consistency(&pred, &gold, &g);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn selective_prefers_confident() {
        let margins = vec![0.9, 0.1, 0.8, 0.05];
        let correct = vec![true, false, true, false];
        let acc = selective_accuracy(&margins, &correct, 0.5);
        assert!((acc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kendall_identity() {
        let o = vec![0, 1, 2];
        assert!((kendall_tau(&o, &o) - 1.0).abs() < 1e-12);
    }
}
