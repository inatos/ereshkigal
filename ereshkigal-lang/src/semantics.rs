//! Pure semantics helpers: abstain, guards, expected cost.

use crate::decree::{Decree, Guard, OnAbstain};
use crate::decided::DecideStatus;
use crate::metrics::top2_margin_from_probs;
use crate::softmax::argmax;
use std::collections::BTreeMap;

pub fn status_for(decree: &Decree, probs: &[f64], fitted_margin: Option<f64>) -> DecideStatus {
    let margin = top2_margin_from_probs(probs);
    let top = argmax(probs).map(|i| probs[i]).unwrap_or(0.0);
    if let Some(spec) = &decree.abstain {
        if let Some(m) = spec.min_margin.or(fitted_margin) {
            if margin < m {
                return DecideStatus::Abstain;
            }
        }
        if let Some(p) = spec.min_probability {
            if top < p {
                return DecideStatus::Abstain;
            }
        }
    }
    DecideStatus::Chosen
}

pub fn apply_on_abstain(decree: &Decree, status: DecideStatus, idx: Option<usize>) -> (DecideStatus, Option<usize>) {
    if status != DecideStatus::Abstain {
        return (status, idx);
    }
    match decree.on_abstain {
        OnAbstain::Return | OnAbstain::Escalate | OnAbstain::Fail => (DecideStatus::Abstain, None),
        OnAbstain::Default => {
            if let Some(id) = &decree.default_id {
                (DecideStatus::Chosen, decree.option_index(id))
            } else {
                (DecideStatus::Abstain, None)
            }
        }
    }
}

pub fn expected_cost_choice(decree: &Decree, probs: &[f64]) -> Option<String> {
    if decree.costs.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best = f64::INFINITY;
    for (j, opt) in decree.options.iter().enumerate() {
        let mut c = 0.0;
        for (i, truth) in decree.options.iter().enumerate() {
            let p = probs.get(i).copied().unwrap_or(0.0);
            let cost = decree
                .costs
                .get(&truth.id)
                .and_then(|m| m.get(&opt.id))
                .copied()
                .unwrap_or(if i == j { 0.0 } else { 1.0 });
            c += p * cost;
        }
        if c < best {
            best = c;
            best_i = j;
        }
    }
    Some(decree.options[best_i].id.clone())
}

pub fn eval_guard(g: &Guard, facts: &BTreeMap<String, String>, probs: &BTreeMap<String, Vec<f64>>) -> bool {
    match g {
        Guard::Is { node, is } => facts.get(node).map(|v| v == is).unwrap_or(false),
        Guard::In { node, r#in } => facts
            .get(node)
            .map(|v| r#in.iter().any(|x| x == v))
            .unwrap_or(false),
        Guard::PAtLeast { node, p_at_least, id } => {
            let Some(p) = probs.get(node) else { return false };
            if let Some(opt) = id {
                // id is option index via 0 if missing
                let _ = opt;
                p.iter().copied().fold(0.0, f64::max) >= *p_at_least
            } else {
                p.iter().copied().fold(0.0, f64::max) >= *p_at_least
            }
        }
        Guard::All { all } => all.iter().all(|x| eval_guard(x, facts, probs)),
        Guard::Any { any } => any.iter().any(|x| eval_guard(x, facts, probs)),
        Guard::Not { not } => !eval_guard(not, facts, probs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decree::{AbstainSpec, DecreeKind};
    use crate::types::OptionSpec;

    fn d() -> Decree {
        Decree {
            name: "x".into(),
            question: "q".into(),
            options: vec![
                OptionSpec {
                    id: "yes".into(),
                    description: "y".into(),
                },
                OptionSpec {
                    id: "no".into(),
                    description: "n".into(),
                },
            ],
            kind: DecreeKind::Enum,
            abstain: Some(AbstainSpec {
                coverage: Some(0.8),
                min_margin: Some(0.5),
                min_probability: None,
            }),
            on_abstain: OnAbstain::Return,
            default_id: None,
            costs: Default::default(),
            tests: vec![],
            variants: vec![],
            backend: Default::default(),
            debias: Default::default(),
            tree: None,
            recipe: None,
        }
    }

    #[test]
    fn low_margin_abstains() {
        let st = status_for(&d(), &[0.51, 0.49], None);
        assert_eq!(st, DecideStatus::Abstain);
    }
}
