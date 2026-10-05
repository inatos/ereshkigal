//! Program / decree runtime over a Scorer.

use crate::decree::{bind_state, ForEachOp, Guard, Library, Program};
use crate::decided::{DecideStatus, Decided, ProgramResult};
use crate::error::{Error, Result};
use crate::metrics::top2_margin_from_probs;
use crate::score::Scorer;
use crate::semantics::{apply_on_abstain, eval_guard, expected_cost_choice, status_for};
use crate::softmax::argmax;
use crate::store::ReplayStore;
use crate::types::DecisionRow;
use serde_json::Value;
use std::collections::BTreeMap;

pub struct Runtime {
    pub scorer: Scorer,
    pub store: ReplayStore,
    pub temperature: Option<f64>,
    pub fitted_margins: BTreeMap<String, f64>,
    pub conformal_qhat: BTreeMap<String, f64>,
    pub escalate: Option<Box<Runtime>>,
    pub aurc_ok: bool,
}

impl Runtime {
    pub fn new(scorer: Scorer) -> Self {
        Self {
            scorer,
            store: ReplayStore::memory(),
            temperature: None,
            fitted_margins: BTreeMap::new(),
            conformal_qhat: BTreeMap::new(),
            escalate: None,
            aurc_ok: false,
        }
    }

    pub fn decide(&mut self, lib: &Library, name: &str, state: &Value) -> Result<Decided> {
        let decree = lib.decree(name)?;
        if decree.kind == crate::decree::DecreeKind::Tree {
            return self.decide_tree(lib, name, state);
        }
        if decree.kind == crate::decree::DecreeKind::Multilabel {
            return self.decide_multilabel(lib, name, state);
        }
        let row = decree.to_row(format!("{name}-live"), state.clone())?;
        self.score_decree(decree, &row)
    }

    fn score_decree(&mut self, decree: &crate::decree::Decree, row: &DecisionRow) -> Result<Decided> {
        let mut result = self.scorer.score_direct(row)?;
        if let Some(t) = self.temperature {
            result.calibrated_probabilities = Some(crate::apply_temperature(&result.option_logits, t)?);
        }
        let probs = result
            .calibrated_probabilities
            .clone()
            .unwrap_or_else(|| result.probabilities.clone());
        let mut idx = argmax(&probs);
        let fitted = self.fitted_margins.get(&decree.name).copied();
        let mut status = status_for(decree, &probs, fitted);
        if status == DecideStatus::Abstain
            && matches!(decree.on_abstain, crate::decree::OnAbstain::Escalate)
            && self.aurc_ok
        {
            if let Some(esc) = self.escalate.as_mut() {
                return esc.score_decree(decree, row);
            }
        }
        let applied = apply_on_abstain(decree, status, idx);
        status = applied.0;
        idx = applied.1;
        let choice = idx.map(|i| decree.options[i].id.clone());
        let conformal_set = self.conformal_qhat.get(&decree.name).map(|q| {
            crate::option_set(&probs, *q)
                .into_iter()
                .map(|i| decree.options[i].id.clone())
                .collect()
        });
        Ok(Decided {
            id: row.id.clone(),
            decree: decree.name.clone(),
            choice,
            choice_index: idx,
            probabilities: probs.clone(),
            calibrated_probabilities: result.calibrated_probabilities.clone(),
            option_ids: result.option_ids.clone(),
            margin: top2_margin_from_probs(&probs),
            status,
            conformal_set,
            min_cost_choice: expected_cost_choice(decree, &probs),
            path_probability: idx.map(|i| probs[i]),
            prompt_sha256: result.prompt_sha256,
            backend: format!("{:?}", decree.backend),
            cache_hit: result.cache_hit,
        })
    }

    fn decide_tree(&mut self, lib: &Library, name: &str, state: &Value) -> Result<Decided> {
        let decree = lib.decree(name)?;
        let groups = decree
            .tree
            .as_ref()
            .ok_or_else(|| Error::Validation("tree missing groups".into()))?;
        // Level 0: pick group (as options).
        let mut group_dec = decree.clone();
        group_dec.kind = crate::decree::DecreeKind::Enum;
        group_dec.options = groups
            .iter()
            .map(|g| crate::types::OptionSpec {
                id: g.id.clone(),
                description: g.description.clone(),
            })
            .collect();
        let row = group_dec.to_row(format!("{name}-g"), state.clone())?;
        let gdec = self.score_decree(&group_dec, &row)?;
        let gid = gdec
            .choice
            .clone()
            .ok_or_else(|| Error::Validation("tree abstained at group".into()))?;
        let children = groups
            .iter()
            .find(|g| g.id == gid)
            .map(|g| g.children.clone())
            .unwrap_or_default();
        let mut leaf = decree.clone();
        leaf.kind = crate::decree::DecreeKind::Enum;
        leaf.options = decree
            .options
            .iter()
            .filter(|o| children.contains(&o.id))
            .cloned()
            .collect();
        if leaf.options.len() < 2 {
            return Ok(gdec);
        }
        let row = leaf.to_row(format!("{name}-l"), state.clone())?;
        let mut leaf_d = self.score_decree(&leaf, &row)?;
        if let (Some(pg), Some(pl)) = (gdec.path_probability, leaf_d.path_probability) {
            leaf_d.path_probability = Some(pg * pl);
        }
        Ok(leaf_d)
    }

    fn decide_multilabel(&mut self, lib: &Library, name: &str, state: &Value) -> Result<Decided> {
        let decree = lib.decree(name)?;
        let mut on = Vec::new();
        let mut probs = Vec::new();
        for opt in &decree.options {
            let mut yesno = decree.clone();
            yesno.kind = crate::decree::DecreeKind::Enum;
            yesno.options = vec![
                crate::types::OptionSpec {
                    id: "yes".into(),
                    description: opt.description.clone(),
                },
                crate::types::OptionSpec {
                    id: "no".into(),
                    description: format!("Not: {}", opt.description),
                },
            ];
            let row = yesno.to_row(format!("{}-{}", name, opt.id), state.clone())?;
            let d = self.score_decree(&yesno, &row)?;
            let p_yes = d.probabilities.first().copied().unwrap_or(0.0);
            probs.push(p_yes);
            if d.choice.as_deref() == Some("yes") {
                on.push(opt.id.clone());
            }
        }
        Ok(Decided {
            id: name.into(),
            decree: name.into(),
            choice: Some(on.join(",")),
            choice_index: None,
            probabilities: probs,
            calibrated_probabilities: None,
            option_ids: decree.options.iter().map(|o| o.id.clone()).collect(),
            margin: 0.0,
            status: DecideStatus::Chosen,
            conformal_set: None,
            min_cost_choice: None,
            path_probability: None,
            prompt_sha256: String::new(),
            backend: "multilabel".into(),
            cache_hit: None,
        })
    }

    pub fn run(&mut self, lib: &Library, name: &str, state: &Value) -> Result<ProgramResult> {
        let program: &Program = lib.program(name)?;
        let mut facts: BTreeMap<String, String> = BTreeMap::new();
        let mut node_probs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut nodes_out = Vec::new();
        let mut skipped = Vec::new();
        let mut forwards = 0usize;
        let mut path_p = 1.0;
        for node in &program.nodes {
            if let Some(g) = &node.when {
                if !eval_guard(g, &facts, &node_probs) {
                    skipped.push(node.id.clone());
                    continue;
                }
            }
            let bound = if node.with.is_empty() {
                state.clone()
            } else {
                let mut f = BTreeMap::new();
                for w in &node.with {
                    if let Some(v) = facts.get(w) {
                        f.insert(w.clone(), v.clone());
                    }
                }
                bind_state(state, &f)
            };
            if let Some(fe) = &node.foreach {
                let items = json_pointer(&bound, &fe.path);
                match fe.op {
                    ForEachOp::Filter => {
                        let mut kept = Vec::new();
                        for item in items {
                            let d = self.decide(lib, &node.decree, &item)?;
                            forwards += 1;
                            if d.choice.as_deref() == Some("yes") {
                                kept.push(item);
                            }
                        }
                        facts.insert(node.id.clone(), serde_json::to_string(&kept)?);
                    }
                    ForEachOp::TopK => {
                        let k = fe.k.unwrap_or(1);
                        let mut scored = Vec::new();
                        for item in items {
                            let d = self.decide(lib, &node.decree, &item)?;
                            forwards += 1;
                            let p = d.probabilities.first().copied().unwrap_or(0.0);
                            scored.push((p, item));
                        }
                        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        let top: Vec<Value> = scored.into_iter().take(k).map(|(_, v)| v).collect();
                        facts.insert(node.id.clone(), serde_json::to_string(&top)?);
                    }
                    ForEachOp::SortPairwise => {
                        let n = items.len();
                        let mut pairs = Vec::new();
                        for i in 0..n {
                            for j in (i + 1)..n {
                                let p_ij = self.compare_pair(lib, &node.decree, &items[i], &items[j])?;
                                let p_ji = self.compare_pair(lib, &node.decree, &items[j], &items[i])?;
                                forwards += 2;
                                pairs.push(((i, j), crate::ranking::both_order_p(p_ij, p_ji)));
                            }
                        }
                        let scores = crate::ranking::bradley_terry(n, &pairs, 30)?;
                        let order = crate::ranking::order_from_scores(&scores);
                        let ordered: Vec<Value> = order.into_iter().map(|i| items[i].clone()).collect();
                        facts.insert(node.id.clone(), serde_json::to_string(&ordered)?);
                    }
                    ForEachOp::Map => {
                        let mut out = Vec::new();
                        for item in items {
                            let d = self.decide(lib, &node.decree, &item)?;
                            forwards += 1;
                            out.push(d.choice.unwrap_or_default());
                        }
                        facts.insert(node.id.clone(), out.join(","));
                    }
                    ForEachOp::Group => {
                        let mut buckets: BTreeMap<String, Vec<Value>> = BTreeMap::new();
                        for item in items {
                            let d = self.decide(lib, &node.decree, &item)?;
                            forwards += 1;
                            let key = d.choice.unwrap_or_else(|| "_abstain".into());
                            buckets.entry(key).or_default().push(item);
                        }
                        facts.insert(node.id.clone(), serde_json::to_string(&buckets)?);
                    }
                }
                continue;
            }
            let d = self.decide(lib, &node.decree, &bound)?;
            forwards += 1;
            if let Some(p) = d.path_probability {
                path_p *= p;
            }
            if let Some(c) = &d.choice {
                facts.insert(node.id.clone(), c.clone());
            }
            node_probs.insert(node.id.clone(), d.probabilities.clone());
            nodes_out.push(d);
        }
        let value = resolve_result(program, &facts);
        Ok(ProgramResult {
            program: name.into(),
            value,
            nodes: nodes_out,
            path_probability: Some(path_p),
            skipped,
            forwards,
        })
    }

    fn compare_pair(&mut self, lib: &Library, decree: &str, a: &Value, b: &Value) -> Result<f64> {
        let state = serde_json::json!({"left": a, "right": b});
        let d = self.decide(lib, decree, &state)?;
        Ok(d.probabilities.first().copied().unwrap_or(0.5))
    }
}

fn json_pointer(v: &Value, path: &str) -> Vec<Value> {
    let mut cur = v;
    for part in path.split('.').filter(|s| !s.is_empty() && *s != "evidence") {
        match cur {
            Value::Object(m) => {
                if let Some(n) = m.get(part) {
                    cur = n;
                } else if let Some(n) = m.get("input") {
                    cur = n;
                } else {
                    return vec![];
                }
            }
            Value::Array(_) => break,
            _ => return vec![cur.clone()],
        }
    }
    match cur {
        Value::Array(a) => a.clone(),
        other => vec![other.clone()],
    }
}

fn resolve_result(program: &Program, facts: &BTreeMap<String, String>) -> String {
    if program.result.is_empty() {
        return facts.values().next().cloned().unwrap_or_default();
    }
    for arm in &program.result {
        if arm.when.iter().all(|w| facts.values().any(|v| v == w) || facts.contains_key(w)) {
            return arm.then.clone();
        }
    }
    facts.values().next().cloned().unwrap_or_default()
}

#[allow(dead_code)]
fn _use_guard_type(_: &Guard) {}
