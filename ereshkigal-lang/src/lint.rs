//! Lints over a loaded library.

use crate::decree::{DecreeKind, Guard, Library, OnAbstain};
use crate::error::{Error, Result};
use crate::types::LETTERS;
use std::collections::{BTreeSet, HashSet};

pub fn lint_library(lib: &Library) -> Result<()> {
    let mut errs = Vec::new();
    for (name, d) in &lib.decrees {
        let ids: HashSet<_> = d.options.iter().map(|o| o.id.as_str()).collect();
        if ids.len() != d.options.len() {
            errs.push(format!("{name}: duplicate option ids"));
        }
        if d.options.len() < 2 {
            errs.push(format!("{name}: need at least two options"));
        }
        if d.kind != DecreeKind::Tree && d.options.len() > LETTERS.len() {
            errs.push(format!("{name}: >16 options requires kind=tree"));
        }
        if d.kind == DecreeKind::Tree {
            if d.tree.as_ref().map(|t| t.is_empty()).unwrap_or(true) {
                errs.push(format!("{name}: tree kind needs tree groups"));
            }
        }
        let q = d.question.to_lowercase();
        let evidence_like = q.contains("evidence") || q.contains("claim") || q.contains("true");
        let has_insuff = d
            .options
            .iter()
            .any(|o| o.id.contains("insuff") || o.description.to_lowercase().contains("insufficient"));
        if evidence_like && !has_insuff {
            errs.push(format!("{name}: evidence question missing insufficient option"));
        }
        for i in 0..d.options.len() {
            for j in (i + 1)..d.options.len() {
                if d.options[i].description.trim().eq_ignore_ascii_case(d.options[j].description.trim())
                {
                    errs.push(format!(
                        "{name}: near-duplicate options {} / {}",
                        d.options[i].id, d.options[j].id
                    ));
                }
            }
        }
        if matches!(d.on_abstain, OnAbstain::Default) && d.default_id.is_none() {
            errs.push(format!("{name}: on_abstain=default needs default_id"));
        }
        if d.tests.is_empty() {
            errs.push(format!("{name}: decree has no tests"));
        }
        for t in &d.tests {
            if d.option_index(&t.expect).is_none() && t.expect != "abstain" {
                errs.push(format!("{name}: test expect {} not in options", t.expect));
            }
        }
    }
    for (pname, p) in &lib.programs {
        let mut seen = BTreeSet::new();
        for n in &p.nodes {
            if !seen.insert(n.id.clone()) {
                errs.push(format!("{pname}: duplicate node {}", n.id));
            }
            if !lib.decrees.contains_key(&n.decree) {
                errs.push(format!("{pname}: unknown decree {}", n.decree));
            }
            for w in &n.with {
                if !seen.contains(w) && w != &n.id {
                    // earlier nodes only — `seen` already has this and previous
                    if !p.nodes.iter().any(|x| x.id == *w) {
                        errs.push(format!("{pname}: with {w} unknown"));
                    }
                }
            }
            if let Some(g) = &n.when {
                check_guard(pname, g, &p.nodes.iter().map(|x| x.id.as_str()).collect(), &mut errs);
            }
        }
        if has_cycle(p) {
            errs.push(format!("{pname}: cycle in with/when graph"));
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(Error::Lint(errs.join("; ")))
    }
}

fn check_guard(pname: &str, g: &Guard, nodes: &BTreeSet<&str>, errs: &mut Vec<String>) {
    match g {
        Guard::Is { node, .. } | Guard::In { node, .. } | Guard::PAtLeast { node, .. } => {
            if !nodes.contains(node.as_str()) {
                errs.push(format!("{pname}: guard node {node} unknown"));
            }
        }
        Guard::All { all } => {
            for x in all {
                check_guard(pname, x, nodes, errs);
            }
        }
        Guard::Any { any } => {
            for x in any {
                check_guard(pname, x, nodes, errs);
            }
        }
        Guard::Not { not } => check_guard(pname, not, nodes, errs),
    }
}

fn has_cycle(p: &crate::decree::Program) -> bool {
    use std::collections::HashMap;
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for n in &p.nodes {
        for w in &n.with {
            adj.entry(&n.id).or_default().push(w);
        }
        collect_guard_nodes(n.when.as_ref(), &n.id, &mut adj);
    }
    fn visit<'a>(
        n: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        stack: &mut HashSet<&'a str>,
        seen: &mut HashSet<&'a str>,
    ) -> bool {
        if stack.contains(n) {
            return true;
        }
        if !seen.insert(n) {
            return false;
        }
        stack.insert(n);
        if let Some(ns) = adj.get(n) {
            for m in ns {
                if visit(m, adj, stack, seen) {
                    return true;
                }
            }
        }
        stack.remove(n);
        false
    }
    let mut seen = HashSet::new();
    let mut stack = HashSet::new();
    p.nodes.iter().any(|n| visit(&n.id, &adj, &mut stack, &mut seen))
}

fn collect_guard_nodes<'a>(
    g: Option<&'a Guard>,
    id: &'a str,
    adj: &mut std::collections::HashMap<&'a str, Vec<&'a str>>,
) {
    let Some(g) = g else { return };
    match g {
        Guard::Is { node, .. } | Guard::In { node, .. } | Guard::PAtLeast { node, .. } => {
            adj.entry(id).or_default().push(node);
        }
        Guard::All { all } => {
            for x in all {
                collect_guard_nodes(Some(x), id, adj);
            }
        }
        Guard::Any { any } => {
            for x in any {
                collect_guard_nodes(Some(x), id, adj);
            }
        }
        Guard::Not { not } => collect_guard_nodes(Some(not), id, adj),
    }
}
