//! Prefix radix KV (SGLang RadixAttention-inspired) and exact prompt replay
//! (Dream-RSI prefix-only stored outcomes). Not an autonomous scheduler.

use llama_cpp_2::context::session::SeqState;
use std::collections::HashMap;

#[derive(Default)]
struct Node {
    children: HashMap<i32, Box<Node>>,
    state: Option<SeqState>,
}

/// Token-prefix trie of serialized llama.cpp sequence states.
#[derive(Default)]
pub struct PrefixRadix {
    root: Node,
    inserts: usize,
}

impl PrefixRadix {
    pub fn insert(&mut self, tokens: &[i32], state: SeqState) {
        if tokens.is_empty() {
            return;
        }
        let mut node = &mut self.root;
        for &t in tokens {
            node = node
                .children
                .entry(t)
                .or_insert_with(|| Box::new(Node::default()));
        }
        node.state = Some(state);
        self.inserts += 1;
    }

    /// Longest prefix that has a stored state. Returns (token_len, state).
    pub fn longest(&self, tokens: &[i32]) -> Option<(usize, &SeqState)> {
        let mut node = &self.root;
        let mut best: Option<(usize, &SeqState)> = None;
        for (i, t) in tokens.iter().enumerate() {
            node = node.children.get(t)?;
            if let Some(ref st) = node.state {
                best = Some((i + 1, st));
            }
        }
        best
    }

    pub fn len(&self) -> usize {
        self.inserts
    }

    pub fn is_empty(&self) -> bool {
        self.inserts == 0
    }
}

/// Exact `prompt_sha256` → last score (zero-compute replay).
#[derive(Default)]
pub struct ReplayCache {
    inner: HashMap<String, crate::types::ScoreResult>,
}

impl ReplayCache {
    pub fn get(&self, sha: &str) -> Option<&crate::types::ScoreResult> {
        self.inner.get(sha)
    }

    pub fn insert(&mut self, sha: String, result: crate::types::ScoreResult) {
        self.inner.insert(sha, result);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_longest_is_none() {
        let r = PrefixRadix::default();
        assert!(r.longest(&[1, 2, 3]).is_none());
        assert!(r.is_empty());
    }
}
