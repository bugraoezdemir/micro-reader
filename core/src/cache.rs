//! Decoded segments kept for reuse, least recently used out (as micro-reader's
//! Python `DecodeCache`): a strip or tile that neighbouring reads share is decoded
//! once.  Keyed by (file, byte offset), which names a stored segment uniquely.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub type Key = (u64, u64);

pub struct Cache {
    items: HashMap<Key, (Arc<Vec<u8>>, u64)>,
    order: BTreeMap<u64, Key>,
    tick: u64,
    bytes: usize,
    /// what all segments together may hold
    pub limit: usize,
    /// larger segments are not kept (one whole-plane strip would evict the rest)
    pub item_limit: usize,
}

impl Cache {
    pub fn new(limit: usize, item_limit: usize) -> Self {
        Self { items: HashMap::new(), order: BTreeMap::new(), tick: 0, bytes: 0, limit, item_limit }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn get(&mut self, key: Key) -> Option<Arc<Vec<u8>>> {
        self.tick += 1;
        let tick = self.tick;
        let (seg, at) = self.items.get_mut(&key)?;
        self.order.remove(at);
        *at = tick;
        self.order.insert(tick, key);
        Some(seg.clone())
    }

    pub fn put(&mut self, key: Key, seg: Arc<Vec<u8>>) {
        let n = seg.len();
        if n > self.item_limit || n > self.limit || self.items.contains_key(&key) {
            return;
        }
        self.tick += 1;
        self.items.insert(key, (seg, self.tick));
        self.order.insert(self.tick, key);
        self.bytes += n;
        self.shrink(self.limit);
    }

    /// Drop least recently used segments until at most `limit` bytes are kept.
    pub fn shrink(&mut self, limit: usize) {
        while self.bytes > limit {
            let Some((_, key)) = self.order.pop_first() else { break };
            if let Some((seg, _)) = self.items.remove(&key) {
                self.bytes -= seg.len();
            }
        }
    }

    /// Forget every segment of file `file`.
    pub fn forget(&mut self, file: u64) {
        let keys: Vec<Key> = self.items.keys().filter(|k| k.0 == file).copied().collect();
        for key in keys {
            if let Some((seg, at)) = self.items.remove(&key) {
                self.order.remove(&at);
                self.bytes -= seg.len();
            }
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.order.clear();
        self.bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_recently_used_goes_first() {
        let mut c = Cache::new(30, 20);
        c.put((1, 0), Arc::new(vec![0; 10]));
        c.put((1, 10), Arc::new(vec![0; 10]));
        c.put((1, 20), Arc::new(vec![0; 10]));
        assert!(c.get((1, 0)).is_some());              // now the most recent
        c.put((2, 0), Arc::new(vec![0; 10]));          // evicts (1, 10)
        assert!(c.get((1, 10)).is_none());
        assert!(c.get((1, 0)).is_some() && c.get((1, 20)).is_some());
        c.put((3, 0), Arc::new(vec![0; 21]));          // above the item limit: not kept
        assert!(c.get((3, 0)).is_none());
        c.forget(1);
        assert_eq!(c.bytes(), 10);
    }
}
