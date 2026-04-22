
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// 3D Spatial coordinate for each entry in the neural map
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuralPosition {
    /// X-dimension: Linear insertion order index
    pub sequence: u64,
    /// Y-dimension: Importance/weight layer (0=base, max=critical)
    pub priority_layer: u32,
    /// Z-dimension: Neural depth - access frequency tier
    pub access_depth: u16,
    /// Guidance: Temporal locality score (0.0-1.0)
    pub temporal_score: f32,
}

impl NeuralPosition {
    /// Create new neural position
    pub const fn new(seq: u64, priority: u32, depth: u16, temporal: f32) -> Self {
        Self {
            sequence: seq,
            priority_layer: priority,
            access_depth: depth,
            temporal_score: temporal,
        }
    }

    /// Default position for new entries
    pub const fn default_at(sequence: u64) -> Self {
        Self::new(sequence, 0, 0, 0.5)
    }

    /// Hot/cold position based on access patterns
    pub fn hot_position(sequence: u64) -> Self {
        Self::new(sequence, 100, 50, 0.95)
    }

    /// Cold storage position
    pub fn cold_position(sequence: u64) -> Self {
        Self::new(sequence, 0, 500, 0.05)
    }

    /// Calculate 3D neural relevance score
    pub fn relevance_score(&self) -> f64 {
        let seq_factor = 1.0 / (1.0 + (self.sequence as f64 * 0.001));
        let priority = self.priority_layer as f64 / u32::MAX as f64;
        let depth_inv = 1.0 - (self.access_depth as f64 / u16::MAX as f64);
        let temporal = self.temporal_score as f64;
        
        (seq_factor * 0.1 + priority * 0.3 + depth_inv * 0.3 + temporal * 0.3)
            .clamp(0.0, 1.0)
    }
}

impl fmt::Display for NeuralPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[S:{} P:{} D:{} T:{:.2}]",
            self.sequence,
            self.priority_layer,
            self.access_depth,
            self.temporal_score
        )
    }
}

/// Entry node in the doubly-linked neural list
#[derive(Debug, Clone)]
pub struct NeuralEntry<K, V> {
    /// Entry key
    pub key: K,
    /// Entry value
    pub value: V,
    /// 3D spatial position
    pub position: NeuralPosition,
    /// Previous entry index (None if first)
    prev: Option<u64>,
    /// Next entry index (None if last)
    next: Option<u64>,
}

impl<K, V> NeuralEntry<K, V> {
    /// Create new neural entry
    pub fn new(key: K, value: V, position: NeuralPosition) -> Self {
        Self {
            key,
            value,
            position,
            prev: None,
            next: None,
        }
    }

    /// Check if this entry has higher priority than other
    pub fn higher_priority_than(&self, other: &Self) -> bool {
        self.position.relevance_score() > other.position.relevance_score()
    }
}

/// 3D Spatial Neural Ordered HashMap
/// Maintains insertion order while enabling 3D spatial queries
#[derive(Debug, Clone)]
pub struct NeuralOrderedMap<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Hash map for O(1) key lookups
    index: HashMap<K, u64>,
    /// Storage of entries (index -> entry)
    entries: HashMap<u64, NeuralEntry<K, V>>,
    /// First entry index (oldest)
    head: Option<u64>,
    /// Last entry index (newest)
    tail: Option<u64>,
    /// Next available sequence number
    next_sequence: u64,
    /// 3D spatial acceleration structure
    spatial_layers: HashMap<u32, Vec<u64>>,
    /// Phantom data for type safety
    _phantom: PhantomData<V>,
}

impl<K, V> NeuralOrderedMap<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Create new empty neural ordered map
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            entries: HashMap::new(),
            head: None,
            tail: None,
            next_sequence: 0,
            spatial_layers: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with estimated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            index: HashMap::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            next_sequence: 0,
            spatial_layers: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get value by key (updates temporal score)
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.index.get(key).copied()?;
        
        // Update temporal score on access (neural learning)
        if let Some(entry) = self.entries.get_mut(&idx) {
            entry.position.temporal_score = (entry.position.temporal_score * 0.9 + 0.1).min(1.0);
            entry.position.access_depth = entry.position.access_depth.saturating_sub(10);
        }
        
        self.entries.get(&idx).map(|e| &e.value)
    }

    /// Get immutable reference without updating scores
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.index.get(key)
            .and_then(|&idx| self.entries.get(&idx))
            .map(|e| &e.value)
    }

    /// Insert or update key-value pair
    /// If key exists, updates value but preserves position
    /// If new, appends to end with new 3D position
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists
        if let Some(&idx) = self.index.get(&key) {
            // Update existing entry
            if let Some(entry) = self.entries.get_mut(&idx) {
                let old = std::mem::replace(&mut entry.value, value);
                // Boost priority on update
                entry.position.priority_layer = entry.position.priority_layer.saturating_add(5);
                return Some(old);
            }
        }

        // Create new entry with 3D position
        let position = NeuralPosition::default_at(self.next_sequence);
        let mut entry = NeuralEntry::new(key.clone(), value, position);
        
        // Link to tail
        if let Some(tail_idx) = self.tail {
            entry.prev = Some(tail_idx);
            if let Some(tail_entry) = self.entries.get_mut(&tail_idx) {
                tail_entry.next = Some(self.next_sequence);
            }
        } else {
            // First entry
            self.head = Some(self.next_sequence);
        }
        
        self.tail = Some(self.next_sequence);
        
        // Add to spatial layer
        self.spatial_layers
            .entry(position.priority_layer)
            .or_default()
            .push(self.next_sequence);
        
        // Store entry and index
        self.index.insert(key, self.next_sequence);
        self.entries.insert(self.next_sequence, entry);
        
        self.next_sequence += 1;
        
        None
    }

    /// Insert with custom 3D position
    pub fn insert_with_position(
        &mut self,
        key: K,
        value: V,
        position: NeuralPosition,
    ) -> Option<V> {
        if self.index.contains_key(&key) {
            return self.insert(key, value);
        }

        let mut entry = NeuralEntry::new(key.clone(), value, position);
        let seq = self.next_sequence;
        
        // Link to tail
        if let Some(tail_idx) = self.tail {
            entry.prev = Some(tail_idx);
            if let Some(tail_entry) = self.entries.get_mut(&tail_idx) {
                tail_entry.next = Some(seq);
            }
        } else {
            self.head = Some(seq);
        }
        
        self.tail = Some(seq);
        
        // Add to spatial layer
        self.spatial_layers
            .entry(position.priority_layer)
            .or_default()
            .push(seq);
        
        self.index.insert(key, seq);
        self.entries.insert(seq, entry);
        
        self.next_sequence += 1;
        
        None
    }

    /// Remove entry by key
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.index.remove(key)?;
        let entry = self.entries.remove(&idx)?;
        
        // Unlink from list
        if let Some(prev) = entry.prev {
            if let Some(prev_entry) = self.entries.get_mut(&prev) {
                prev_entry.next = entry.next;
            }
        } else {
            // Was head
            self.head = entry.next;
        }
        
        if let Some(next) = entry.next {
            if let Some(next_entry) = self.entries.get_mut(&next) {
                next_entry.prev = entry.prev;
            }
        } else {
            // Was tail
            self.tail = entry.prev;
        }
        
        // Remove from spatial layer
        if let Some(layer) = self.spatial_layers.get_mut(&entry.position.priority_layer) {
            layer.retain(|&x| x != idx);
        }
        
        Some(entry.value)
    }

    /// Check if contains key
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.index.contains_key(key)
    }

    /// Iterate in insertion order (Oldest to Newest)
    pub fn iter(&self) -> NeuralOrderedIter<'_, K, V> {
        NeuralOrderedIter {
            map: self,
            current: self.head,
        }
    }

    /// Iterate in reverse insertion order
    pub fn iter_rev(&self) -> NeuralOrderedIterRev<'_, K, V> {
        NeuralOrderedIterRev {
            map: self,
            current: self.tail,
        }
    }

    /// Iterate by 3D priority (highest relevance first)
    pub fn iter_by_priority(&self) -> impl Iterator<Item = (&K, &V)> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| {
            b.position.relevance_score()
                .partial_cmp(&a.position.relevance_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        entries.into_iter().map(|e| (&e.key, &e.value))
    }

    /// Get first entry (oldest)
    pub fn first(&self) -> Option<(&K, &V)> {
        self.head.and_then(|idx| {
            self.entries.get(&idx).map(|e| (&e.key, &e.value))
        })
    }

    /// Get last entry (newest)
    pub fn last(&self) -> Option<(&K, &V)> {
        self.tail.and_then(|idx| {
            self.entries.get(&idx).map(|e| (&e.key, &e.value))
        })
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.index.clear();
        self.entries.clear();
        self.spatial_layers.clear();
        self.head = None;
        self.tail = None;
        self.next_sequence = 0;
    }

    /// Get keys in order
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Get values in order
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Promote entry to higher priority layer
    pub fn promote(&mut self, key: &K, new_layer: u32) -> bool {
        if let Some(&idx) = self.index.get(key) {
            if let Some(entry) = self.entries.get_mut(&idx) {
                // Remove from old layer
                if let Some(layer) = self.spatial_layers.get_mut(&entry.position.priority_layer) {
                    layer.retain(|&x| x != idx);
                }
                
                // Update and add to new layer
                entry.position.priority_layer = new_layer;
                self.spatial_layers
                    .entry(new_layer)
                    .or_default()
                    .push(idx);
                
                return true;
            }
        }
        false
    }

    /// Get entries at specific priority layer
    pub fn at_layer(&self, layer: u32) -> impl Iterator<Item = (&K, &V)> {
        self.spatial_layers
            .get(&layer)
            .map(|indices| indices.iter())
            .into_iter()
            .flatten()
            .filter_map(move |&idx| {
                self.entries.get(&idx).map(|e| (&e.key, &e.value))
            })
    }

    /// Get 3D spatial summary
    pub fn spatial_summary(&self) -> SpatialSummary {
        SpatialSummary {
            total_entries: self.len(),
            layers: self.spatial_layers.len(),
            head_sequence: self.head.unwrap_or(0),
            tail_sequence: self.tail.unwrap_or(0),
            average_relevance: if !self.is_empty() {
                self.entries.values()
                    .map(|e| e.position.relevance_score())
                    .sum::<f64>() / self.len() as f64
            } else {
                0.0
            },
        }
    }

    /// Convert to standard HashMap (loses order)
    pub fn to_hashmap(&self) -> HashMap<K, V>
    where
        K: Clone,
        V: Clone,
    {
        self.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl<K, V> Default for NeuralOrderedMap<K, V>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Spatial summary statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpatialSummary {
    pub total_entries: usize,
    pub layers: usize,
    pub head_sequence: u64,
    pub tail_sequence: u64,
    pub average_relevance: f64,
}

/// Iterator over ordered entries
pub struct NeuralOrderedIter<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    map: &'a NeuralOrderedMap<K, V>,
    current: Option<u64>,
}

impl<'a, K, V> Iterator for NeuralOrderedIter<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current?;
        let entry = self.map.entries.get(&idx)?;
        
        self.current = entry.next;
        
        Some((&entry.key, &entry.value))
    }
}

/// Reverse iterator
pub struct NeuralOrderedIterRev<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    map: &'a NeuralOrderedMap<K, V>,
    current: Option<u64>,
}

impl<'a, K, V> Iterator for NeuralOrderedIterRev<'a, K, V>
where
    K: Eq + Hash + Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current?;
        let entry = self.map.entries.get(&idx)?;
        
        self.current = entry.prev;
        
        Some((&entry.key, &entry.value))
    }
}

/// Thread-safe variant using Arc
#[derive(Debug, Clone)]
pub struct ArcNeuralOrderedMap<K, V>(Arc<NeuralOrderedMap<K, V>>)
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Send + Sync;

impl<K, V> ArcNeuralOrderedMap<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    /// Create new thread-safe map
    pub fn new() -> Self {
        Self(Arc::new(NeuralOrderedMap::new()))
    }

    /// Get clone of value
    pub fn get_cloned<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.peek(key).cloned()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<K, V> Default for ArcNeuralOrderedMap<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating maps with custom 3D settings
pub struct NeuralMapBuilder {
    initial_capacity: usize,
    default_priority: u32,
    hot_threshold: f32,
}

impl NeuralMapBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            initial_capacity: 16,
            default_priority: 0,
            hot_threshold: 0.7,
        }
    }

    /// Set initial capacity
    pub fn capacity(mut self, cap: usize) -> Self {
        self.initial_capacity = cap;
        self
    }

    /// Set default priority
    pub fn priority(mut self, priority: u32) -> Self {
        self.default_priority = priority;
        self
    }

    /// Build the map
    pub fn build<K, V>(self) -> NeuralOrderedMap<K, V>
    where
        K: Eq + Hash + Clone,
    {
        NeuralOrderedMap::with_capacity(self.initial_capacity)
    }
}

impl Default for NeuralMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        
        // Update preserves position
        map.insert("b", 20);
        assert_eq!(map.get(&"b"), Some(&20));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_iteration_order() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("first", 1);
        map.insert("second", 2);
        map.insert("third", 3);
        
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec![&"first", &"second", &"third"]);
        
        let values: Vec<_> = map.values().collect();
        assert_eq!(values, vec![&1, &2, &3]);
    }

    #[test]
    fn test_reverse_iteration() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        
        let keys: Vec<_> = map.iter_rev().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![&"c", &"b", &"a"]);
    }

    #[test]
    fn test_3d_position() {
        let pos = NeuralPosition::new(0, 100, 50, 0.9);
        assert_eq!(pos.sequence, 0);
        assert_eq!(pos.priority_layer, 100);
        assert!(pos.relevance_score() > 0.0);
    }

    #[test]
    fn test_priority_promotion() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("key", "value");
        assert!(map.promote(&"key", 100));
        
        let entries_at_layer: Vec<_> = map.at_layer(100).collect();
        assert_eq!(entries_at_layer.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        
        assert_eq!(map.remove(&"b"), Some(2));
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&"b"));
        
        // Order preserved: a, c
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec![&"a", &"c"]);
    }

    #[test]
    fn test_spatial_summary() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("a", 1);
        map.insert("b", 2);
        map.promote(&"b", 50);
        
        let summary = map.spatial_summary();
        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.layers, 2);
    }

    #[test]
    fn test_temporal_learning() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert("key", "value");
        let initial_score = map.entries.get(&0).unwrap().position.temporal_score;
        
        // Access multiple times
        for _ in 0..10 {
            map.get(&"key");
        }
        
        let final_score = map.entries.get(&0).unwrap().position.temporal_score;
        assert!(final_score > initial_score);
    }

    #[test]
    fn test_priority_iteration() {
        let mut map = NeuralOrderedMap::new();
        
        map.insert_with_position("low", 1, NeuralPosition::cold_position(0));
        map.insert_with_position("high", 2, NeuralPosition::hot_position(1));
        
        let prioritized: Vec<_> = map.iter_by_priority().map(|(k, _)| *k).collect();
        // High priority should come first
        assert_eq!(prioritized[0], "high");
    }

    #[test]
    fn test_builder() {
        let map: NeuralOrderedMap<String, i32> = NeuralMapBuilder::new()
            .capacity(100)
            .priority(10)
            .build();
        
        assert!(map.is_empty());
    }

    #[test]
    fn test_to_hashmap() {
        let mut map = NeuralOrderedMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        
        let hashmap = map.to_hashmap();
        assert_eq!(hashmap.len(), 2);
        assert_eq!(hashmap.get("a"), Some(&1));
    }
}