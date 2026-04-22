use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use std::vec::IntoIter;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Neural loading state for tracking lazy data fetch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuralLoadState {
    /// Data not yet requested
    Dormant,
    /// Data currently loading
    Streaming,
    /// Partial data loaded
    Partial(u32, u32), // (loaded_count, total_count)
    /// All data loaded successfully
    Complete,
    /// Loading failed
    Failed(NeuralLoadError),
}

/// Error types for neural lazy loading
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralLoadError {
    IoError { context: String },
    SerializationError { reason: String },
    SpatialConfigError { width: u32, height: u32, depth: u32 },
    Timeout { duration_ms: u64 },
    MemoryExhausted { requested_bytes: u64, available_bytes: u64 },
}

impl std::fmt::Display for NeuralLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError { context } => write!(f, "I/O error: {}", context),
            Self::SerializationError { reason } => write!(f, "Serialization error: {}", reason),
            Self::SpatialConfigError { width, height, depth } => {
                write!(f, "Invalid spatial config: {}x{}x{}", width, height, depth)
            }
            Self::Timeout { duration_ms } => write!(f, "Loading timeout: {}ms", duration_ms),
            Self::MemoryExhausted { requested_bytes, available_bytes } => {
                write!(f, "Memory exhausted: requested {} bytes, available {} bytes", 
                    requested_bytes, available_bytes)
            }
        }
    }
}

impl std::error::Error for NeuralLoadError {}

/// 3D spatial context for lazy-loaded neural data
#[derive(Debug, Clone)]
pub struct NeuralSpatialContext {
    /// 3D spatial dimensions
    pub dimensions: SpatialTensorMetadata,
    /// Priority for loading (higher = load first)
    pub load_priority: u32,
    /// Memory alignment requirements
    pub alignment: u32,
    /// Whether to use zero-copy mapping
    pub zero_copy_preferred: bool,
    /// Estimated memory footprint
    pub estimated_bytes: u64,
}

impl NeuralSpatialContext {
    /// Create new spatial context
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        let dimensions = SpatialTensorMetadata::new(width, height, depth);
        let estimated_bytes = dimensions.volume() * 4; // Assume 4 bytes per element default
        
        Self {
            dimensions,
            load_priority: 100,
            alignment: 64, // 64-byte alignment for SIMD
            zero_copy_preferred: true,
            estimated_bytes,
        }
    }

    /// With custom priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.load_priority = priority;
        self
    }

    /// Disable zero-copy
    pub fn with_copy_required(mut self) -> Self {
        self.zero_copy_preferred = false;
        self
    }

    /// With custom alignment
    pub fn with_alignment(mut self, alignment: u32) -> Self {
        self.alignment = alignment;
        self
    }

    /// Calculate volume
    #[inline]
    pub fn volume(&self) -> u64 {
        self.dimensions.volume()
    }
}

impl Default for NeuralSpatialContext {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

/// Callback for successful loading completion
pub type NeuralSuccessCallback = Box<dyn FnOnce() -> Result<(), NeuralLoadError> + Send>;

/// Titan-level lazy loader with 3D spatial optimization
pub struct TitanNeuralLazyLoader<T: Clone + Send> {
    /// Total expected item count
    total_count: u64,
    /// Currently loaded values
    loaded_values: Arc<RwLock<Vec<T>>>,
    /// State tracking
    state: Arc<RwLock<NeuralLoadState>>,
    /// Source generator function
    generator: Arc<Mutex<Box<dyn FnMut() -> Result<T, NeuralLoadError> + Send>>>,
    /// Spatial context for 3D optimization
    spatial_context: NeuralSpatialContext,
    /// Success callback
    on_complete: Option<NeuralSuccessCallback>,
    /// Item iterator for sequential access
    iterator_position: Arc<Mutex<usize>>,
    /// Phantom marker for type safety
    _phantom: PhantomData<T>,
}

impl<T: Clone + Send + 'static> TitanNeuralLazyLoader<T> {
    /// Create new lazy loader with count and generator
    pub fn new<F>(
        total_count: u64,
        spatial_context: NeuralSpatialContext,
        generator: F,
    ) -> Self
    where
        F: FnMut() -> Result<T, NeuralLoadError> + Send + 'static,
    {
        let initial_capacity = total_count.min(1024) as usize; // Reasonable initial capacity
        
        Self {
            total_count,
            loaded_values: Arc::new(RwLock::new(Vec::with_capacity(initial_capacity))),
            state: Arc::new(RwLock::new(NeuralLoadState::Dormant)),
            generator: Arc::new(Mutex::new(Box::new(generator))),
            spatial_context,
            on_complete: None,
            iterator_position: Arc::new(Mutex::new(0)),
            _phantom: PhantomData,
        }
    }

    /// With success callback
    pub fn on_complete<F>(mut self, callback: F) -> Self
    where
        F: FnOnce() -> Result<(), NeuralLoadError> + Send + 'static,
    {
        self.on_complete = Some(Box::new(callback));
        self
    }

    /// Get current loading state
    pub fn state(&self) -> NeuralLoadState {
        *self.state.read().unwrap()
    }

    /// Get spatial context
    pub fn spatial_context(&self) -> &NeuralSpatialContext {
        &self.spatial_context
    }

    /// Get total expected count
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Get currently loaded count
    pub fn loaded_count(&self) -> usize {
        self.loaded_values.read().unwrap().len()
    }

    /// Check if fully loaded
    pub fn is_complete(&self) -> bool {
        self.loaded_count() as u64 >= self.total_count
    }

    /// Check if loading has started
    pub fn has_started(&self) -> bool {
        !matches!(self.state(), NeuralLoadState::Dormant)
    }

    /// Load next item (pull-based)
    pub fn load_next(&self) -> Option<T> {
        let mut state = self.state.write().unwrap();
        
        // Update state to streaming if dormant
        if matches!(*state, NeuralLoadState::Dormant) {
            *state = NeuralLoadState::Streaming;
        }
        
        drop(state); // Release lock before potentially blocking operation
        
        let mut generator = self.generator.lock().unwrap();
        let loaded = self.loaded_values.read().unwrap();
        let current_count = loaded.len() as u64;
        drop(loaded);
        
        if current_count >= self.total_count {
            // All items loaded, trigger completion if not already done
            self.check_and_trigger_completion();
            return None;
        }
        
        // Generate next item
        match generator() {
            Ok(item) => {
                drop(generator);
                
                // Store item
                let mut values = self.loaded_values.write().unwrap();
                values.push(item.clone());
                let new_count = values.len() as u32;
                let total = self.total_count as u32;
                drop(values);
                
                // Update state
                let mut state = self.state.write().unwrap();
                if new_count >= total {
                    *state = NeuralLoadState::Complete;
                    drop(state);
                    self.check_and_trigger_completion();
                } else {
                    *state = NeuralLoadState::Partial(new_count, total);
                }
                
                Some(item)
            }
            Err(e) => {
                drop(generator);
                let mut state = self.state.write().unwrap();
                *state = NeuralLoadState::Failed(e.clone());
                eprintln!("Error loading item {}: {}", current_count, e);
                None
            }
        }
    }

    /// Load remaining items (collect all)
    pub fn load_all(&self) -> Vec<T> {
        // Load all remaining items
        while self.load_next().is_some() {}
        
        // Return cloned values
        self.loaded_values.read().unwrap().clone()
    }

    /// Load all remaining items without storing (just trigger loading)
    pub fn rest(&self) -> bool {
        let mut collected = false;
        while let Some(_) = self.load_next() {
            collected = true;
        }
        collected
    }

    /// Get value at index (loads up to that index if needed)
    pub fn get(&self, index: usize) -> Option<T> {
        // Check if already loaded
        {
            let loaded = self.loaded_values.read().unwrap();
            if index < loaded.len() {
                return Some(loaded[index].clone());
            }
        }
        
        // Need to load more items
        while self.loaded_count() <= index {
            if self.load_next().is_none() {
                break;
            }
        }
        
        // Try again
        let loaded = self.loaded_values.read().unwrap();
        if index < loaded.len() {
            Some(loaded[index].clone())
        } else {
            None
        }
    }

    /// Check and trigger completion callback
    fn check_and_trigger_completion(&self) {
        if let Some(callback) = self.on_complete.take() {
            if let Err(e) = callback() {
                eprintln!("Completion callback error: {}", e);
            }
        }
    }

    /// Get iterator over all items (lazy loading as needed)
    pub fn iter(&self) -> NeuralLazyIterator<T> {
        NeuralLazyIterator {
            loader: self,
            position: 0,
        }
    }

    /// Get iterator over items with index
    pub fn enumerate(&self) -> NeuralLazyEnumerate<T> {
        NeuralLazyEnumerate {
            loader: self,
            position: 0,
        }
    }

    /// Preload items up to count (eager loading with limit)
    pub fn preload(&self, count: usize) -> usize {
        let mut loaded = 0;
        while self.loaded_count() < count {
            if self.load_next().is_none() {
                break;
            }
            loaded += 1;
        }
        loaded
    }

    /// Get memory estimate for fully loaded state
    pub fn estimated_memory_bytes(&self) -> u64 {
        self.spatial_context.estimated_bytes
    }

    /// Get load progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        self.loaded_count() as f64 / self.total_count as f64
    }

    /// Reset loader state (clears loaded values)
    pub fn reset(&self) {
        let mut values = self.loaded_values.write().unwrap();
        values.clear();
        drop(values);
        
        let mut state = self.state.write().unwrap();
        *state = NeuralLoadState::Dormant;
    }

    /// Create batch loader for parallel loading
    pub fn batch_loader(&self, batch_size: usize) -> NeuralBatchLoader<T> {
        NeuralBatchLoader {
            loader: self,
            batch_size,
            current_position: Arc::new(Mutex::new(0)),
        }
    }
}

/// Iterator for lazy loading
pub struct NeuralLazyIterator<'a, T: Clone + Send> {
    loader: &'a TitanNeuralLazyLoader<T>,
    position: usize,
}

impl<'a, T: Clone + Send> Iterator for NeuralLazyIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.loader.get(self.position);
        if item.is_some() {
            self.position += 1;
        }
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.loader.total_count as usize).saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

/// Enumerate iterator with index
pub struct NeuralLazyEnumerate<'a, T: Clone + Send> {
    loader: &'a TitanNeuralLazyLoader<T>,
    position: usize,
}

impl<'a, T: Clone + Send> Iterator for NeuralLazyEnumerate<'a, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.loader.get(self.position)?;
        let result = (self.position, item);
        self.position += 1;
        Some(result)
    }
}

/// Batch loader for parallel/efficient loading
pub struct NeuralBatchLoader<'a, T: Clone + Send> {
    loader: &'a TitanNeuralLazyLoader<T>,
    batch_size: usize,
    current_position: Arc<Mutex<usize>>,
}

impl<'a, T: Clone + Send> NeuralBatchLoader<'a, T> {
    /// Load next batch
    pub fn next_batch(&self) -> Option<Vec<T>> {
        let mut pos = self.current_position.lock().unwrap();
        let start = *pos;
        
        // Check if we have more to load
        if start >= self.loader.total_count as usize {
            return None;
        }
        
        // Calculate batch size
        let remaining = self.loader.total_count as usize - start;
        let batch_size = remaining.min(self.batch_size);
        
        drop(pos); // Release lock
        
        // Ensure items are loaded
        let end = start + batch_size;
        while self.loader.loaded_count() < end {
            if self.loader.load_next().is_none() {
                break;
            }
        }
        
        // Collect batch
        let values = self.loader.loaded_values.read().unwrap();
        let batch: Vec<T> = values[start..values.len().min(end)]
            .iter()
            .cloned()
            .collect();
        drop(values);
        
        // Update position
        let mut pos = self.current_position.lock().unwrap();
        *pos += batch.len();
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Get current position
    pub fn position(&self) -> usize {
        *self.current_position.lock().unwrap()
    }

    /// Reset batch position
    pub fn reset(&self) {
        let mut pos = self.current_position.lock().unwrap();
        *pos = 0;
    }
}

/// Spatial-aware lazy loader registry
pub struct NeuralLazyRegistry<T: Clone + Send> {
    loaders: HashMap<String, Arc<TitanNeuralLazyLoader<T>>>,
    spatial_index: HashMap<(u32, u32, u32), Vec<String>>,
}

impl<T: Clone + Send + 'static> NeuralLazyRegistry<T> {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
            spatial_index: HashMap::new(),
        }
    }

    /// Register a lazy loader
    pub fn register(
        &mut self,
        name: String,
        loader: Arc<TitanNeuralLazyLoader<T>>,
    ) {
        // Add to spatial index
        let spatial = loader.spatial_context();
        let key = (spatial.dimensions.width, spatial.dimensions.height, spatial.dimensions.depth);
        self.spatial_index.entry(key).or_default().push(name.clone());
        
        self.loaders.insert(name, loader);
    }

    /// Get loader by name
    pub fn get(&self, name: &str) -> Option<Arc<TitanNeuralLazyLoader<T>>> {
        self.loaders.get(name).cloned()
    }

    /// Find loaders by spatial region
    pub fn find_by_spatial(&self, width: u32, height: u32, depth: u32) -> Vec<&str> {
        self.spatial_index
            .get(&(width, height, depth))
            .map(|names| names.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Preload all loaders up to certain priority
    pub fn preload_by_priority(&self, min_priority: u32) -> usize {
        let mut total_loaded = 0;
        
        for loader in self.loaders.values() {
            if loader.spatial_context().load_priority >= min_priority {
                let loaded = loader.preload(1); // At least load first item
                total_loaded += loaded;
            }
        }
        
        total_loaded
    }

    /// Total memory estimate for all loaders
    pub fn total_memory_estimate(&self) -> u64 {
        self.loaders
            .values()
            .map(|l| l.estimated_memory_bytes())
            .sum()
    }

    /// Count of registered loaders
    pub fn len(&self) -> usize {
        self.loaders.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.loaders.is_empty()
    }
}

impl<T: Clone + Send> Default for NeuralLazyRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating lazy loaders
pub struct NeuralLazyLoaderBuilder<T: Clone + Send> {
    count: u64,
    spatial_context: NeuralSpatialContext,
    generator: Option<Box<dyn FnMut() -> Result<T, NeuralLoadError> + Send>>,
    on_complete: Option<NeuralSuccessCallback>,
}

impl<T: Clone + Send + 'static> NeuralLazyLoaderBuilder<T> {
    /// Create new builder
    pub fn new(count: u64) -> Self {
        Self {
            count,
            spatial_context: NeuralSpatialContext::default(),
            generator: None,
            on_complete: None,
        }
    }

    /// With spatial context
    pub fn spatial(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.spatial_context = NeuralSpatialContext::new(width, height, depth);
        self
    }

    /// With custom spatial context
    pub fn with_spatial_context(mut self, context: NeuralSpatialContext) -> Self {
        self.spatial_context = context;
        self
    }

    /// With generator function
    pub fn generator<F>(mut self, gen: F) -> Self
    where
        F: FnMut() -> Result<T, NeuralLoadError> + Send + 'static,
    {
        self.generator = Some(Box::new(gen));
        self
    }

    /// With success callback
    pub fn on_complete<F>(mut self, callback: F) -> Self
    where
        F: FnOnce() -> Result<(), NeuralLoadError> + Send + 'static,
    {
        self.on_complete = Some(Box::new(callback));
        self
    }

    /// Build lazy loader
    pub fn build(self) -> Result<TitanNeuralLazyLoader<T>, NeuralLoadError> {
        let generator = self.generator
            .ok_or_else(|| NeuralLoadError::SerializationError {
                reason: "Generator function required".to_string(),
            })?;
        
        let mut loader = TitanNeuralLazyLoader::new(self.count, self.spatial_context, generator);
        
        if let Some(callback) = self.on_complete {
            // This is a bit tricky since we already constructed the loader
            // In practice, we'd redesign this
            loader.on_complete = Some(callback);
        }
        
        Ok(loader)
    }
}

/// Type aliases for convenience
pub type LazyLoader<T> = TitanNeuralLazyLoader<T>;
pub type LazyRegistry<T> = NeuralLazyRegistry<T>;
pub type LazyBuilder<T> = NeuralLazyLoaderBuilder<T>;
pub type LoadState = NeuralLoadState;
pub type LoadError = NeuralLoadError;
pub type SpatialContext = NeuralSpatialContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_loading() {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();
        
        let loader = TitanNeuralLazyLoader::new(
            5,
            SpatialContext::new(10, 10, 10),
            move || {
                let mut count = counter_clone.lock().unwrap();
                *count += 1;
                Ok(*count - 1)
            },
        );
        
        // Load first 3 items
        assert_eq!(loader.load_next(), Some(0));
        assert_eq!(loader.load_next(), Some(1));
        assert_eq!(loader.load_next(), Some(2));
        
        assert_eq!(loader.loaded_count(), 3);
        assert!(!loader.is_complete());
        
        // Load remaining
        loader.rest();
        
        assert_eq!(loader.loaded_count(), 5);
        assert!(loader.is_complete());
    }

    #[test]
    fn test_lazy_iterator() {
        let loader = TitanNeuralLazyLoader::new(
            5,
            SpatialContext::new(1, 1, 1),
            || Ok(42u32),
        );
        
        let mut count = 0;
        for item in loader.iter() {
            assert_eq!(item, 42);
            count += 1;
        }
        
        assert_eq!(count, 5);
    }

    #[test]
    fn test_batch_loader() {
        let loader = TitanNeuralLazyLoader::new(
            10,
            SpatialContext::new(1, 1, 1),
            || Ok(1u32),
        );
        
        let batch_loader = loader.batch_loader(3);
        
        let batch1 = batch_loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 3);
        
        let batch2 = batch_loader.next_batch().unwrap();
        assert_eq!(batch2.len(), 3);
        
        let batch3 = batch_loader.next_batch().unwrap();
        assert_eq!(batch3.len(), 3);
        
        let batch4 = batch_loader.next_batch().unwrap();
        assert_eq!(batch4.len(), 1);
        
        assert!(batch_loader.next_batch().is_none());
    }

    #[test]
    fn test_spatial_context() {
        let ctx = SpatialContext::new(256, 256, 128)
            .with_priority(200)
            .with_alignment(128);
        
        assert_eq!(ctx.dimensions.width, 256);
        assert_eq!(ctx.dimensions.height, 256);
        assert_eq!(ctx.dimensions.depth, 128);
        assert_eq!(ctx.load_priority, 200);
        assert_eq!(ctx.alignment, 128);
        assert_eq!(ctx.volume(), 256 * 256 * 128);
    }

    #[test]
    fn test_lazy_registry() {
        let mut registry = LazyRegistry::<u32>::new();
        
        let loader1 = Arc::new(TitanNeuralLazyLoader::new(
            10,
            SpatialContext::new(256, 256, 128),
            || Ok(1u32),
        ));
        
        let loader2 = Arc::new(TitanNeuralLazyLoader::new(
            20,
            SpatialContext::new(512, 512, 256),
            || Ok(2u32),
        ));
        
        registry.register("loader1".to_string(), loader1);
        registry.register("loader2".to_string(), loader2);
        
        assert_eq!(registry.len(), 2);
        
        let found = registry.find_by_spatial(256, 256, 128);
        assert!(found.contains(&"loader1"));
        
        let estimate = registry.total_memory_estimate();
        assert!(estimate > 0);
    }

    #[test]
    fn test_load_state_transitions() {
        let loader = TitanNeuralLazyLoader::new(
            3,
            SpatialContext::new(1, 1, 1),
            || Ok(1u32),
        );
        
        assert!(matches!(loader.state(), LoadState::Dormant));
        
        loader.load_next();
        assert!(matches!(loader.state(), LoadState::Partial(1, 3)));
        
        loader.rest();
        assert!(matches!(loader.state(), LoadState::Complete));
    }

    #[test]
    fn test_progress_tracking() {
        let loader = TitanNeuralLazyLoader::new(
            10,
            SpatialContext::new(1, 1, 1),
            || Ok(1u32),
        );
        
        assert_eq!(loader.progress(), 0.0);
        
        loader.preload(5);
        assert_eq!(loader.progress(), 0.5);
        
        loader.rest();
        assert_eq!(loader.progress(), 1.0);
    }
}