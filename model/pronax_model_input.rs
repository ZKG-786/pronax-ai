use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// 3D multimodal embedding data with spatial metadata
/// Enhanced version of Ollama's Multimodal struct with 3D positioning
#[derive(Debug, Clone)]
pub struct NeuralMultimodalEmbedding {
    /// Tensor handle (opaque, backend-specific)
    pub tensor: Option<Arc<dyn Any + Send + Sync>>,
    
    /// Implementation-specific opaque metadata
    /// Stores layout info, preprocessing parameters, or raw data
    pub metadata: Option<Arc<dyn Any + Send + Sync>>,
    
    /// 3D spatial dimensions of this embedding
    pub spatial_dims: SpatialTensorMetadata,
    
    /// Position in 3D embedding space
    pub spatial_position: ConversionCoordinate,
    
    /// Embedding type identifier
    pub embedding_type: MultimodalType,
    
    /// Cache hash for deduplication
    pub cache_hash: u64,
}

impl NeuralMultimodalEmbedding {
    /// Create new embedding with 3D spatial data
    pub fn new(
        spatial_dims: SpatialTensorMetadata,
        embedding_type: MultimodalType,
        cache_hash: u64,
    ) -> Self {
        Self {
            tensor: None,
            metadata: None,
            spatial_dims,
            spatial_position: ConversionCoordinate::standard(),
            embedding_type,
            cache_hash,
        }
    }
    
    /// Create from raw bytes (image/audio/video)
    pub fn from_raw_data(
        data: Vec<u8>,
        dims: SpatialTensorMetadata,
        mtype: MultimodalType,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let cache_hash = hasher.finish();
        
        Self {
            tensor: None,
            metadata: Some(Arc::new(data)),
            spatial_dims: dims,
            spatial_position: ConversionCoordinate::new(
                0,
                dims.height as u16,
                dims.depth as u8,
                1.0,
            ),
            embedding_type: mtype,
            cache_hash,
        }
    }
    
    /// Set tensor handle
    pub fn with_tensor(mut self, tensor: Arc<dyn Any + Send + Sync>) -> Self {
        self.tensor = Some(tensor);
        self
    }
    
    /// Set metadata
    pub fn with_metadata(mut self, metadata: Arc<dyn Any + Send + Sync>) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Set 3D spatial position
    pub fn with_position(mut self, pos: ConversionCoordinate) -> Self {
        self.spatial_position = pos;
        self
    }
    
    /// Check if tensor is loaded
    pub fn has_tensor(&self) -> bool {
        self.tensor.is_some()
    }
    
    /// Get total elements in embedding
    pub fn element_count(&self) -> u64 {
        self.spatial_dims.volume()
    }
}

impl Default for NeuralMultimodalEmbedding {
    fn default() -> Self {
        Self::new(
            SpatialTensorMetadata::new(1, 1, 1),
            MultimodalType::Unknown,
            0,
        )
    }
}

/// Multimodal data type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultimodalType {
    /// Unknown/unspecified type
    Unknown,
    /// Image data (JPEG, PNG, etc.)
    Image,
    /// Audio data
    Audio,
    /// Video data
    Video,
    /// Document/PDF
    Document,
    /// 3D mesh/point cloud
    Mesh3D,
    /// Custom user-defined type
    Custom(u32),
}

impl MultimodalType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Document => "document",
            Self::Mesh3D => "mesh3d",
            Self::Custom(_) => "custom",
        }
    }
    
    /// Check if type requires vision processing
    pub fn requires_vision(&self) -> bool {
        matches!(self, Self::Image | Self::Video)
    }
    
    /// Get default 3D dimensions for this type
    pub fn default_dimensions(&self) -> SpatialTensorMetadata {
        match self {
            Self::Image => SpatialTensorMetadata::new(224, 224, 3),
            Self::Audio => SpatialTensorMetadata::new(16000, 1, 1), // 1 second at 16kHz
            Self::Video => SpatialTensorMetadata::new(224, 224, 16), // 16 frames
            Self::Mesh3D => SpatialTensorMetadata::new(1024, 1024, 1024),
            _ => SpatialTensorMetadata::new(1, 1, 1),
        }
    }
}

/// Neural input element - single token or multimodal data
/// Enhanced version of Ollama's Input with 3D spatial tracking
#[derive(Debug, Clone)]
pub struct NeuralInput {
    /// Token ID (-1 for multimodal-only)
    pub token_id: i32,
    
    /// Multimodal embeddings associated with this input
    pub multimodal: Vec<NeuralMultimodalEmbedding>,
    
    /// Cache hash for multimodal data
    pub multimodal_hash: u64,
    
    /// Force batching constraint: these many tokens must be processed together
    pub batch_constraint: usize,
    
    /// Position in 3D input space
    pub spatial_position: (f32, f32, f32),
    
    /// Sequence ID for this input
    pub sequence_id: u32,
    
    /// Position within sequence
    pub position_in_sequence: usize,
    
    /// Input type classification
    pub input_type: InputType,
}

impl NeuralInput {
    /// Create text token input
    pub fn token(token_id: i32, position: usize, sequence: u32) -> Self {
        Self {
            token_id,
            multimodal: Vec::new(),
            multimodal_hash: 0,
            batch_constraint: 1,
            spatial_position: (position as f32, 0.0, 0.0),
            sequence_id: sequence,
            position_in_sequence: position,
            input_type: InputType::Text,
        }
    }
    
    /// Create multimodal input
    pub fn multimodal(
        embeddings: Vec<NeuralMultimodalEmbedding>,
        position: usize,
        sequence: u32,
    ) -> Self {
        let hash = embeddings.first().map(|e| e.cache_hash).unwrap_or(0);
        let mtype = embeddings.first().map(|e| e.embedding_type).unwrap_or(MultimodalType::Unknown);
        
        Self {
            token_id: -1,
            multimodal: embeddings,
            multimodal_hash: hash,
            batch_constraint: 1,
            spatial_position: (position as f32, 0.0, 0.0),
            sequence_id: sequence,
            position_in_sequence: position,
            input_type: InputType::Multimodal(mtype),
        }
    }
    
    /// Create mixed text + multimodal input
    pub fn mixed(
        token_id: i32,
        embeddings: Vec<NeuralMultimodalEmbedding>,
        position: usize,
        sequence: u32,
    ) -> Self {
        let mut input = Self::token(token_id, position, sequence);
        input.multimodal = embeddings;
        input.input_type = InputType::Mixed;
        input
    }
    
    /// Set batch constraint
    pub fn with_batch_constraint(mut self, count: usize) -> Self {
        self.batch_constraint = count;
        self
    }
    
    /// Set 3D spatial position
    pub fn with_spatial_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.spatial_position = (x, y, z);
        self
    }
    
    /// Check if this is a text-only input
    pub fn is_text(&self) -> bool {
        matches!(self.input_type, InputType::Text)
    }
    
    /// Check if this contains multimodal data
    pub fn is_multimodal(&self) -> bool {
        matches!(self.input_type, InputType::Multimodal(_) | InputType::Mixed)
    }
    
    /// Get total embedding count
    pub fn embedding_count(&self) -> usize {
        self.multimodal.len()
    }
    
    /// Validate input consistency
    pub fn validate(&self) -> Result<(), InputError> {
        if self.token_id < 0 && self.multimodal.is_empty() {
            return Err(InputError::InvalidInput(
                "Input must have either token or multimodal data".to_string()
            ));
        }
        
        if self.batch_constraint == 0 {
            return Err(InputError::InvalidInput(
                "Batch constraint must be at least 1".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Input type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputType {
    Text,
    Multimodal(MultimodalType),
    Mixed,
    Padding,
    Special,
}

impl InputType {
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text | Self::Special | Self::Padding)
    }
    
    pub fn is_multimodal(&self) -> bool {
        matches!(self, Self::Multimodal(_) | Self::Mixed)
    }
}

/// Input error types
#[derive(Debug, Clone)]
pub enum InputError {
    InvalidInput(String),
    BatchTooLarge(String),
    SequenceError(String),
    MultimodalError(String),
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            Self::BatchTooLarge(s) => write!(f, "Batch too large: {}", s),
            Self::SequenceError(s) => write!(f, "Sequence error: {}", s),
            Self::MultimodalError(s) => write!(f, "Multimodal error: {}", s),
        }
    }
}

impl std::error::Error for InputError {}

/// Multimodal index with 3D tracking
/// Enhanced version of Ollama's MultimodalIndex
#[derive(Debug, Clone)]
pub struct NeuralMultimodalIndex {
    /// Index into inputs slice
    pub input_index: usize,
    
    /// Associated multimodal embeddings
    pub embeddings: Vec<NeuralMultimodalEmbedding>,
    
    /// 3D bounding box for spatial layout
    pub spatial_bounds: SpatialTensorMetadata,
    
    /// Position in 3D space
    pub position_3d: (f32, f32, f32),
}

impl NeuralMultimodalIndex {
    /// Create new multimodal index
    pub fn new(index: usize, embeddings: Vec<NeuralMultimodalEmbedding>) -> Self {
        let bounds = embeddings.first()
            .map(|e| e.spatial_dims)
            .unwrap_or_else(|| SpatialTensorMetadata::new(1, 1, 1));
        
        Self {
            input_index: index,
            embeddings,
            spatial_bounds: bounds,
            position_3d: (0.0, 0.0, 0.0),
        }
    }
    
    /// Total elements across all embeddings
    pub fn total_elements(&self) -> u64 {
        self.embeddings.iter().map(|e| e.element_count()).sum()
    }
}

/// 3D-aware batch for model forward pass
/// Enhanced version of Ollama's Batch with spatial metadata
#[derive(Debug, Clone)]
pub struct NeuralBatch {
    /// Input tokens tensor (opaque handle)
    pub input_tensor: Option<Arc<dyn Any + Send + Sync>>,
    
    /// Output indices tensor (opaque handle)
    pub output_tensor: Option<Arc<dyn Any + Send + Sync>>,
    
    /// Individual input elements
    pub inputs: Vec<NeuralInput>,
    
    /// Positions for each input in their sequences
    pub positions: Vec<i32>,
    
    /// Sequence ID for each input
    pub sequence_ids: Vec<u32>,
    
    /// 3D spatial coordinates for each input
    pub spatial_coords: Vec<(f32, f32, f32)>,
    
    /// Multimodal indices with 3D tracking
    pub multimodal_indices: Vec<NeuralMultimodalIndex>,
    
    /// Batch metadata
    pub metadata: BatchMetadata,
}

impl NeuralBatch {
    /// Create empty batch
    pub fn new() -> Self {
        Self {
            input_tensor: None,
            output_tensor: None,
            inputs: Vec::new(),
            positions: Vec::new(),
            sequence_ids: Vec::new(),
            spatial_coords: Vec::new(),
            multimodal_indices: Vec::new(),
            metadata: BatchMetadata::default(),
        }
    }
    
    /// Create batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            input_tensor: None,
            output_tensor: None,
            inputs: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            sequence_ids: Vec::with_capacity(capacity),
            spatial_coords: Vec::with_capacity(capacity),
            multimodal_indices: Vec::new(),
            metadata: BatchMetadata::default(),
        }
    }
    
    /// Add input to batch
    pub fn add_input(&mut self, input: NeuralInput) -> Result<(), InputError> {
        input.validate()?;
        
        self.positions.push(input.position_in_sequence as i32);
        self.sequence_ids.push(input.sequence_id);
        self.spatial_coords.push(input.spatial_position);
        
        // Add multimodal index if present
        if !input.multimodal.is_empty() {
            let idx = self.inputs.len();
            self.multimodal_indices.push(NeuralMultimodalIndex::new(
                idx,
                input.multimodal.clone(),
            ));
        }
        
        self.inputs.push(input);
        self.metadata.total_tokens += 1;
        
        Ok(())
    }
    
    /// Batch size
    pub fn len(&self) -> usize {
        self.inputs.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
    
    /// Get unique sequence count
    pub fn sequence_count(&self) -> usize {
        let mut unique = self.sequence_ids.clone();
        unique.sort_unstable();
        unique.dedup();
        unique.len()
    }
    
    /// Get multimodal count
    pub fn multimodal_count(&self) -> usize {
        self.multimodal_indices.len()
    }
    
    /// Validate batch consistency
    pub fn validate(&self) -> Result<(), InputError> {
        let len = self.len();
        
        if self.positions.len() != len {
            return Err(InputError::BatchTooLarge(
                format!("Positions {} != inputs {}", self.positions.len(), len)
            ));
        }
        
        if self.sequence_ids.len() != len {
            return Err(InputError::BatchTooLarge(
                format!("Sequence IDs {} != inputs {}", self.sequence_ids.len(), len)
            ));
        }
        
        if self.spatial_coords.len() != len {
            return Err(InputError::BatchTooLarge(
                format!("Spatial coords {} != inputs {}", self.spatial_coords.len(), len)
            ));
        }
        
        // Validate all inputs
        for (i, input) in self.inputs.iter().enumerate() {
            if let Err(e) = input.validate() {
                return Err(InputError::InvalidInput(
                    format!("Input {}: {}", i, e)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Split batch by sequence
    pub fn split_by_sequence(&self) -> Vec<Vec<&NeuralInput>> {
        let mut map: HashMap<u32, Vec<&NeuralInput>> = HashMap::new();
        
        for input in &self.inputs {
            map.entry(input.sequence_id)
                .or_default()
                .push(input);
        }
        
        map.into_values().collect()
    }
    
    /// Get 3D spatial bounds of entire batch
    pub fn spatial_bounds(&self) -> SpatialTensorMetadata {
        if self.spatial_coords.is_empty() {
            return SpatialTensorMetadata::new(1, 1, 1);
        }
        
        let max_x = self.spatial_coords.iter().map(|(x, _, _)| *x).fold(0.0f32, f32::max);
        let max_y = self.spatial_coords.iter().map(|(_, y, _)| *y).fold(0.0f32, f32::max);
        let max_z = self.spatial_coords.iter().map(|(_, _, z)| *z).fold(0.0f32, f32::max);
        
        SpatialTensorMetadata::new(max_x as u32 + 1, max_y as u32 + 1, max_z as u32 + 1)
    }
}

impl Default for NeuralBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch metadata
#[derive(Debug, Clone, Copy)]
pub struct BatchMetadata {
    pub total_tokens: usize,
    pub total_multimodal: usize,
    pub max_sequence_length: usize,
    pub num_sequences: usize,
}

impl Default for BatchMetadata {
    fn default() -> Self {
        Self {
            total_tokens: 0,
            total_multimodal: 0,
            max_sequence_length: 0,
            num_sequences: 0,
        }
    }
}

/// Batch builder for constructing batches
pub struct BatchBuilder {
    batch: NeuralBatch,
    current_sequence: u32,
    position_counter: usize,
}

impl BatchBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            batch: NeuralBatch::new(),
            current_sequence: 0,
            position_counter: 0,
        }
    }
    
    /// Set current sequence
    pub fn sequence(mut self, seq: u32) -> Self {
        self.current_sequence = seq;
        self.position_counter = 0;
        self
    }
    
    /// Add text token
    pub fn add_token(mut self, token_id: i32) -> Self {
        let input = NeuralInput::token(token_id, self.position_counter, self.current_sequence);
        self.batch.add_input(input).ok();
        self.position_counter += 1;
        self
    }
    
    /// Add multimodal data
    pub fn add_multimodal(mut self, embeddings: Vec<NeuralMultimodalEmbedding>) -> Self {
        let input = NeuralInput::multimodal(embeddings, self.position_counter, self.current_sequence);
        self.batch.add_input(input).ok();
        self.position_counter += 1;
        self
    }
    
    /// Add mixed input
    pub fn add_mixed(mut self, token_id: i32, embeddings: Vec<NeuralMultimodalEmbedding>) -> Self {
        let input = NeuralInput::mixed(token_id, embeddings, self.position_counter, self.current_sequence);
        self.batch.add_input(input).ok();
        self.position_counter += 1;
        self
    }
    
    /// Build the batch
    pub fn build(self) -> NeuralBatch {
        self.batch
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for input handling
pub mod input_utils {
    use super::*;
    
    /// Compute 3D position for token in sequence
    pub fn compute_token_position(
        seq_pos: usize,
        layer_idx: usize,
        head_idx: usize,
        max_seq_len: usize,
    ) -> (f32, f32, f32) {
        let x = (seq_pos as f32 / max_seq_len as f32) * 1000.0;
        let y = layer_idx as f32 * 10.0;
        let z = head_idx as f32 * 5.0;
        (x, y, z)
    }
    
    /// Compute cache hash for multimodal data
    pub fn compute_cache_hash(data: &[u8], mtype: MultimodalType) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        mtype.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Validate batch size limits
    pub fn validate_batch_size(batch: &NeuralBatch, max_size: usize) -> Result<(), InputError> {
        if batch.len() > max_size {
            return Err(InputError::BatchTooLarge(
                format!("Batch size {} exceeds maximum {}", batch.len(), max_size)
            ));
        }
        Ok(())
    }
    
    /// Merge multiple batches
    pub fn merge_batches(batches: &[NeuralBatch]) -> NeuralBatch {
        let mut merged = NeuralBatch::new();
        
        for batch in batches {
            for input in &batch.inputs {
                merged.add_input(input.clone()).ok();
            }
        }
        
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multimodal_embedding() {
        let dims = SpatialTensorMetadata::new(224, 224, 3);
        let emb = NeuralMultimodalEmbedding::new(dims, MultimodalType::Image, 12345);
        
        assert_eq!(emb.element_count(), 224 * 224 * 3);
        assert_eq!(emb.embedding_type, MultimodalType::Image);
        assert!(!emb.has_tensor());
    }
    
    #[test]
    fn test_multimodal_from_raw() {
        let data = vec![1, 2, 3, 4, 5];
        let emb = NeuralMultimodalEmbedding::from_raw_data(
            data.clone(),
            SpatialTensorMetadata::new(100, 100, 3),
            MultimodalType::Image,
        );
        
        assert!(emb.cache_hash != 0);
        assert_eq!(emb.embedding_type, MultimodalType::Image);
    }
    
    #[test]
    fn test_neural_input_token() {
        let input = NeuralInput::token(100, 0, 0);
        
        assert_eq!(input.token_id, 100);
        assert!(input.is_text());
        assert!(!input.is_multimodal());
        assert!(input.validate().is_ok());
    }
    
    #[test]
    fn test_neural_input_multimodal() {
        let emb = NeuralMultimodalEmbedding::new(
            SpatialTensorMetadata::new(224, 224, 3),
            MultimodalType::Image,
            12345,
        );
        
        let input = NeuralInput::multimodal(vec![emb], 0, 0);
        
        assert_eq!(input.token_id, -1);
        assert!(input.is_multimodal());
        assert_eq!(input.embedding_count(), 1);
    }
    
    #[test]
    fn test_invalid_input() {
        let input = NeuralInput {
            token_id: -1,
            multimodal: Vec::new(),
            multimodal_hash: 0,
            batch_constraint: 1,
            spatial_position: (0.0, 0.0, 0.0),
            sequence_id: 0,
            position_in_sequence: 0,
            input_type: InputType::Text,
        };
        
        assert!(input.validate().is_err());
    }
    
    #[test]
    fn test_batch_builder() {
        let batch = BatchBuilder::new()
            .sequence(0)
            .add_token(1)
            .add_token(2)
            .add_token(3)
            .build();
        
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.sequence_count(), 1);
    }
    
    #[test]
    fn test_batch_validation() {
        let mut batch = NeuralBatch::new();
        
        for i in 0..5 {
            let input = NeuralInput::token(i, i, 0);
            batch.add_input(input).unwrap();
        }
        
        assert!(batch.validate().is_ok());
    }
    
    #[test]
    fn test_spatial_bounds() {
        let mut batch = NeuralBatch::new();
        
        for i in 0..3 {
            let input = NeuralInput::token(i, i, 0)
                .with_spatial_position(i as f32 * 10.0, i as f32 * 5.0, i as f32);
            batch.add_input(input).unwrap();
        }
        
        let bounds = batch.spatial_bounds();
        assert!(bounds.width >= 20);
        assert!(bounds.height >= 10);
    }
    
    #[test]
    fn test_multimodal_type() {
        assert!(MultimodalType::Image.requires_vision());
        assert!(!MultimodalType::Audio.requires_vision());
        assert_eq!(MultimodalType::Image.as_str(), "image");
    }
    
    #[test]
    fn test_input_utils() {
        let pos = input_utils::compute_token_position(50, 5, 2, 1000);
        assert!(pos.0 > 0.0);
        assert!(pos.1 > 0.0);
        assert!(pos.2 > 0.0);
        
        let hash1 = input_utils::compute_cache_hash(b"test", MultimodalType::Image);
        let hash2 = input_utils::compute_cache_hash(b"test", MultimodalType::Image);
        let hash3 = input_utils::compute_cache_hash(b"different", MultimodalType::Image);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_merge_batches() {
        let batch1 = BatchBuilder::new().add_token(1).add_token(2).build();
        let batch2 = BatchBuilder::new().add_token(3).add_token(4).build();
        
        let merged = input_utils::merge_batches(&[batch1, batch2]);
        assert_eq!(merged.len(), 4);
    }
}
