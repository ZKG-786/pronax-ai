use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput};
use crate::tokenizer::pronax_sentencepiece::NeuralSentencePieceTokenizer;

/// Gemma3 embedding model errors
#[derive(Debug, Clone)]
pub enum Gemma3EmbedError {
    InvalidConfiguration(String),
    ForwardError(String),
    PoolingError(String),
    NormalizationError(String),
    DenseLayerError(String),
}

impl std::fmt::Display for Gemma3EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "Invalid config: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::PoolingError(s) => write!(f, "Pooling error: {}", s),
            Self::NormalizationError(s) => write!(f, "Normalization error: {}", s),
            Self::DenseLayerError(s) => write!(f, "Dense layer error: {}", s),
        }
    }
}

impl std::error::Error for Gemma3EmbedError {}

/// Pooling types for embedding extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma3PoolingType {
    /// No pooling, use all tokens
    None = 0,
    /// Mean pooling across sequence
    Mean = 1,
    /// Max pooling across sequence
    Max = 2,
    /// Use CLS token (first token)
    Cls = 3,
    /// Use last token
    Last = 4,
}

impl Gemma3PoolingType {
    /// Convert from integer
    pub fn from_int(value: u32) -> Self {
        match value {
            1 => Self::Mean,
            2 => Self::Max,
            3 => Self::Cls,
            4 => Self::Last,
            _ => Self::None,
        }
    }
    
    /// Apply pooling to hidden states
    pub fn apply(&self, hidden_states: &[f32], seq_len: usize, hidden_size: usize) -> Vec<f32> {
        match self {
            Self::None => {
                // Return all tokens (already flattened)
                hidden_states.to_vec()
            }
            Self::Mean => {
                // Average across sequence dimension
                let mut pooled = vec![0.0; hidden_size];
                for token_idx in 0..seq_len {
                    let token_start = token_idx * hidden_size;
                    for h in 0..hidden_size {
                        if token_start + h < hidden_states.len() {
                            pooled[h] += hidden_states[token_start + h];
                        }
                    }
                }
                for h in &mut pooled {
                    *h /= seq_len as f32;
                }
                pooled
            }
            Self::Max => {
                // Max across sequence dimension
                let mut pooled = vec![f32::NEG_INFINITY; hidden_size];
                for token_idx in 0..seq_len {
                    let token_start = token_idx * hidden_size;
                    for h in 0..hidden_size {
                        if token_start + h < hidden_states.len() {
                            pooled[h] = pooled[h].max(hidden_states[token_start + h]);
                        }
                    }
                }
                pooled
            }
            Self::Cls => {
                // Return first token (CLS)
                hidden_states[..hidden_size.min(hidden_states.len())].to_vec()
            }
            Self::Last => {
                // Return last token
                let last_token_start = (seq_len.saturating_sub(1)) * hidden_size;
                if last_token_start + hidden_size <= hidden_states.len() {
                    hidden_states[last_token_start..last_token_start + hidden_size].to_vec()
                } else {
                    vec![0.0; hidden_size]
                }
            }
        }
    }
}

impl Default for Gemma3PoolingType {
    fn default() -> Self {
        Self::Mean
    }
}

/// 3D-aware dense linear layer
#[derive(Debug, Clone)]
pub struct Gemma3DenseLayer3D {
    /// Weights [out_features, in_features]
    pub weights: Vec<f32>,
    /// Bias [out_features]
    pub bias: Vec<f32>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma3DenseLayer3D {
    /// Create new dense layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: vec![0.0; out_features * in_features],
            bias: vec![0.0; out_features],
            in_features,
            out_features,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let batch_size = input.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];
        
        for batch_idx in 0..batch_size {
            let input_start = batch_idx * self.in_features;
            let output_start = batch_idx * self.out_features;
            
            for out_idx in 0..self.out_features {
                let mut sum = self.bias[out_idx];
                for in_idx in 0..self.in_features {
                    let weight_idx = out_idx * self.in_features + in_idx;
                    if input_start + in_idx < input.len() && weight_idx < self.weights.len() {
                        sum += input[input_start + in_idx] * self.weights[weight_idx];
                    }
                }
                if output_start + out_idx < output.len() {
                    output[output_start + out_idx] = sum;
                }
            }
        }
        
        output
    }
}

/// 3D-aware Gemma3 embedding configuration
#[derive(Debug, Clone, Copy)]
pub struct Gemma3EmbedConfig3D {
    /// Hidden dimension from text model
    pub hidden_size: usize,
    /// Dense layer 0 output size
    pub dense0_out: usize,
    /// Dense layer 1 output size
    pub dense1_out: usize,
    /// Pooling type
    pub pooling_type: Gemma3PoolingType,
    /// L2 normalization epsilon
    pub l2_eps: f32,
    /// 3D spatial depth
    pub spatial_depth: u8,
}

impl Gemma3EmbedConfig3D {
    /// Default Gemma3 embedding configuration
    pub fn default_config() -> Self {
        Self {
            hidden_size: 2304,
            dense0_out: 4096,
            dense1_out: 1024,
            pooling_type: Gemma3PoolingType::Mean,
            l2_eps: 1e-12,
            spatial_depth: 64,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), Gemma3EmbedError> {
        if self.hidden_size == 0 {
            return Err(Gemma3EmbedError::InvalidConfiguration(
                "hidden_size cannot be zero".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for Gemma3EmbedConfig3D {
    fn default() -> Self {
        Self::default_config()
    }
}

/// 3D embedding result with spatial metadata
#[derive(Debug, Clone)]
pub struct Gemma3EmbeddingResult3D {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Embedding dimension
    pub dimension: usize,
    /// Pooling type used
    pub pooling_type: Gemma3PoolingType,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
    /// L2 norm of the embedding (before normalization)
    pub original_norm: f32,
}

impl Gemma3EmbeddingResult3D {
    /// Create new embedding result
    pub fn new(embedding: Vec<f32>, pooling_type: Gemma3PoolingType, original_norm: f32) -> Self {
        let dimension = embedding.len();
        Self {
            embedding,
            dimension,
            pooling_type,
            spatial: SpatialTensorMetadata::new(dimension as u32, 1, 1),
            original_norm,
        }
    }
    
    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }
        
        let dot_product: f32 = self.embedding.iter()
            .zip(other.embedding.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        
        // Both are L2 normalized, so norms are 1
        dot_product
    }
}

/// 3D-aware Gemma3 embedding model
pub struct Gemma3EmbedModel3D {
    /// Text model (Gemma3 base)
    pub text_model: Gemma3TextModel3D,
    /// Dense projection layers
    pub dense_layers: Vec<Gemma3DenseLayer3D>,
    /// Pooling type
    pub pooling_type: Gemma3PoolingType,
    /// Configuration
    pub config: Gemma3EmbedConfig3D,
    /// Tokenizer
    pub tokenizer: Option<NeuralSentencePieceTokenizer>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

/// Placeholder for Gemma3 Text Model (simplified version)
#[derive(Debug, Clone)]
pub struct Gemma3TextModel3D {
    pub token_embedding: Vec<f32>,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl Gemma3TextModel3D {
    /// Create new text model
    pub fn new(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            token_embedding: vec![0.0; vocab_size * hidden_size],
            hidden_size,
            vocab_size,
        }
    }
    
    /// Forward pass (simplified)
    pub fn forward(&self, batch: &NeuralBatch) -> Vec<f32> {
        let mut hidden_states = Vec::new();
        
        for input in &batch.inputs {
            let token_id = input.token_id.max(0) as usize;
            let start = token_id * self.hidden_size;
            let end = (token_id + 1) * self.hidden_size;
            
            if end <= self.token_embedding.len() {
                hidden_states.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                hidden_states.extend(vec![0.0; self.hidden_size]);
            }
        }
        
        // Scale by sqrt(hidden_size) like Gemma
        let scale = (self.hidden_size as f32).sqrt();
        for h in &mut hidden_states {
            *h *= scale;
        }
        
        hidden_states
    }
}

impl Gemma3EmbedModel3D {
    /// Create new embedding model
    pub fn new(config: Gemma3EmbedConfig3D) -> Result<Self, Gemma3EmbedError> {
        config.validate()?;
        
        let text_model = Gemma3TextModel3D::new(config.hidden_size, 256128);
        
        // Create dense layers
        let dense0 = Gemma3DenseLayer3D::new(config.hidden_size, config.dense0_out);
        let dense1 = Gemma3DenseLayer3D::new(config.dense0_out, config.dense1_out);
        
        Ok(Self {
            text_model,
            dense_layers: vec![dense0, dense1],
            pooling_type: config.pooling_type,
            config,
            tokenizer: None,
            spatial: SpatialTensorMetadata::new(config.dense1_out as u32, 1, 1),
        })
    }
    
    /// Forward pass to generate embeddings
    pub fn forward(&self, batch: &NeuralBatch) -> Result<Gemma3EmbeddingResult3D, Gemma3EmbedError> {
        // Get hidden states from text model
        let hidden_states = self.text_model.forward(batch);
        let seq_len = batch.inputs.len();
        
        // Apply pooling
        let mut pooled = self.pooling_type.apply(&hidden_states, seq_len, self.config.hidden_size);
        
        // Pass through dense layers
        for dense in &self.dense_layers {
            pooled = dense.forward(&pooled);
        }
        
        // Compute original norm before L2 normalization
        let original_norm: f32 = pooled.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        // Apply L2 normalization
        let l2_norm = (pooled.iter().map(|&x| x * x).sum::<f32>() + self.config.l2_eps).sqrt();
        for x in &mut pooled {
            *x /= l2_norm;
        }
        
        Ok(Gemma3EmbeddingResult3D::new(pooled, self.pooling_type, original_norm))
    }
    
    /// Generate embeddings for a single text
    pub fn embed_text(&self, token_ids: &[i32]) -> Result<Gemma3EmbeddingResult3D, Gemma3EmbedError> {
        let inputs: Vec<NeuralInput> = token_ids.iter().enumerate().map(|(i, &id)| {
            NeuralInput::token(id, i as u32, 0)
        }).collect();
        
        let batch = NeuralBatch::new(inputs);
        self.forward(&batch)
    }
    
    /// Batch embedding generation
    pub fn embed_batch(&self, token_id_batches: &[Vec<i32>]) -> Result<Vec<Gemma3EmbeddingResult3D>, Gemma3EmbedError> {
        let mut results = Vec::new();
        
        for token_ids in token_id_batches {
            let result = self.embed_text(token_ids)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Get model info
    pub fn model_info(&self) -> Gemma3EmbedModelInfo {
        Gemma3EmbedModelInfo {
            name: "Gemma3-Embed-3D".to_string(),
            hidden_size: self.config.hidden_size,
            embedding_dim: self.config.dense1_out,
            pooling_type: self.pooling_type,
            num_dense_layers: self.dense_layers.len(),
            total_params: self.estimate_parameters(),
        }
    }
    
    /// Estimate total parameters
    fn estimate_parameters(&self) -> usize {
        // Text model embeddings
        let text_params = self.text_model.vocab_size * self.text_model.hidden_size;
        
        // Dense layers
        let dense_params: usize = self.dense_layers.iter()
            .map(|d| d.in_features * d.out_features + d.out_features)
            .sum();
        
        text_params + dense_params
    }
}

/// Gemma3 embedding model information
#[derive(Debug, Clone)]
pub struct Gemma3EmbedModelInfo {
    pub name: String,
    pub hidden_size: usize,
    pub embedding_dim: usize,
    pub pooling_type: Gemma3PoolingType,
    pub num_dense_layers: usize,
    pub total_params: usize,
}

/// Utility functions
pub mod gemma3_embed_utils {
    use super::*;
    
    /// Compute L2 distance between two embeddings
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Compute dot product between two embeddings
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }
    
    /// Normalize embedding to unit L2 norm
    pub fn normalize_l2(embedding: &mut [f32], eps: f32) {
        let norm = (embedding.iter().map(|&x| x * x).sum::<f32>() + eps).sqrt();
        for x in embedding {
            *x /= norm;
        }
    }
    
    /// Estimate memory for embedding model
    pub fn estimate_memory(config: &Gemma3EmbedConfig3D, vocab_size: usize) -> u64 {
        // Text embeddings
        let text_embed = (vocab_size * config.hidden_size * 4) as u64;
        
        // Dense layers
        let dense0 = (config.hidden_size * config.dense0_out * 4) as u64;
        let dense1 = (config.dense0_out * config.dense1_out * 4) as u64;
        let dense_bias = ((config.dense0_out + config.dense1_out) * 4) as u64;
        
        text_embed + dense0 + dense1 + dense_bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pooling_types() {
        let hidden_size = 10;
        let seq_len = 5;
        let data: Vec<f32> = (0..hidden_size * seq_len).map(|i| i as f32).collect();
        
        // Test CLS pooling
        let cls_result = Gemma3PoolingType::Cls.apply(&data, seq_len, hidden_size);
        assert_eq!(cls_result.len(), hidden_size);
        assert_eq!(cls_result[0], 0.0);
        
        // Test Mean pooling
        let mean_result = Gemma3PoolingType::Mean.apply(&data, seq_len, hidden_size);
        assert_eq!(mean_result.len(), hidden_size);
        
        // Test Last pooling
        let last_result = Gemma3PoolingType::Last.apply(&data, seq_len, hidden_size);
        assert_eq!(last_result.len(), hidden_size);
        assert_eq!(last_result[0], 40.0); // First element of last token
    }
    
    #[test]
    fn test_dense_layer() {
        let dense = Gemma3DenseLayer3D::new(10, 20);
        let input = vec![1.0; 10];
        let output = dense.forward(&input);
        
        assert_eq!(output.len(), 20);
    }
    
    #[test]
    fn test_embedding_result() {
        let embedding = vec![0.5, 0.5, 0.5, 0.5];
        let result = Gemma3EmbeddingResult3D::new(embedding, Gemma3PoolingType::Mean, 1.0);
        
        assert_eq!(result.dimension, 4);
        assert_eq!(result.pooling_type, Gemma3PoolingType::Mean);
        
        // Test cosine similarity with itself (should be ~1.0 for normalized)
        let normalized = vec![0.5; 4];
        let result2 = Gemma3EmbeddingResult3D::new(normalized, Gemma3PoolingType::Mean, 1.0);
        let sim = result2.cosine_similarity(&result2);
        assert!(sim > 0.99);
    }
    
    #[test]
    fn test_model_creation() {
        let config = Gemma3EmbedConfig3D::default();
        let model = Gemma3EmbedModel3D::new(config);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert_eq!(info.name, "Gemma3-Embed-3D");
        assert_eq!(info.embedding_dim, 1024);
    }
    
    #[test]
    fn test_forward_pass() {
        let config = Gemma3EmbedConfig3D::default();
        let model = Gemma3EmbedModel3D::new(config).unwrap();
        
        let token_ids = vec![1, 2, 3, 4, 5];
        let result = model.embed_text(&token_ids);
        
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.dimension, config.dense1_out);
        
        // Check L2 norm is approximately 1 (normalized)
        let l2_norm: f32 = embedding.embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(l2_norm > 0.99 && l2_norm < 1.01);
    }
    
    #[test]
    fn test_utils() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let dist = gemma3_embed_utils::l2_distance(&a, &b);
        assert!((dist - 1.414).abs() < 0.01);
        
        let dot = gemma3_embed_utils::dot_product(&a, &b);
        assert_eq!(dot, 0.0);
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = Gemma3EmbedConfig3D::default();
        let mem = gemma3_embed_utils::estimate_memory(&config, 256128);
        
        assert!(mem > 0);
        assert!(mem > 2_000_000_000); // > 2GB for embeddings
    }
}
