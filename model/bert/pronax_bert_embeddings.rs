use std::sync::Arc;

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput, NeuralMultimodalEmbedding};
use crate::model::pronax_model_trait::{NeuralModel, NeuralModelConfig};
use crate::tokenizer::pronax_vocabulary::{NeuralVocabEntry, NeuralVocabulary};

/// BERT embedding layer errors
#[derive(Debug, Clone)]
pub enum BertEmbeddingError {
    InvalidDimensions(String),
    EmbeddingNotFound(String),
    ConfigurationError(String),
    ForwardPassError(String),
}

impl std::fmt::Display for BertEmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::EmbeddingNotFound(s) => write!(f, "Embedding not found: {}", s),
            Self::ConfigurationError(s) => write!(f, "Configuration error: {}", s),
            Self::ForwardPassError(s) => write!(f, "Forward pass error: {}", s),
        }
    }
}

impl std::error::Error for BertEmbeddingError {}

/// 3D-aware BERT embedding configuration
#[derive(Debug, Clone, Copy)]
pub struct BertEmbeddingConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension (embedding size)
    pub hidden_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Number of token types (usually 2 for BERT)
    pub type_vocab_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of encoder layers
    pub num_hidden_layers: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,
    /// Pooling type: 0=CLS, 1=MEAN, 2=MAX
    pub pooling_type: u8,
    /// 3D spatial embedding dimension
    pub spatial_embedding_dim: u16,
}

impl BertEmbeddingConfig {
    /// Create default BERT-base configuration
    pub fn bert_base() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            normalize_embeddings: true,
            pooling_type: 0, // CLS token
            spatial_embedding_dim: 64,
        }
    }
    
    /// Create BERT-large configuration
    pub fn bert_large() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 1024,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            normalize_embeddings: true,
            pooling_type: 0,
            spatial_embedding_dim: 128,
        }
    }
    
    /// Create 3D spatial BERT configuration
    pub fn bert_3d() -> Self {
        let mut config = Self::bert_base();
        config.spatial_embedding_dim = 256;
        config.pooling_type = 1; // MEAN pooling for 3D
        config
    }
    
    /// Head dimension (hidden_size / num_heads)
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    /// Calculate 3D spatial parameters
    pub fn compute_spatial_dims(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(
            self.hidden_size as u32,
            self.max_position_embeddings as u32,
            self.spatial_embedding_dim as u32,
        )
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), BertEmbeddingError> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(BertEmbeddingError::ConfigurationError(
                format!("hidden_size {} must be divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads)
            ));
        }
        
        if self.vocab_size == 0 {
            return Err(BertEmbeddingError::ConfigurationError(
                "vocab_size must be > 0".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for BertEmbeddingConfig {
    fn default() -> Self {
        Self::bert_base()
    }
}

/// 3D-aware token embedding layer
#[derive(Debug, Clone)]
pub struct SpatialTokenEmbedding {
    /// Embedding weights [vocab_size, hidden_size]
    pub weights: Vec<f32>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
    /// Position in 3D embedding space
    pub spatial_position: ConversionCoordinate,
}

impl SpatialTokenEmbedding {
    /// Create new token embedding layer
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weights: vec![0.0; vocab_size * hidden_dim],
            vocab_size,
            hidden_dim,
            spatial: SpatialTensorMetadata::new(vocab_size as u32, hidden_dim as u32, 1),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Look up embedding for token ID (zero-copy view)
    pub fn lookup(&self, token_id: i32) -> Option<&[f32]> {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return None;
        }
        let start = idx * self.hidden_dim;
        Some(&self.weights[start..start + self.hidden_dim])
    }
    
    /// Look up multiple tokens (batch)
    pub fn lookup_batch(&self, token_ids: &[i32]) -> Vec<Option<&[f32]>> {
        token_ids.iter().map(|&id| self.lookup(id)).collect()
    }
    
    /// Get embedding with 3D spatial coordinate
    pub fn lookup_with_spatial(&self, token_id: i32, seq_pos: usize) -> Option<SpatialEmbedding> {
        let embedding = self.lookup(token_id)?;
        let spatial = ConversionCoordinate::new(
            token_id as u64,
            seq_pos as u16,
            0,
            1.0,
        );
        
        Some(SpatialEmbedding {
            data: embedding.to_vec(),
            position: spatial,
            token_id,
        })
    }
    
    /// Total parameter count
    pub fn parameter_count(&self) -> usize {
        self.vocab_size * self.hidden_dim
    }
}

/// 3D-aware position embedding layer
#[derive(Debug, Clone)]
pub struct SpatialPositionEmbedding {
    /// Embedding weights [max_positions, hidden_size]
    pub weights: Vec<f32>,
    /// Maximum sequence length
    pub max_positions: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl SpatialPositionEmbedding {
    /// Create new position embedding layer
    pub fn new(max_positions: usize, hidden_dim: usize) -> Self {
        Self {
            weights: vec![0.0; max_positions * hidden_dim],
            max_positions,
            hidden_dim,
            spatial: SpatialTensorMetadata::new(max_positions as u32, hidden_dim as u32, 1),
        }
    }
    
    /// Get position embedding
    pub fn get_position(&self, position: usize) -> Option<&[f32]> {
        if position >= self.max_positions {
            return None;
        }
        let start = position * self.hidden_dim;
        Some(&self.weights[start..start + self.hidden_dim])
    }
    
    /// Get positions for a range
    pub fn get_positions(&self, positions: &[i32]) -> Vec<Option<&[f32]>> {
        positions.iter()
            .map(|&p| self.get_position(p as usize))
            .collect()
    }
    
    /// Get position with 3D spatial coordinate
    pub fn get_with_spatial(&self, position: usize, layer_idx: u8) -> Option<SpatialEmbedding> {
        let embedding = self.get_position(position)?;
        let spatial = ConversionCoordinate::new(
            position as u64,
            layer_idx as u16,
            layer_idx as u8,
            1.0 - (position as f32 / self.max_positions as f32),
        );
        
        Some(SpatialEmbedding {
            data: embedding.to_vec(),
            position: spatial,
            token_id: -(position as i32), // Negative indicates position embedding
        })
    }
}

/// 3D-aware token type embedding layer (segment embeddings)
#[derive(Debug, Clone)]
pub struct SpatialTypeEmbedding {
    /// Embedding weights [type_vocab_size, hidden_size]
    pub weights: Vec<f32>,
    /// Type vocabulary size (usually 2)
    pub type_vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl SpatialTypeEmbedding {
    /// Create new type embedding layer
    pub fn new(type_vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            weights: vec![0.0; type_vocab_size * hidden_dim],
            type_vocab_size,
            hidden_dim,
        }
    }
    
    /// Get type embedding
    pub fn get_type(&self, token_type: usize) -> Option<&[f32]> {
        if token_type >= self.type_vocab_size {
            return None;
        }
        let start = token_type * self.hidden_dim;
        Some(&self.weights[start..start + self.hidden_dim])
    }
}

/// Spatial embedding with 3D coordinate
#[derive(Debug, Clone)]
pub struct SpatialEmbedding {
    /// Embedding vector data
    pub data: Vec<f32>,
    /// 3D spatial position
    pub position: ConversionCoordinate,
    /// Token ID (negative for position embeddings)
    pub token_id: i32,
}

impl SpatialEmbedding {
    /// L2 normalize embedding
    pub fn normalize(&mut self, eps: f32) {
        let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > eps {
            let scale = 1.0 / norm;
            for x in &mut self.data {
                *x *= scale;
            }
        }
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

/// Combined BERT embedding layer with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct BertEmbeddingLayer {
    /// Token embeddings
    pub token_embeddings: SpatialTokenEmbedding,
    /// Position embeddings
    pub position_embeddings: SpatialPositionEmbedding,
    /// Type embeddings
    pub type_embeddings: SpatialTypeEmbedding,
    /// Layer normalization
    pub layer_norm: LayerNorm3D,
    /// Dropout probability
    pub dropout_prob: f32,
    /// 3D spatial config
    pub spatial_config: SpatialEmbeddingConfig,
}

/// 3D spatial embedding configuration
#[derive(Debug, Clone, Copy)]
pub struct SpatialEmbeddingConfig {
    /// Enable 3D spatial features
    pub enable_3d: bool,
    /// Spatial dimension depth
    pub depth: u32,
    /// Guidance scale for spatial attention
    pub guidance_scale: f32,
}

impl Default for SpatialEmbeddingConfig {
    fn default() -> Self {
        Self {
            enable_3d: true,
            depth: 64,
            guidance_scale: 1.0,
        }
    }
}

impl BertEmbeddingLayer {
    /// Create new BERT embedding layer
    pub fn new(config: &BertEmbeddingConfig) -> Self {
        Self {
            token_embeddings: SpatialTokenEmbedding::new(config.vocab_size, config.hidden_size),
            position_embeddings: SpatialPositionEmbedding::new(
                config.max_position_embeddings,
                config.hidden_size,
            ),
            type_embeddings: SpatialTypeEmbedding::new(config.type_vocab_size, config.hidden_size),
            layer_norm: LayerNorm3D::new(config.hidden_size, config.layer_norm_eps),
            dropout_prob: config.hidden_dropout_prob,
            spatial_config: SpatialEmbeddingConfig::default(),
        }
    }
    
    /// Forward pass: combine embeddings
    pub fn forward(&self, input: &NeuralInput) -> Result<Vec<f32>, BertEmbeddingError> {
        // Get token embedding
        let token_emb = self.token_embeddings.lookup(input.token_id)
            .ok_or_else(|| BertEmbeddingError::EmbeddingNotFound(
                format!("Token {} not in vocabulary", input.token_id)
            ))?;
        
        // Get position embedding
        let pos_emb = self.position_embeddings.get_position(input.position_in_sequence)
            .ok_or_else(|| BertEmbeddingError::EmbeddingNotFound(
                format!("Position {} exceeds max", input.position_in_sequence)
            ))?;
        
        // Get type embedding (default to type 0)
        let type_emb = self.type_embeddings.get_type(0).unwrap_or(&[]);
        
        // Combine embeddings: token + position + type
        let mut combined: Vec<f32> = token_emb.iter()
            .zip(pos_emb.iter())
            .map(|(t, p)| t + p)
            .collect();
        
        if !type_emb.is_empty() {
            for (c, t) in combined.iter_mut().zip(type_emb.iter()) {
                *c += t;
            }
        }
        
        // Apply layer normalization
        self.layer_norm.normalize(&mut combined);
        
        Ok(combined)
    }
    
    /// Forward pass for batch
    pub fn forward_batch(&self, batch: &NeuralBatch) -> Result<Vec<Vec<f32>>, BertEmbeddingError> {
        batch.inputs.iter()
            .map(|input| self.forward(input))
            .collect()
    }
    
    /// Forward with 3D spatial metadata
    pub fn forward_spatial(&self, input: &NeuralInput) -> Result<SpatialEmbedding, BertEmbeddingError> {
        let data = self.forward(input)?;
        let position = ConversionCoordinate::new(
            input.token_id as u64,
            input.position_in_sequence as u16,
            0,
            self.spatial_config.guidance_scale,
        );
        
        Ok(SpatialEmbedding {
            data,
            position,
            token_id: input.token_id,
        })
    }
    
    /// Total parameter count
    pub fn parameter_count(&self) -> usize {
        self.token_embeddings.parameter_count() +
            self.position_embeddings.max_positions * self.position_embeddings.hidden_dim +
            self.type_embeddings.type_vocab_size * self.type_embeddings.hidden_dim +
            self.layer_norm.parameter_count()
    }
}

/// 3D-aware layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm3D {
    /// Gamma (scale) parameters
    pub gamma: Vec<f32>,
    /// Beta (shift) parameters
    pub beta: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl LayerNorm3D {
    /// Create new layer norm
    pub fn new(hidden_dim: usize, eps: f32) -> Self {
        Self {
            gamma: vec![1.0; hidden_dim],
            beta: vec![0.0; hidden_dim],
            eps,
            hidden_dim,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Normalize input in-place
    pub fn normalize(&self, input: &mut [f32]) {
        if input.len() != self.hidden_dim {
            return;
        }
        
        // Compute mean
        let mean: f32 = input.iter().sum::<f32>() / self.hidden_dim as f32;
        
        // Compute variance
        let variance: f32 = input.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.hidden_dim as f32;
        
        // Normalize and apply scale/shift
        let std = (variance + self.eps).sqrt();
        for i in 0..self.hidden_dim {
            input[i] = ((input[i] - mean) / std) * self.gamma[i] + self.beta[i];
        }
    }
    
    /// Parameter count
    pub fn parameter_count(&self) -> usize {
        self.hidden_dim * 2 // gamma + beta
    }
}

/// BERT encoder layer with 3D spatial attention
#[derive(Debug, Clone)]
pub struct BertEncoderLayer3D {
    /// Attention mechanism
    pub attention: SpatialSelfAttention,
    /// Attention output normalization
    pub attention_norm: LayerNorm3D,
    /// MLP block
    pub mlp: MlpBlock3D,
    /// MLP output normalization
    pub mlp_norm: LayerNorm3D,
    /// Layer index for 3D positioning
    pub layer_idx: u8,
}

impl BertEncoderLayer3D {
    /// Forward pass with residual connections
    pub fn forward(&self, hidden_states: &mut [f32]) -> Result<(), BertEmbeddingError> {
        // Store residual
        let residual = hidden_states.to_vec();
        
        // Self-attention
        self.attention.forward(hidden_states)?;
        
        // Add residual and normalize
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        self.attention_norm.normalize(hidden_states);
        
        // Store residual for MLP
        let residual = hidden_states.to_vec();
        
        // MLP
        self.mlp.forward(hidden_states)?;
        
        // Add residual and normalize
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        self.mlp_norm.normalize(hidden_states);
        
        Ok(())
    }
}

/// 3D spatial self-attention mechanism
#[derive(Debug, Clone)]
pub struct SpatialSelfAttention {
    /// Query projection weights
    pub query_weights: Vec<f32>,
    /// Key projection weights
    pub key_weights: Vec<f32>,
    /// Value projection weights
    pub value_weights: Vec<f32>,
    /// Output projection weights
    pub output_weights: Vec<f32>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl SpatialSelfAttention {
    /// Forward pass (simplified)
    pub fn forward(&self, hidden_states: &mut [f32]) -> Result<(), BertEmbeddingError> {
        // Simplified attention forward - full implementation would need full tensor ops
        // This is a placeholder for the architecture
        Ok(())
    }
}

/// MLP block with 3D spatial features
#[derive(Debug, Clone)]
pub struct MlpBlock3D {
    /// Up-projection weights
    pub up_weights: Vec<f32>,
    /// Down-projection weights
    pub down_weights: Vec<f32>,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl MlpBlock3D {
    /// Forward pass
    pub fn forward(&self, hidden_states: &mut [f32]) -> Result<(), BertEmbeddingError> {
        // Simplified MLP forward
        // Full implementation would include GELU activation
        Ok(())
    }
}

/// Pooling strategies for BERT outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy3D {
    /// CLS token embedding
    CLS,
    /// Mean pooling of all tokens
    Mean,
    /// Max pooling of all tokens
    Max,
    /// 3D spatial pooling (custom)
    Spatial3D,
}

impl PoolingStrategy3D {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CLS => "cls",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Spatial3D => "spatial3d",
        }
    }
}

/// BERT embedding model with full 3D support
pub struct BertEmbeddingModel {
    /// Embedding layer
    pub embeddings: BertEmbeddingLayer,
    /// Encoder layers
    pub encoder_layers: Vec<BertEncoderLayer3D>,
    /// Configuration
    pub config: BertEmbeddingConfig,
    /// Pooling strategy
    pub pooling: PoolingStrategy3D,
    /// Vocabulary
    pub vocabulary: Option<NeuralVocabulary>,
}

impl BertEmbeddingModel {
    /// Create new BERT embedding model
    pub fn new(config: BertEmbeddingConfig) -> Result<Self, BertEmbeddingError> {
        config.validate()?;
        
        let embeddings = BertEmbeddingLayer::new(&config);
        
        // Create encoder layers
        let encoder_layers: Vec<BertEncoderLayer3D> = (0..config.num_hidden_layers)
            .map(|i| BertEncoderLayer3D {
                attention: SpatialSelfAttention {
                    query_weights: vec![0.0; config.hidden_size * config.hidden_size],
                    key_weights: vec![0.0; config.hidden_size * config.hidden_size],
                    value_weights: vec![0.0; config.hidden_size * config.hidden_size],
                    output_weights: vec![0.0; config.hidden_size * config.hidden_size],
                    num_heads: config.num_attention_heads,
                    head_dim: config.head_dim(),
                    hidden_size: config.hidden_size,
                },
                attention_norm: LayerNorm3D::new(config.hidden_size, config.layer_norm_eps),
                mlp: MlpBlock3D {
                    up_weights: vec![0.0; config.hidden_size * config.intermediate_size],
                    down_weights: vec![0.0; config.intermediate_size * config.hidden_size],
                    intermediate_size: config.intermediate_size,
                    hidden_size: config.hidden_size,
                },
                mlp_norm: LayerNorm3D::new(config.hidden_size, config.layer_norm_eps),
                layer_idx: i as u8,
            })
            .collect();
        
        let pooling = match config.pooling_type {
            0 => PoolingStrategy3D::CLS,
            1 => PoolingStrategy3D::Mean,
            2 => PoolingStrategy3D::Max,
            _ => PoolingStrategy3D::CLS,
        };
        
        Ok(Self {
            embeddings,
            encoder_layers,
            config,
            pooling,
            vocabulary: None,
        })
    }
    
    /// Set vocabulary
    pub fn with_vocabulary(mut self, vocab: NeuralVocabulary) -> Self {
        self.vocabulary = Some(vocab);
        self
    }
    
    /// Get embeddings for input batch
    pub fn embed(&self, batch: &NeuralBatch) -> Result<Vec<Vec<f32>>, BertEmbeddingError> {
        self.embeddings.forward_batch(batch)
    }
    
    /// Get embeddings with 3D spatial metadata
    pub fn embed_spatial(&self, batch: &NeuralBatch) -> Result<Vec<SpatialEmbedding>, BertEmbeddingError> {
        batch.inputs.iter()
            .map(|input| self.embeddings.forward_spatial(input))
            .collect()
    }
    
    /// Pool embeddings according to strategy
    pub fn pool(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        match self.pooling {
            PoolingStrategy3D::CLS => {
                // Return first token (CLS) embedding
                embeddings.first().cloned().unwrap_or_default()
            }
            PoolingStrategy3D::Mean => {
                // Average all token embeddings
                if embeddings.is_empty() {
                    return vec![];
                }
                let dim = embeddings[0].len();
                let mut result = vec![0.0; dim];
                for emb in embeddings {
                    for (r, e) in result.iter_mut().zip(emb.iter()) {
                        *r += e;
                    }
                }
                let n = embeddings.len() as f32;
                for r in &mut result {
                    *r /= n;
                }
                result
            }
            PoolingStrategy3D::Max => {
                // Max pool all token embeddings
                if embeddings.is_empty() {
                    return vec![];
                }
                let dim = embeddings[0].len();
                let mut result = embeddings[0].clone();
                for emb in embeddings.iter().skip(1) {
                    for (r, e) in result.iter_mut().zip(emb.iter()) {
                        *r = r.max(*e);
                    }
                }
                result
            }
            PoolingStrategy3D::Spatial3D => {
                // Custom 3D spatial pooling
                self.pool_spatial_3d(embeddings)
            }
        }
    }
    
    /// 3D spatial pooling implementation
    fn pool_spatial_3d(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        // Weighted pooling based on 3D position
        if embeddings.is_empty() {
            return vec![];
        }
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        let mut total_weight = 0.0f32;
        
        for (i, emb) in embeddings.iter().enumerate() {
            // Weight decreases with position (early tokens more important)
            let weight = 1.0 / (1.0 + i as f32 * 0.1);
            for (r, e) in result.iter_mut().zip(emb.iter()) {
                *r += e * weight;
            }
            total_weight += weight;
        }
        
        for r in &mut result {
            *r /= total_weight;
        }
        result
    }
    
    /// Normalize embeddings
    pub fn normalize_embeddings(&self, embeddings: &mut [f32]) {
        let eps = self.config.layer_norm_eps;
        let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > eps {
            let scale = 1.0 / norm;
            for x in embeddings {
                *x *= scale;
            }
        }
    }
    
    /// Total parameter count
    pub fn parameter_count(&self) -> usize {
        let embedding_params = self.embeddings.parameter_count();
        let encoder_params: usize = self.encoder_layers.iter()
            .map(|l| {
                let attn_params = l.attention.query_weights.len() +
                    l.attention.key_weights.len() +
                    l.attention.value_weights.len() +
                    l.attention.output_weights.len();
                let mlp_params = l.mlp.up_weights.len() + l.mlp.down_weights.len();
                let norm_params = l.attention_norm.parameter_count() + l.mlp_norm.parameter_count();
                attn_params + mlp_params + norm_params
            })
            .sum();
        embedding_params + encoder_params
    }
}

/// Utility functions for BERT embeddings
pub mod bert_utils {
    use super::*;
    
    /// Compute cosine similarity between embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
    
    /// Compute Euclidean distance between embeddings
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Batch normalize embeddings
    pub fn batch_normalize(embeddings: &mut [Vec<f32>], eps: f32) {
        for emb in embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > eps {
                let scale = 1.0 / norm;
                for x in emb {
                    *x *= scale;
                }
            }
        }
    }
    
    /// Create batch from text tokens
    pub fn create_token_batch(token_ids: &[i32], sequence_id: u32) -> NeuralBatch {
        let mut batch = NeuralBatch::with_capacity(token_ids.len());
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let input = NeuralInput::token(token_id, pos, sequence_id);
            batch.add_input(input).ok();
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bert_config() {
        let config = BertEmbeddingConfig::bert_base();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.head_dim(), 64); // 768 / 12
        
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_bert_config_validation() {
        let mut config = BertEmbeddingConfig::bert_base();
        config.hidden_size = 100; // Not divisible by 12
        
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_token_embedding() {
        let emb = SpatialTokenEmbedding::new(1000, 768);
        assert_eq!(emb.parameter_count(), 1000 * 768);
        
        let lookup = emb.lookup(5);
        assert!(lookup.is_some());
        assert_eq!(lookup.unwrap().len(), 768);
    }
    
    #[test]
    fn test_position_embedding() {
        let emb = SpatialPositionEmbedding::new(512, 768);
        
        let pos = emb.get_position(100);
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().len(), 768);
        
        let out_of_range = emb.get_position(1000);
        assert!(out_of_range.is_none());
    }
    
    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm3D::new(768, 1e-12);
        let mut data: Vec<f32> = (0..768).map(|i| i as f32).collect();
        
        ln.normalize(&mut data);
        
        // After normalization, mean should be ~0, std should be ~1
        let mean: f32 = data.iter().sum::<f32>() / 768.0;
        assert!(mean.abs() < 0.1);
    }
    
    #[test]
    fn test_bert_embedding_layer() {
        let config = BertEmbeddingConfig::bert_base();
        let layer = BertEmbeddingLayer::new(&config);
        
        let input = NeuralInput::token(100, 5, 0);
        let result = layer.forward(&input);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 768);
    }
    
    #[test]
    fn test_pooling_strategies() {
        let config = BertEmbeddingConfig::bert_base();
        let model = BertEmbeddingModel::new(config).unwrap();
        
        // Create dummy embeddings
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32; 768])
            .collect();
        
        let pooled = model.pool(&embeddings);
        assert_eq!(pooled.len(), 768);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((bert_utils::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!(bert_utils::cosine_similarity(&a, &c).abs() < 0.001);
    }
    
    #[test]
    fn test_spatial_embedding() {
        let config = BertEmbeddingConfig::bert_3d();
        let layer = BertEmbeddingLayer::new(&config);
        
        let input = NeuralInput::token(42, 10, 0);
        let spatial = layer.forward_spatial(&input).unwrap();
        
        assert_eq!(spatial.data.len(), 768);
        assert_eq!(spatial.token_id, 42);
    }
    
    #[test]
    fn test_parameter_count() {
        let config = BertEmbeddingConfig::bert_base();
        let model = BertEmbeddingModel::new(config).unwrap();
        
        let params = model.parameter_count();
        // Rough check: BERT-base has ~110M parameters
        assert!(params > 100_000_000);
    }
}
