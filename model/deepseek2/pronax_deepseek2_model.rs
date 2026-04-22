use std::sync::Arc;

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput};
use crate::tokenizer::pronax_bpe_tokenizer::NeuralBpeTokenizer;

/// DeepSeek2/3 model errors
#[derive(Debug, Clone)]
pub enum DeepSeek2Error {
    InvalidConfiguration(String),
    AttentionError(String),
    MoeError(String),
    ForwardPassError(String),
    CacheError(String),
    UnsupportedArchitecture(String),
}

impl std::fmt::Display for DeepSeek2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "Invalid config: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::MoeError(s) => write!(f, "MoE error: {}", s),
            Self::ForwardPassError(s) => write!(f, "Forward pass error: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
            Self::UnsupportedArchitecture(s) => write!(f, "Unsupported: {}", s),
        }
    }
}

impl std::error::Error for DeepSeek2Error {}

/// 3D-aware DeepSeek2/3 configuration
/// Supports both MLA (v3) and standard attention (v3.1) modes
#[derive(Debug, Clone, Copy)]
pub struct DeepSeek2Config {
    /// Model dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA)
    pub num_key_value_heads: usize,
    /// Number of layers
    pub num_hidden_layers: usize,
    /// Number of dense layers before MoE
    pub num_dense_layers: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE base frequency
    pub rope_theta: f32,
    /// RoPE scaling factor
    pub rope_scaling_factor: f32,
    /// Original context length for scaling
    pub original_context_length: usize,
    /// Maximum context length
    pub max_position_embeddings: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Use MLA (Multi-head Latent Attention)
    pub use_mla: bool,
    /// LoRA rank for Q projection
    pub q_lora_rank: usize,
    /// LoRA rank for KV compression
    pub kv_lora_rank: usize,
    /// Query/Key nope dimension
    pub qk_nope_head_dim: usize,
    /// Query/Key RoPE dimension
    pub qk_rope_head_dim: usize,
    /// Value head dimension
    pub v_head_dim: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to use per token
    pub num_experts_per_token: usize,
    /// Normalize top-K probabilities
    pub norm_topk_prob: bool,
    /// Scaling factor for routed experts
    pub routed_scaling_factor: f32,
    /// 3D spatial dimension for attention
    pub spatial_attention_dim: u16,
}

impl DeepSeek2Config {
    /// DeepSeek-V3 base configuration (~671B params, activated ~37B)
    pub fn deepseek_v3() -> Self {
        Self {
            hidden_size: 7168,
            num_attention_heads: 128,
            num_key_value_heads: 128,
            num_hidden_layers: 61,
            num_dense_layers: 3,
            intermediate_size: 18432,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            original_context_length: 4096,
            max_position_embeddings: 16384,
            vocab_size: 129280,
            use_mla: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_experts: 256,
            num_experts_per_token: 8,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            spatial_attention_dim: 256,
        }
    }
    
    /// DeepSeek-V2 configuration
    pub fn deepseek_v2() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 26,
            num_dense_layers: 1,
            intermediate_size: 11008,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            original_context_length: 4096,
            max_position_embeddings: 16384,
            vocab_size: 102400,
            use_mla: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_experts: 64,
            num_experts_per_token: 6,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            spatial_attention_dim: 128,
        }
    }
    
    /// Standard dense model configuration (no MoE)
    pub fn dense_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            num_dense_layers: 32,
            intermediate_size: 11008,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            original_context_length: 4096,
            max_position_embeddings: 8192,
            vocab_size: 32000,
            use_mla: false,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            num_experts: 1,
            num_experts_per_token: 1,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
            spatial_attention_dim: 64,
        }
    }
    
    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    
    /// Compute attention scale
    pub fn attention_scale(&self) -> f64 {
        let m_scale = 1.0 + (self.rope_scaling_factor.ln() * 0.1);
        let scale = m_scale * m_scale / (self.head_dim() as f64).sqrt();
        scale
    }
    
    /// 3D spatial metadata for attention
    pub fn spatial_metadata(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(
            self.hidden_size as u32,
            self.num_attention_heads as u32,
            self.spatial_attention_dim as u32,
        )
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), DeepSeek2Error> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(DeepSeek2Error::InvalidConfiguration(
                format!("hidden_size {} not divisible by num_heads {}",
                    self.hidden_size, self.num_attention_heads)
            ));
        }
        
        if self.use_mla && (self.q_lora_rank == 0 || self.kv_lora_rank == 0) {
            return Err(DeepSeek2Error::InvalidConfiguration(
                "MLA requires non-zero q_lora_rank and kv_lora_rank".to_string()
            ));
        }
        
        if self.num_experts > 1 && self.num_experts_per_token > self.num_experts {
            return Err(DeepSeek2Error::InvalidConfiguration(
                "num_experts_per_token cannot exceed num_experts".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Check if layer uses dense or sparse MLP
    pub fn is_layer_dense(&self, layer_idx: usize) -> bool {
        layer_idx < self.num_dense_layers
    }
}

impl Default for DeepSeek2Config {
    fn default() -> Self {
        Self::deepseek_v3()
    }
}

/// 3D-aware RoPE (Rotary Position Embedding) parameters
#[derive(Debug, Clone, Copy)]
pub struct RoPEParams3D {
    /// Base frequency
    pub base: f32,
    /// Scaling factor
    pub scaling_factor: f32,
    /// Original context length
    pub original_context_length: usize,
    /// Extrapolation factor
    pub extrapolation_factor: f32,
    /// Attention scale factor
    pub attention_factor: f32,
    /// 3D spatial frequency scaling
    pub spatial_freq_scale: f32,
}

impl Default for RoPEParams3D {
    fn default() -> Self {
        Self {
            base: 10000.0,
            scaling_factor: 1.0,
            original_context_length: 4096,
            extrapolation_factor: 1.0,
            attention_factor: 1.0,
            spatial_freq_scale: 1.0,
        }
    }
}

impl RoPEParams3D {
    /// Create from config
    pub fn from_config(config: &DeepSeek2Config) -> Self {
        let attention_factor = 1.0 / (1.0 + 0.1 * config.rope_scaling_factor.ln());
        
        Self {
            base: config.rope_theta,
            scaling_factor: config.rope_scaling_factor,
            original_context_length: config.original_context_length,
            extrapolation_factor: 1.0,
            attention_factor,
            spatial_freq_scale: 1.0,
        }
    }
    
    /// Compute rotary frequencies with 3D spatial awareness
    pub fn compute_frequencies(&self, dim: usize, positions: &[i32]) -> Vec<f32> {
        let mut freqs = Vec::with_capacity(positions.len() * dim / 2);
        
        for &pos in positions {
            for i in (0..dim).step_by(2) {
                let freq = self.base.powf(-(i as f32) / dim as f32);
                let scaled_freq = freq * self.scaling_factor * self.spatial_freq_scale;
                let angle = pos as f32 * scaled_freq;
                
                freqs.push(angle.cos());
                freqs.push(angle.sin());
            }
        }
        
        freqs
    }
}

/// Multi-head Latent Attention (MLA) with 3D spatial compression
#[derive(Debug, Clone)]
pub struct SpatialMLAttention {
    /// Query projection weights (or QA/QB for LoRA)
    pub q_weights: Vec<f32>,
    pub q_a_weights: Option<Vec<f32>>,  // LoRA down-projection
    pub q_b_weights: Option<Vec<f32>>,  // LoRA up-projection
    
    /// KV compression weights (KVA/KVB for MLA)
    pub kv_a_weights: Vec<f32>,  // Down-projection
    pub kv_b_weights: Vec<f32>,  // Up-projection
    
    /// Decoupled K/V projections for non-MLA mode
    pub k_b_weights: Option<Vec<f32>>,
    pub v_b_weights: Option<Vec<f32>>,
    
    /// Output projection
    pub output_weights: Vec<f32>,
    
    /// Normalization parameters
    pub q_norm_weights: Vec<f32>,
    pub kv_norm_weights: Vec<f32>,
    
    /// Dimensions
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    
    /// 3D spatial metadata
    pub spatial_position: ConversionCoordinate,
}

impl SpatialMLAttention {
    /// Create new MLA layer
    pub fn new(config: &DeepSeek2Config) -> Self {
        let use_lora = config.q_lora_rank > 0;
        
        Self {
            q_weights: if use_lora {
                Vec::new()
            } else {
                vec![0.0; config.hidden_size * config.hidden_size]
            },
            q_a_weights: if use_lora {
                Some(vec![0.0; config.hidden_size * config.q_lora_rank])
            } else {
                None
            },
            q_b_weights: if use_lora {
                Some(vec![0.0; config.q_lora_rank * config.num_attention_heads * config.head_dim()])
            } else {
                None
            },
            kv_a_weights: vec![0.0; config.hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)],
            kv_b_weights: vec![0.0; config.kv_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)],
            k_b_weights: if config.use_mla {
                Some(vec![0.0; config.num_attention_heads * config.qk_nope_head_dim * config.kv_lora_rank])
            } else {
                None
            },
            v_b_weights: if config.use_mla {
                Some(vec![0.0; config.num_attention_heads * config.v_head_dim * config.kv_lora_rank])
            } else {
                None
            },
            output_weights: vec![0.0; config.num_attention_heads * config.v_head_dim * config.hidden_size],
            q_norm_weights: vec![1.0; if use_lora { config.q_lora_rank } else { config.hidden_size }],
            kv_norm_weights: vec![1.0; config.kv_lora_rank],
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with 3D spatial RoPE
    pub fn forward(
        &self,
        hidden_states: &[f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        rope_params: &RoPEParams3D,
        use_mla: bool,
    ) -> Result<Vec<f32>, DeepSeek2Error> {
        // Compute query projection
        let query = self.compute_query(hidden_states)?;
        
        // Compute compressed KV
        let (compressed_k, k_rot) = self.compute_compressed_kv(hidden_states)?;
        
        // Apply RoPE to rotary components
        let rope_freqs = rope_params.compute_frequencies(self.qk_rope_head_dim, positions);
        
        // MLA vs standard attention path
        let output = if use_mla {
            self.forward_mla(query, compressed_k, k_rot, cache, &rope_freqs)?
        } else {
            self.forward_standard(query, compressed_k, k_rot, cache, &rope_freqs)?
        };
        
        Ok(output)
    }
    
    fn compute_query(&self, hidden_states: &[f32]) -> Result<Vec<f32>, DeepSeek2Error> {
        // Simplified matrix multiplication
        let mut query = vec![0.0; self.num_heads * self.head_dim()];
        
        if let (Some(q_a), Some(q_b)) = (&self.q_a_weights, &self.q_b_weights) {
            // LoRA path: hidden -> QA -> norm -> QB
            // Simplified: just return zeros for structure
            Ok(query)
        } else {
            // Direct projection
            Ok(query)
        }
    }
    
    fn compute_compressed_kv(&self, hidden_states: &[f32]) -> Result<(Vec<f32>, Vec<f32>), DeepSeek2Error> {
        // Returns compressed key and rotary component
        let compressed = vec![0.0; self.kv_lora_rank];
        let rotary = vec![0.0; self.qk_rope_head_dim];
        Ok((compressed, rotary))
    }
    
    fn forward_mla(
        &self,
        query: Vec<f32>,
        compressed_k: Vec<f32>,
        k_rot: Vec<f32>,
        _cache: &mut CausalKVCache,
        _rope_freqs: &[f32],
    ) -> Result<Vec<f32>, DeepSeek2Error> {
        // MLA forward pass with absorbed projections
        let output = vec![0.0; self.hidden_size];
        Ok(output)
    }
    
    fn forward_standard(
        &self,
        query: Vec<f32>,
        compressed_k: Vec<f32>,
        k_rot: Vec<f32>,
        _cache: &mut CausalKVCache,
        _rope_freqs: &[f32],
    ) -> Result<Vec<f32>, DeepSeek2Error> {
        // Standard GQA forward pass
        let output = vec![0.0; self.hidden_size];
        Ok(output)
    }
    
    fn head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// RMS Normalization with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct SpatialRMSNorm {
    /// Scale parameters
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Dimension
    pub dim: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl SpatialRMSNorm {
    /// Create new RMS norm
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
            dim,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Normalize input
    pub fn normalize(&self, input: &mut [f32]) {
        if input.len() != self.dim {
            return;
        }
        
        // Compute RMS
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / self.dim as f32 + self.eps).sqrt();
        
        // Apply scale
        for (x, &w) in input.iter_mut().zip(self.weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

/// Dense MLP block
#[derive(Debug, Clone)]
pub struct DenseMlpBlock {
    /// Gate projection
    pub gate_weights: Vec<f32>,
    /// Up projection
    pub up_weights: Vec<f32>,
    /// Down projection
    pub down_weights: Vec<f32>,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl DenseMlpBlock {
    /// Create new dense MLP
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_weights: vec![0.0; hidden_size * intermediate_size],
            up_weights: vec![0.0; hidden_size * intermediate_size],
            down_weights: vec![0.0; intermediate_size * hidden_size],
            intermediate_size,
            hidden_size,
        }
    }
    
    /// Forward pass (SiLU gate)
    pub fn forward(&self, hidden_states: &mut [f32]) -> Result<(), DeepSeek2Error> {
        // Simplified MLP forward
        Ok(())
    }
    
    /// SiLU activation: x * sigmoid(x)
    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}

/// Sparse MoE (Mixture of Experts) block with 3D routing
#[derive(Debug, Clone)]
pub struct SparseMoeBlock {
    /// Router gate
    pub router_weights: Vec<f32>,
    /// Expert gate projections [num_experts, hidden_size, intermediate_size]
    pub expert_gate_weights: Vec<f32>,
    /// Expert up projections
    pub expert_up_weights: Vec<f32>,
    /// Expert down projections
    pub expert_down_weights: Vec<f32>,
    /// Shared expert (dense)
    pub shared_expert: DenseMlpBlock,
    /// Expert bias for probability adjustment
    pub expert_bias: Option<Vec<f32>>,
    
    /// Configuration
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    
    /// 3D spatial routing metadata
    pub expert_spatial_positions: Vec<ConversionCoordinate>,
}

impl SparseMoeBlock {
    /// Create new sparse MoE block
    pub fn new(config: &DeepSeek2Config) -> Self {
        let num_experts = config.num_experts;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        // Create 3D spatial positions for each expert
        let expert_spatial_positions: Vec<ConversionCoordinate> = (0..num_experts)
            .map(|i| {
                let x = (i % 16) as u64;
                let y = ((i / 16) % 16) as u16;
                let z = (i / 256) as u8;
                ConversionCoordinate::new(x, y, z, 1.0)
            })
            .collect();
        
        Self {
            router_weights: vec![0.0; num_experts * hidden_size],
            expert_gate_weights: vec![0.0; num_experts * hidden_size * intermediate_size],
            expert_up_weights: vec![0.0; num_experts * hidden_size * intermediate_size],
            expert_down_weights: vec![0.0; num_experts * intermediate_size * hidden_size],
            shared_expert: DenseMlpBlock::new(hidden_size, intermediate_size),
            expert_bias: None,
            num_experts,
            num_experts_per_token: config.num_experts_per_token,
            hidden_size,
            intermediate_size,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor,
            expert_spatial_positions,
        }
    }
    
    /// Forward pass with top-K routing
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
    ) -> Result<(), DeepSeek2Error> {
        // Compute routing scores
        let scores = self.compute_router_scores(hidden_states)?;
        
        // Select top-K experts
        let topk_indices = self.select_topk(&scores)?;
        let topk_weights = self.compute_topk_weights(&scores, &topk_indices)?;
        
        // Compute expert outputs
        let expert_output = self.compute_experts(hidden_states, &topk_indices, &topk_weights)?;
        
        // Add shared expert output
        let mut shared_output = hidden_states.to_vec();
        self.shared_expert.forward(&mut shared_output)?;
        
        // Combine: routed + shared
        for (h, (e, s)) in hidden_states.iter_mut()
            .zip(expert_output.iter().zip(shared_output.iter()))
        {
            *h = e + s;
        }
        
        Ok(())
    }
    
    fn compute_router_scores(&self, hidden_states: &[f32]) -> Result<Vec<f32>, DeepSeek2Error> {
        // Simplified: return uniform scores
        Ok(vec![1.0 / self.num_experts as f32; self.num_experts])
    }
    
    fn select_topk(&self, scores: &[f32]) -> Result<Vec<usize>, DeepSeek2Error> {
        // Simplified top-K selection
        Ok((0..self.num_experts_per_token).collect())
    }
    
    fn compute_topk_weights(
        &self,
        scores: &[f32],
        indices: &[usize],
    ) -> Result<Vec<f32>, DeepSeek2Error> {
        let mut weights: Vec<f32> = indices.iter()
            .map(|&i| scores.get(i).copied().unwrap_or(0.0))
            .collect();
        
        if self.norm_topk_prob {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        
        // Apply scaling
        for w in &mut weights {
            *w *= self.routed_scaling_factor;
        }
        
        Ok(weights)
    }
    
    fn compute_experts(
        &self,
        hidden_states: &[f32],
        indices: &[usize],
        weights: &[f32],
    ) -> Result<Vec<f32>, DeepSeek2Error> {
        // Simplified expert computation
        let output = vec![0.0; hidden_states.len()];
        Ok(output)
    }
    
    /// Get 3D spatial position of an expert
    pub fn get_expert_spatial(&self, expert_idx: usize) -> Option<&ConversionCoordinate> {
        self.expert_spatial_positions.get(expert_idx)
    }
}

/// MLP trait for polymorphic layer handling
#[derive(Debug, Clone)]
pub enum MlpBlock {
    Dense(DenseMlpBlock),
    Sparse(SparseMoeBlock),
}

impl MlpBlock {
    /// Forward pass
    pub fn forward(&self, hidden_states: &mut [f32]) -> Result<(), DeepSeek2Error> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states),
            Self::Sparse(moe) => moe.forward(hidden_states),
        }
    }
    
    /// Check if sparse MoE
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }
}

/// Transformer layer with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct DeepSeek2Layer {
    /// Attention normalization
    pub attn_norm: SpatialRMSNorm,
    /// Multi-head latent attention
    pub attention: SpatialMLAttention,
    /// MLP normalization
    pub mlp_norm: SpatialRMSNorm,
    /// MLP block (dense or sparse)
    pub mlp: MlpBlock,
    /// Layer index
    pub layer_idx: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl DeepSeek2Layer {
    /// Create new transformer layer
    pub fn new(config: &DeepSeek2Config, layer_idx: usize) -> Self {
        let mlp = if config.is_layer_dense(layer_idx) {
            MlpBlock::Dense(DenseMlpBlock::new(config.hidden_size, config.intermediate_size))
        } else {
            MlpBlock::Sparse(SparseMoeBlock::new(config))
        };
        
        Self {
            attn_norm: SpatialRMSNorm::new(config.hidden_size, config.rms_norm_eps),
            attention: SpatialMLAttention::new(config),
            mlp_norm: SpatialRMSNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp,
            layer_idx,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 8) as u16,
                (layer_idx % 8) as u8,
                1.0,
            ),
        }
    }
    
    /// Forward pass with residual connections
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        rope_params: &RoPEParams3D,
        use_mla: bool,
    ) -> Result<(), DeepSeek2Error> {
        // Attention path with residual
        let residual = hidden_states.to_vec();
        self.attn_norm.normalize(hidden_states);
        
        let attn_output = self.attention.forward(
            hidden_states,
            positions,
            cache,
            rope_params,
            use_mla,
        )?;
        
        // Add residual
        for (h, (r, a)) in hidden_states.iter_mut()
            .zip(residual.iter().zip(attn_output.iter()))
        {
            *h = r + a;
        }
        
        // MLP path with residual
        let residual = hidden_states.to_vec();
        self.mlp_norm.normalize(hidden_states);
        self.mlp.forward(hidden_states)?;
        
        // Add residual
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        Ok(())
    }
}

/// Token embedding with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct SpatialTokenEmbedding {
    /// Embedding weights [vocab_size, hidden_size]
    pub weight: Vec<f32>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl SpatialTokenEmbedding {
    /// Create new token embedding
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight: vec![0.0; vocab_size * hidden_size],
            vocab_size,
            hidden_size,
            spatial: SpatialTensorMetadata::new(vocab_size as u32, hidden_size as u32, 1),
        }
    }
    
    /// Lookup embedding for token
    pub fn lookup(&self, token_id: i32) -> Option<&[f32]> {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return None;
        }
        let start = idx * self.hidden_size;
        Some(&self.weight[start..start + self.hidden_size])
    }
    
    /// Forward pass for batch
    pub fn forward_batch(&self, batch: &NeuralBatch) -> Vec<Vec<f32>> {
        batch.inputs.iter()
            .map(|input| {
                self.lookup(input.token_id)
                    .map(|e| e.to_vec())
                    .unwrap_or_else(|| vec![0.0; self.hidden_size])
            })
            .collect()
    }
}

/// Output layer (LM head)
#[derive(Debug, Clone)]
pub struct OutputLayer {
    /// Weight matrix [hidden_size, vocab_size]
    pub weight: Vec<f32>,
    /// Output normalization
    pub norm: SpatialRMSNorm,
    /// Hidden size
    pub hidden_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl OutputLayer {
    /// Create new output layer
    pub fn new(hidden_size: usize, vocab_size: usize, eps: f32) -> Self {
        Self {
            weight: vec![0.0; hidden_size * vocab_size],
            norm: SpatialRMSNorm::new(hidden_size, eps),
            hidden_size,
            vocab_size,
        }
    }
    
    /// Project hidden states to logits
    pub fn forward(&self, hidden_states: &mut [f32]) -> Vec<f32> {
        self.norm.normalize(hidden_states);
        // Simplified projection
        vec![0.0; self.vocab_size]
    }
}

/// Complete DeepSeek2/3 model with 3D spatial awareness
pub struct DeepSeek2Model {
    /// Token embeddings
    pub token_embedding: SpatialTokenEmbedding,
    /// Transformer layers
    pub layers: Vec<DeepSeek2Layer>,
    /// Output layer
    pub output: OutputLayer,
    /// Configuration
    pub config: DeepSeekConfig,
    /// KV cache
    pub cache: CausalKVCache,
    /// RoPE parameters
    pub rope_params: RoPEParams3D,
    /// Tokenizer
    pub tokenizer: Option<NeuralBpeTokenizer>,
}

/// DeepSeek configuration bundle
#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    pub model_config: DeepSeek2Config,
    pub rope_params: RoPEParams3D,
}

impl DeepSeek2Model {
    /// Create new model from configuration
    pub fn new(config: DeepSeek2Config) -> Result<Self, DeepSeek2Error> {
        config.validate()?;
        
        let rope_params = RoPEParams3D::from_config(&config);
        
        let token_embedding = SpatialTokenEmbedding::new(config.vocab_size, config.hidden_size);
        
        let layers: Vec<DeepSeek2Layer> = (0..config.num_hidden_layers)
            .map(|i| DeepSeek2Layer::new(&config, i))
            .collect();
        
        let output = OutputLayer::new(
            config.hidden_size,
            config.vocab_size,
            config.rms_norm_eps,
        );
        
        let cache = CausalKVCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.hidden_size / config.num_key_value_heads,
        );
        
        Ok(Self {
            token_embedding,
            layers,
            output,
            config: DeepSeekConfig {
                model_config: config,
                rope_params,
            },
            cache,
            rope_params,
            tokenizer: None,
        })
    }
    
    /// Set tokenizer
    pub fn with_tokenizer(mut self, tokenizer: NeuralBpeTokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }
    
    /// Forward pass for batch
    pub fn forward(&mut self, batch: &NeuralBatch) -> Result<Vec<Vec<f32>>, DeepSeek2Error> {
        // Get embeddings
        let mut hidden_states = self.token_embedding.forward_batch(batch);
        
        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (i, h) in hidden_states.iter_mut().enumerate() {
                let positions: Vec<i32> = batch.inputs.iter()
                    .map(|inp| inp.position_in_sequence as i32)
                    .collect();
                
                layer.forward(
                    h,
                    &positions,
                    &mut self.cache,
                    &self.rope_params,
                    self.config.model_config.use_mla,
                )?;
            }
        }
        
        // Output projection
        let logits: Vec<Vec<f32>> = hidden_states.iter_mut()
            .map(|h| self.output.forward(h))
            .collect();
        
        Ok(logits)
    }
    
    /// Get model info
    pub fn model_info(&self) -> ModelInfo {
        let config = &self.config.model_config;
        
        ModelInfo {
            name: "DeepSeek2/3".to_string(),
            total_params: self.estimate_parameters(),
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            vocab_size: config.vocab_size,
            use_mla: config.use_mla,
            use_moe: config.num_experts > 1,
            num_experts: config.num_experts,
        }
    }
    
    /// Estimate total parameters
    fn estimate_parameters(&self) -> usize {
        let config = &self.config.model_config;
        
        // Embeddings
        let embedding_params = config.vocab_size * config.hidden_size;
        
        // Per layer params
        let attention_params = if config.use_mla {
            // MLA: compressed attention
            let q_params = config.hidden_size * config.q_lora_rank.max(config.hidden_size);
            let kv_params = config.hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim);
            let output_params = config.num_attention_heads * config.v_head_dim * config.hidden_size;
            q_params + kv_params + output_params
        } else {
            // Standard GQA
            let q_params = config.hidden_size * config.hidden_size;
            let kv_params = 2 * config.hidden_size * config.num_key_value_heads * config.head_dim();
            let output_params = config.hidden_size * config.hidden_size;
            q_params + kv_params + output_params
        };
        
        let mlp_params = if config.num_experts > 1 {
            // MoE: sparse + shared
            let expert_params = config.num_experts * 3 * config.hidden_size * config.intermediate_size;
            let shared_params = 3 * config.hidden_size * config.intermediate_size;
            let router_params = config.hidden_size * config.num_experts;
            expert_params + shared_params + router_params
        } else {
            // Dense MLP
            3 * config.hidden_size * config.intermediate_size
        };
        
        let layer_params = attention_params + mlp_params + 2 * config.hidden_size; // norms
        
        // Total
        embedding_params + config.num_hidden_layers * layer_params + embedding_params // output
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub total_params: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub use_mla: bool,
    pub use_moe: bool,
    pub num_experts: usize,
}

/// Utility functions
pub mod deepseek_utils {
    use super::*;
    
    /// Compute model FLOPs for given sequence length
    pub fn estimate_flops(config: &DeepSeek2Config, seq_len: usize, batch_size: usize) -> u64 {
        let hidden = config.hidden_size as u64;
        let layers = config.num_hidden_layers as u64;
        let heads = config.num_attention_heads as u64;
        let seq = seq_len as u64;
        let batch = batch_size as u64;
        
        // Attention FLOPs
        let attn_flops = 2 * batch * seq * seq * heads * hidden * layers;
        
        // MLP FLOPs
        let mlp_flops = 2 * batch * seq * hidden * (config.intermediate_size as u64) * layers;
        
        attn_flops + mlp_flops
    }
    
    /// Check if model fits in memory
    pub fn fits_in_memory(config: &DeepSeek2Config, available_bytes: u64) -> bool {
        // Rough estimate: 4 bytes per parameter
        let params = config.vocab_size * config.hidden_size +
            config.num_hidden_layers * (
                config.hidden_size * config.hidden_size * 4 +
                config.hidden_size * config.intermediate_size * 3
            );
        let bytes_needed = (params as u64) * 4;
        bytes_needed <= available_bytes
    }
    
    /// Get tokenizer regex patterns for DeepSeek
    pub fn get_tokenizer_patterns(model_type: &str) -> Vec<&'static str> {
        match model_type {
            "deepseek-v3" => vec![
                r"\p{N}{1,3}",
                r"[一-龥぀-ゟ゠-ヿ]+",
                r"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = DeepSeek2Config::deepseek_v3();
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.num_experts, 256);
        assert!(config.use_mla);
        
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = DeepSeek2Config::deepseek_v2();
        config.hidden_size = 100;
        
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_rope_params() {
        let config = DeepSeek2Config::deepseek_v3();
        let rope = RoPEParams3D::from_config(&config);
        
        assert!(rope.attention_factor > 0.0);
        assert!(rope.attention_factor <= 1.0);
    }
    
    #[test]
    fn test_mla_attention() {
        let config = DeepSeek2Config::deepseek_v3();
        let attn = SpatialMLAttention::new(&config);
        
        assert!(attn.q_a_weights.is_some());
        assert!(attn.k_b_weights.is_some());
    }
    
    #[test]
    fn test_sparse_moe() {
        let config = DeepSeek2Config::deepseek_v3();
        let moe = SparseMoeBlock::new(&config);
        
        assert_eq!(moe.num_experts, 256);
        assert_eq!(moe.expert_spatial_positions.len(), 256);
    }
    
    #[test]
    fn test_layer_creation() {
        let config = DeepSeek2Config::deepseek_v3();
        let layer = DeepSeek2Layer::new(&config, 0);
        
        assert!(matches!(layer.mlp, MlpBlock::Dense(_)));
        
        let moe_layer = DeepSeek2Layer::new(&config, 10);
        assert!(matches!(moe_layer.mlp, MlpBlock::Sparse(_)));
    }
    
    #[test]
    fn test_model_creation() {
        let config = DeepSeek2Config::deepseek_v2();
        let model = DeepSeek2Model::new(config);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert!(info.use_mla);
        assert!(info.use_moe);
    }
    
    #[test]
    fn test_rms_norm() {
        let norm = SpatialRMSNorm::new(768, 1e-6);
        let mut data: Vec<f32> = (0..768).map(|i| i as f32).collect();
        
        norm.normalize(&mut data);
        
        // Check that values are normalized
        let rms = (data.iter().map(|&x| x * x).sum::<f32>() / 768.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_utils() {
        let config = DeepSeek2Config::deepseek_v3();
        let flops = deepseek_utils::estimate_flops(&config, 1024, 1);
        assert!(flops > 0);
        
        let patterns = deepseek_utils::get_tokenizer_patterns("deepseek-v3");
        assert!(!patterns.is_empty());
    }
}
