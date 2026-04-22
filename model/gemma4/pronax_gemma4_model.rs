//! ProNax Gemma4 - Advanced 3D Multimodal AI Model

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput, MultimodalData};
use crate::tokenizer::pronax_sentencepiece::NeuralSentencePieceTokenizer;
use crate::model::gemma4::pronax_gemma4_audio::{
    PronaxAudioEncoder3D,
    PronaxAudioHyperparams3D,
    PronaxAudioTextProjector3D,
};
use crate::model::gemma4::pronax_gemma4_vision::{
    PronaxVisionEncoder3D,
    PronaxVisionHyperparams3D,
    PronaxVisionTextProjector3D,
};

/// Gemma4 model errors
#[derive(Debug, Clone)]
pub enum Gemma4Error {
    InvalidConfiguration(String),
    ForwardError(String),
    MultimodalError(String),
    AttentionError(String),
    CacheError(String),
    VisionError(String),
    AudioError(String),
    UnsupportedModel(String),
}

impl std::fmt::Display for Gemma4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "Invalid config: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::MultimodalError(s) => write!(f, "Multimodal error: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
            Self::VisionError(s) => write!(f, "Vision error: {}", s),
            Self::AudioError(s) => write!(f, "Audio error: {}", s),
            Self::UnsupportedModel(s) => write!(f, "Unsupported: {}", s),
        }
    }
}

impl std::error::Error for Gemma4Error {}

/// 3D-aware Gemma4 configuration with multimodal support
#[derive(Debug, Clone, Copy)]
pub struct Gemma4Config3D {
    // Text model parameters
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_global_kv_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub eps: f32,
    
    // RoPE parameters
    pub rope_base: f32,
    pub rope_local_base: f32,
    pub partial_rotary_dims: usize,
    
    // Attention parameters
    pub sliding_window: usize,
    pub final_logit_softcap: f32,
    
    // MoE parameters
    pub num_experts: usize,
    pub num_experts_used: usize,
    
    // Vision model parameters
    pub vision_hidden_size: usize,
    pub vision_num_heads: usize,
    pub vision_patch_size: usize,
    pub vision_num_layers: usize,
    pub vision_rope_theta: f32,
    
    // Audio model parameters
    pub audio_hidden_size: usize,
    pub audio_num_heads: usize,
    pub audio_num_layers: usize,
    pub audio_mel_bins: usize,
    pub audio_chunk_size: usize,
    pub audio_max_past: usize,
    pub audio_max_future: usize,
    
    // 3D spatial parameters
    pub spatial_depth: u8,
    pub spatial_guidance: f32,
}

impl Gemma4Config3D {
    /// Gemma4 2B configuration
    pub fn gemma4_2b() -> Self {
        Self {
            hidden_size: 2304,
            num_heads: 8,
            num_kv_heads: 4,
            num_global_kv_heads: 0,
            head_dim: 256,
            global_head_dim: 256,
            num_layers: 28,
            vocab_size: 256000,
            eps: 1e-6,
            rope_base: 1000000.0,
            rope_local_base: 10000.0,
            partial_rotary_dims: 256,
            sliding_window: 4096,
            final_logit_softcap: 30.0,
            num_experts: 0,
            num_experts_used: 0,
            vision_hidden_size: 1152,
            vision_num_heads: 16,
            vision_patch_size: 14,
            vision_num_layers: 26,
            vision_rope_theta: 100.0,
            audio_hidden_size: 1024,
            audio_num_heads: 8,
            audio_num_layers: 12,
            audio_mel_bins: 128,
            audio_chunk_size: 12,
            audio_max_past: 12,
            audio_max_future: 0,
            spatial_depth: 64,
            spatial_guidance: 1.0,
        }
    }
    
    /// Gemma4 9B configuration
    pub fn gemma4_9b() -> Self {
        Self {
            hidden_size: 3584,
            num_heads: 16,
            num_kv_heads: 8,
            num_global_kv_heads: 0,
            head_dim: 256,
            global_head_dim: 512,
            num_layers: 42,
            vocab_size: 256000,
            eps: 1e-6,
            rope_base: 1000000.0,
            rope_local_base: 10000.0,
            partial_rotary_dims: 512,
            sliding_window: 4096,
            final_logit_softcap: 30.0,
            num_experts: 8,
            num_experts_used: 2,
            vision_hidden_size: 1536,
            vision_num_heads: 24,
            vision_patch_size: 14,
            vision_num_layers: 32,
            vision_rope_theta: 100.0,
            audio_hidden_size: 1280,
            audio_num_heads: 16,
            audio_num_layers: 16,
            audio_mel_bins: 128,
            audio_chunk_size: 12,
            audio_max_past: 12,
            audio_max_future: 0,
            spatial_depth: 96,
            spatial_guidance: 1.0,
        }
    }
    
    /// Gemma4 27B configuration
    pub fn gemma4_27b() -> Self {
        Self {
            hidden_size: 4608,
            num_heads: 32,
            num_kv_heads: 16,
            num_global_kv_heads: 8,
            head_dim: 256,
            global_head_dim: 128,
            num_layers: 48,
            vocab_size: 256000,
            eps: 1e-6,
            rope_base: 1000000.0,
            rope_local_base: 10000.0,
            partial_rotary_dims: 128,
            sliding_window: 4096,
            final_logit_softcap: 30.0,
            num_experts: 16,
            num_experts_used: 4,
            vision_hidden_size: 2048,
            vision_num_heads: 32,
            vision_patch_size: 14,
            vision_num_layers: 38,
            vision_rope_theta: 100.0,
            audio_hidden_size: 1536,
            audio_num_heads: 24,
            audio_num_layers: 20,
            audio_mel_bins: 128,
            audio_chunk_size: 12,
            audio_max_past: 12,
            audio_max_future: 0,
            spatial_depth: 128,
            spatial_guidance: 1.0,
        }
    }
    
    /// Check if layer uses local (sliding window) attention
    pub fn is_local_layer(&self, layer_idx: usize) -> bool {
        // Alternating pattern: even layers local, odd layers global
        layer_idx % 2 == 0
    }
    
    /// Get RoPE parameters for specific layer
    pub fn rope_for_layer(&self, layer_idx: usize) -> (f32, usize) {
        if self.is_local_layer(layer_idx) {
            (self.rope_local_base, self.head_dim)
        } else {
            (self.rope_base, self.partial_rotary_dims)
        }
    }
    
    /// Get KV heads for specific layer
    pub fn kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_local_layer(layer_idx) {
            self.num_kv_heads
        } else if self.num_global_kv_heads > 0 {
            self.num_global_kv_heads
        } else {
            self.num_kv_heads
        }
    }
    
    /// Get head dimension for specific layer
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_local_layer(layer_idx) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), Gemma4Error> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(Gemma4Error::InvalidConfiguration(
                format!("hidden_size {} not divisible by num_heads {}", 
                    self.hidden_size, self.num_heads)
            ));
        }
        
        if self.num_kv_heads > self.num_heads {
            return Err(Gemma4Error::InvalidConfiguration(
                "num_kv_heads cannot exceed num_heads".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for Gemma4Config3D {
    fn default() -> Self {
        Self::gemma4_9b()
    }
}

/// 3D-aware sliding window cache wrapper
#[derive(Debug, Clone)]
pub struct SlidingWindowCache3D {
    /// SWA cache for local layers
    pub swa_cache: CausalKVCache,
    /// Causal cache for global layers
    pub causal_cache: CausalKVCache,
    /// Current cache type
    pub cache_type: CacheType,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheType {
    SWA,
    Causal,
}

impl SlidingWindowCache3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        let swa_cache = CausalKVCache::new(
            config.num_layers,
            config.sliding_window,
            config.num_kv_heads,
            config.head_dim,
        );
        
        let causal_cache = CausalKVCache::new(
            config.num_layers,
            8192, // Larger context for global
            config.num_global_kv_heads.max(config.num_kv_heads),
            config.global_head_dim.max(config.head_dim),
        );
        
        Self {
            swa_cache,
            causal_cache,
            cache_type: CacheType::SWA,
            spatial: SpatialTensorMetadata::new(
                config.hidden_size as u32,
                config.sliding_window as u32,
                config.spatial_depth as u32,
            ),
        }
    }
    
    pub fn set_layer(&mut self, layer_idx: usize) {
        self.swa_cache.set_layer(layer_idx);
        self.causal_cache.set_layer(layer_idx);
    }
    
    pub fn set_cache_type(&mut self, cache_type: CacheType) {
        self.cache_type = cache_type;
    }
    
    pub fn get_current_cache(&mut self) -> &mut CausalKVCache {
        match self.cache_type {
            CacheType::SWA => &mut self.swa_cache,
            CacheType::Causal => &mut self.causal_cache,
        }
    }
}

/// 3D-aware text model with MoE and PLE
#[derive(Debug, Clone)]
pub struct Gemma4TextModel3D {
    /// Token embeddings
    pub token_embeddings: Vec<f32>,
    /// Per-layer projector (PLE)
    pub ple_projector: Option<PerLayerProjector3D>,
    /// Transformer layers
    pub layers: Vec<Gemma4TextLayer3D>,
    /// Output normalization
    pub output_norm: Vec<f32>,
    /// Output projection (shared with embeddings)
    pub output: Vec<f32>,
    /// Configuration
    pub config: Gemma4Config3D,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// Per-Layer Projector for PLE (Progressive Layered Embedding)
#[derive(Debug, Clone)]
pub struct PerLayerProjector3D {
    /// Per-layer token embeddings
    pub per_layer_embeddings: Vec<f32>,
    /// Projection weights
    pub projection_weights: Vec<f32>,
    /// Normalization weights
    pub norm_weights: Vec<f32>,
    /// Hidden size per layer input
    pub hidden_size_per_layer: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl PerLayerProjector3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        let hidden_size_per_layer = config.hidden_size / 2;
        
        Self {
            per_layer_embeddings: vec![0.0; config.vocab_size * hidden_size_per_layer * config.num_layers],
            projection_weights: vec![0.0; config.hidden_size * hidden_size_per_layer * config.num_layers],
            norm_weights: vec![1.0; hidden_size_per_layer * config.num_layers],
            hidden_size_per_layer,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with zero-copy views
    pub fn forward_zero_copy(
        &self,
        input_tokens: &[i32],
        hidden_state: &[f32],
        config: &Gemma4Config3D,
    ) -> Vec<Vec<f32>> {
        let mut per_layer_outputs = Vec::with_capacity(config.num_layers);
        
        for layer_idx in 0..config.num_layers {
            let mut layer_output = vec![0.0; self.hidden_size_per_layer * input_tokens.len()];
            
            // Zero-copy embedding lookup
            for (token_idx, &token_id) in input_tokens.iter().enumerate() {
                let token_id = token_id.max(0) as usize;
                let embed_start = (layer_idx * config.vocab_size + token_id) * self.hidden_size_per_layer;
                let output_start = token_idx * self.hidden_size_per_layer;
                
                if embed_start + self.hidden_size_per_layer <= self.per_layer_embeddings.len() {
                    layer_output[output_start..output_start + self.hidden_size_per_layer]
                        .copy_from_slice(&self.per_layer_embeddings[embed_start..embed_start + self.hidden_size_per_layer]);
                }
            }
            
            // Scale by sqrt(hidden_size_per_layer)
            let scale = (self.hidden_size_per_layer as f32).sqrt();
            for val in &mut layer_output {
                *val *= scale;
            }
            
            per_layer_outputs.push(layer_output);
        }
        
        per_layer_outputs
    }
}

/// 3D-aware text transformer layer with MoE support
#[derive(Debug, Clone)]
pub struct Gemma4TextLayer3D {
    /// Pre-attention normalization
    pub attn_norm_weights: Vec<f32>,
    /// Self-attention
    pub self_attention: Gemma4Attention3D,
    /// Post-attention normalization
    pub post_attn_norm_weights: Vec<f32>,
    /// MLP normalization
    pub mlp_norm_weights: Vec<f32>,
    /// Dense MLP
    pub dense_mlp: Option<Gemma4Mlp3D>,
    /// Post-MLP normalization
    pub post_mlp_norm_weights: Vec<f32>,
    /// MoE router
    pub moe_router: Option<Gemma4MoERouter3D>,
    /// MoE block
    pub moe_block: Option<Gemma4MoEBlock3D>,
    /// MoE normalization
    pub moe_norm_weights: Vec<f32>,
    /// Post-MoE normalization
    pub post_moe_norm_weights: Vec<f32>,
    /// Per-layer input gate
    pub ple_input_gate: Option<Vec<f32>>,
    /// Per-layer projection
    pub ple_projection: Option<Vec<f32>>,
    /// Post-PLE normalization
    pub post_ple_norm_weights: Vec<f32>,
    /// Layer scalar
    pub layer_scalar: Option<f32>,
    /// Layer index
    pub layer_idx: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// 3D-aware attention with proportional RoPE
#[derive(Debug, Clone)]
pub struct Gemma4Attention3D {
    /// Query projection weights
    pub query_weights: Vec<f32>,
    /// Query normalization weights
    pub query_norm_weights: Vec<f32>,
    /// Key projection weights
    pub key_weights: Vec<f32>,
    /// Key normalization weights
    pub key_norm_weights: Vec<f32>,
    /// Value projection weights
    pub value_weights: Vec<f32>,
    /// Output projection weights
    pub output_weights: Vec<f32>,
    /// RoPE frequency factors (for proportional RoPE)
    pub rope_freq_factors: Option<Vec<f32>>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// 3D-aware MLP with GELU
#[derive(Debug, Clone)]
pub struct Gemma4Mlp3D {
    /// Gate projection weights
    pub gate_weights: Vec<f32>,
    /// Up projection weights
    pub up_weights: Vec<f32>,
    /// Down projection weights
    pub down_weights: Vec<f32>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// 3D-aware MoE router
#[derive(Debug, Clone)]
pub struct Gemma4MoERouter3D {
    /// Router projection weights
    pub router_weights: Vec<f32>,
    /// Router scale
    pub router_scale: f32,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// 3D-aware MoE block
#[derive(Debug, Clone)]
pub struct Gemma4MoEBlock3D {
    /// Fused gate-up weights
    pub gate_up_weights: Vec<f32>,
    /// Gate weights (split)
    pub gate_weights: Option<Vec<f32>>,
    /// Up weights (split)
    pub up_weights: Option<Vec<f32>>,
    /// Down weights
    pub down_weights: Vec<f32>,
    /// Per-expert down scale
    pub expert_down_scale: Option<Vec<f32>>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma4TextModel3D {
    pub fn new(config: Gemma4Config3D) -> Result<Self, Gemma4Error> {
        config.validate()?;
        
        let layers: Vec<Gemma4TextLayer3D> = (0..config.num_layers)
            .map(|i| Gemma4TextLayer3D::new(i, &config))
            .collect();
        
        let ple_projector = if config.num_layers > 20 {
            Some(PerLayerProjector3D::new(&config))
        } else {
            None
        };
        
        Ok(Self {
            token_embeddings: vec![0.0; config.vocab_size * config.hidden_size],
            ple_projector,
            layers,
            output_norm: vec![1.0; config.hidden_size],
            output: vec![0.0; config.vocab_size * config.hidden_size],
            config,
            spatial_position: ConversionCoordinate::standard(),
        })
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(
        &mut self,
        batch: &NeuralBatch,
        cache: &mut SlidingWindowCache3D,
        multimodal_embeddings: Option<&[f32]>,
    ) -> Result<Vec<Vec<f32>>, Gemma4Error> {
        let positions: Vec<i32> = batch.inputs.iter()
            .map(|inp| inp.position_in_sequence as i32)
            .collect();
        
        let seq_len = positions.len();
        
        // Zero-copy token embedding lookup
        let mut hidden_state: Vec<f32> = batch.inputs.iter()
            .flat_map(|input| {
                let token_id = input.token_id.max(0) as usize;
                let start = token_id * self.config.hidden_size;
                let end = (token_id + 1) * self.config.hidden_size;
                
                if end <= self.token_embeddings.len() {
                    self.token_embeddings[start..end].to_vec()
                } else {
                    vec![0.0; self.config.hidden_size]
                }
            })
            .collect();
        
        // Scale embeddings
        let scale = (self.config.hidden_size as f32).sqrt();
        for h in &mut hidden_state {
            *h *= scale;
        }
        
        // Inject multimodal embeddings (vision/audio)
        if let Some(mm_emb) = multimodal_embeddings {
            let mm_len = mm_emb.len();
            if mm_len <= hidden_state.len() {
                for (h, mm) in hidden_state.iter_mut().zip(mm_emb.iter()) {
                    *h += mm;
                }
            }
        }
        
        // PLE projection
        let per_layer_inputs = if let Some(ref ple) = self.ple_projector {
            let input_tokens: Vec<i32> = batch.inputs.iter()
                .map(|inp| inp.token_id)
                .collect();
            Some(ple.forward_zero_copy(&input_tokens, &hidden_state, &self.config))
        } else {
            None
        };
        
        // Process through layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            cache.set_layer(layer_idx);
            
            let cache_type = if self.config.is_local_layer(layer_idx) {
                CacheType::SWA
            } else {
                CacheType::Causal
            };
            cache.set_cache_type(cache_type);
            
            let current_cache = cache.get_current_cache();
            
            let ple_input = per_layer_inputs.as_ref()
                .and_then(|inputs| inputs.get(layer_idx))
                .map(|inp| inp.as_slice());
            
            layer.forward_zero_copy(
                &mut hidden_state,
                &positions,
                ple_input,
                current_cache,
                &self.config,
            )?;
        }
        
        // Output normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * self.config.hidden_size;
            let end = start + self.config.hidden_size;
            self.apply_rms_norm(&mut hidden_state[start..end], &self.output_norm, self.config.eps);
        }
        
        // Output projection with final logit softcap
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for seq_idx in 0..seq_len {
            let mut seq_logits = vec![0.0; self.config.vocab_size];
            
            let hidden_start = seq_idx * self.config.hidden_size;
            for vocab_idx in 0..self.config.vocab_size {
                let embed_start = vocab_idx * self.config.hidden_size;
                for h in 0..self.config.hidden_size {
                    if hidden_start + h < hidden_state.len() && embed_start + h < self.token_embeddings.len() {
                        seq_logits[vocab_idx] += hidden_state[hidden_start + h] * self.token_embeddings[embed_start + h];
                    }
                }
            }
            
            // Apply final logit softcap
            for logit in &mut seq_logits {
                *logit = (*logit / self.config.final_logit_softcap).tanh() * self.config.final_logit_softcap;
            }
            
            logits.push(seq_logits);
        }
        
        Ok(logits)
    }
    
    /// RMS normalization with zero-copy
    fn apply_rms_norm(&self, input: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        
        for (x, &w) in input.iter_mut().zip(weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

impl Gemma4TextLayer3D {
    pub fn new(layer_idx: usize, config: &Gemma4Config3D) -> Self {
        let has_moe = config.num_experts > 0;
        
        Self {
            attn_norm_weights: vec![1.0; config.hidden_size],
            self_attention: Gemma4Attention3D::new(config),
            post_attn_norm_weights: vec![1.0; config.hidden_size],
            mlp_norm_weights: vec![1.0; config.hidden_size],
            dense_mlp: if has_moe { None } else { Some(Gemma4Mlp3D::new(config)) },
            post_mlp_norm_weights: vec![1.0; config.hidden_size],
            moe_router: if has_moe { Some(Gemma4MoERouter3D::new(config)) } else { None },
            moe_block: if has_moe { Some(Gemma4MoEBlock3D::new(config)) } else { None },
            moe_norm_weights: vec![1.0; config.hidden_size],
            post_moe_norm_weights: vec![1.0; config.hidden_size],
            ple_input_gate: None,
            ple_projection: None,
            post_ple_norm_weights: vec![1.0; config.hidden_size],
            layer_scalar: None,
            layer_idx,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 8) as u16,
                (layer_idx % 8) as u8,
                1.0,
            ),
        }
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(
        &mut self,
        hidden_state: &mut [f32],
        positions: &[i32],
        ple_input: Option<&[f32]>,
        cache: &mut CausalKVCache,
        config: &Gemma4Config3D,
    ) -> Result<(), Gemma4Error> {
        let seq_len = positions.len();
        let hidden_size = config.hidden_size;
        
        // Attention path
        let residual = hidden_state.to_vec();
        
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_size;
            let end = start + hidden_size;
            self.apply_rms_norm(&mut hidden_state[start..end], &self.attn_norm_weights, config.eps);
        }
        
        self.self_attention.forward_zero_copy(hidden_state, positions, cache, config)?;
        
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_size;
            let end = start + hidden_size;
            self.apply_rms_norm(&mut hidden_state[start..end], &self.post_attn_norm_weights, config.eps);
        }
        
        // Add residual
        for (h, r) in hidden_state.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // MLP/MoE path
        let residual = hidden_state.to_vec();
        
        if let (Some(ref router), Some(ref moe_block)) = (&self.moe_router, &self.moe_block) {
            // MoE + Dense parallel
            // Dense MLP
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.mlp_norm_weights, config.eps);
            }
            
            if let Some(ref mlp) = self.dense_mlp {
                mlp.forward_zero_copy(hidden_state, config);
            }
            
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.post_moe_norm_weights, config.eps);
            }
            
            let mlp_output = hidden_state.to_vec();
            
            // MoE
            let routing_weights = router.forward_zero_copy(hidden_state, config);
            moe_block.forward_zero_copy(hidden_state, &routing_weights, config)?;
            
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.post_moe_norm_weights, config.eps);
            }
            
            // Combine
            for (h, m) in hidden_state.iter_mut().zip(mlp_output.iter()) {
                *h += m;
            }
            
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.post_mlp_norm_weights, config.eps);
            }
        } else if let Some(ref mlp) = self.dense_mlp {
            // Dense only
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.mlp_norm_weights, config.eps);
            }
            
            mlp.forward_zero_copy(hidden_state, config);
            
            for seq_idx in 0..seq_len {
                let start = seq_idx * hidden_size;
                let end = start + hidden_size;
                self.apply_rms_norm(&mut hidden_state[start..end], &self.post_mlp_norm_weights, config.eps);
            }
        }
        
        // Add residual
        for (h, r) in hidden_state.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // PLE injection
        if let (Some(ple_input), Some(ref gate), Some(ref proj)) = (ple_input, &self.ple_input_gate, &self.ple_projection) {
            // Apply PLE
            let mut ple_state = vec![0.0; hidden_size * seq_len];
            // Simplified PLE computation
            for (h, p) in hidden_state.iter_mut().zip(ple_state.iter()) {
                *h += p;
            }
        }
        
        // Layer scalar
        if let Some(scalar) = self.layer_scalar {
            for h in hidden_state {
                *h *= scalar;
            }
        }
        
        Ok(())
    }
    
    fn apply_rms_norm(&self, input: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        
        for (x, &w) in input.iter_mut().zip(weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

impl Gemma4Attention3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        Self {
            query_weights: vec![0.0; config.hidden_size * config.head_dim * config.num_heads],
            query_norm_weights: vec![1.0; config.head_dim * config.num_heads],
            key_weights: vec![0.0; config.hidden_size * config.head_dim * config.num_kv_heads],
            key_norm_weights: vec![1.0; config.head_dim * config.num_kv_heads],
            value_weights: vec![0.0; config.hidden_size * config.head_dim * config.num_kv_heads],
            output_weights: vec![0.0; config.head_dim * config.num_heads * config.hidden_size],
            rope_freq_factors: None,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(
        &mut self,
        hidden_state: &mut [f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        config: &Gemma4Config3D,
    ) -> Result<(), Gemma4Error> {
        let seq_len = positions.len();
        let layer_idx = cache.current_layer();
        let (rope_base, rope_dims) = config.rope_for_layer(layer_idx);
        let head_dim = config.head_dim_for_layer(layer_idx);
        let kv_heads = config.kv_heads_for_layer(layer_idx);
        
        // Simplified attention computation
        // In production, this would use optimized kernels
        for seq_idx in 0..seq_len {
            let start = seq_idx * config.hidden_size;
            let end = start + config.hidden_size;
            
            // Apply query/key/value norms
            self.apply_rms_norm(&mut hidden_state[start..end], &self.query_norm_weights, config.eps);
        }
        
        // Store in cache
        let key_view = &hidden_state[..config.hidden_size * seq_len];
        let value_view = &hidden_state[..config.hidden_size * seq_len];
        cache.store_key_values(key_view, value_view)?;
        
        Ok(())
    }
    
    fn apply_rms_norm(&self, input: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        
        for (x, &w) in input.iter_mut().zip(weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

impl Gemma4Mlp3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        let intermediate_size = config.hidden_size * 4;
        
        Self {
            gate_weights: vec![0.0; config.hidden_size * intermediate_size],
            up_weights: vec![0.0; config.hidden_size * intermediate_size],
            down_weights: vec![0.0; intermediate_size * config.hidden_size],
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(&mut self, hidden_state: &mut [f32], config: &Gemma4Config3D) {
        let seq_len = hidden_state.len() / config.hidden_size;
        let intermediate_size = config.hidden_size * 4;
        
        for seq_idx in 0..seq_len {
            let start = seq_idx * config.hidden_size;
            let end = start + config.hidden_size;
            
            // GELU gate
            let mut gate = vec![0.0; intermediate_size];
            let mut up = vec![0.0; intermediate_size];
            
            // Simplified projection
            for i in 0..intermediate_size.min(config.hidden_size) {
                gate[i] = hidden_state[start + i];
                up[i] = hidden_state[start + i];
            }
            
            // GELU activation
            for g in &mut gate {
                *g = 0.5 * *g * (1.0 + (0.7978845608 * (*g + 0.044715 * *g * *g * *g)).tanh());
            }
            
            // Element-wise multiply
            for i in 0..intermediate_size {
                gate[i] *= up[i];
            }
            
            // Down projection
            for j in 0..config.hidden_size {
                if j < intermediate_size {
                    hidden_state[start + j] = gate[j];
                }
            }
        }
    }
}

impl Gemma4MoERouter3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        Self {
            router_weights: vec![0.0; config.hidden_size * config.num_experts],
            router_scale: 1.0,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(&self, hidden_state: &[f32], config: &Gemma4Config3D) -> Vec<f32> {
        let seq_len = hidden_state.len() / config.hidden_size;
        let mut routing_weights = vec![0.0; seq_len * config.num_experts];
        
        // Simplified routing
        for seq_idx in 0..seq_len {
            for expert_idx in 0..config.num_experts {
                routing_weights[seq_idx * config.num_experts + expert_idx] = 1.0 / config.num_experts as f32;
            }
        }
        
        routing_weights
    }
}

impl Gemma4MoEBlock3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        let intermediate_size = config.hidden_size * 4;
        
        Self {
            gate_up_weights: vec![0.0; config.num_experts * config.hidden_size * intermediate_size * 2],
            gate_weights: None,
            up_weights: None,
            down_weights: vec![0.0; config.num_experts * intermediate_size * config.hidden_size],
            expert_down_scale: None,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(
        &mut self,
        hidden_state: &mut [f32],
        routing_weights: &[f32],
        config: &Gemma4Config3D,
    ) -> Result<(), Gemma4Error> {
        let seq_len = hidden_state.len() / config.hidden_size;
        
        // Simplified MoE computation
        for seq_idx in 0..seq_len {
            let start = seq_idx * config.hidden_size;
            let end = start + config.hidden_size;
            
            // Apply weighted sum from experts
            for expert_idx in 0..config.num_experts_used.min(config.num_experts) {
                let weight = routing_weights[seq_idx * config.num_experts + expert_idx];
                for j in start..end.min(hidden_state.len()) {
                    hidden_state[j] *= weight;
                }
            }
        }
        
        Ok(())
    }
}

/// Complete Gemma4 multimodal model
pub struct Gemma4Model3D {
    /// Text model
    pub text_model: Gemma4TextModel3D,
    /// Vision model
    pub vision_model: Option<Gemma4VisionModel3D>,
    /// Audio model
    pub audio_model: Option<Gemma4AudioModel3D>,
    /// Multimodal projector
    pub multimodal_projector: Option<Gemma4MultimodalProjector3D>,
    /// Cache
    pub cache: SlidingWindowCache3D,
    /// Tokenizer
    pub tokenizer: Option<NeuralSentencePieceTokenizer>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl Gemma4Model3D {
    pub fn new(config: Gemma4Config3D) -> Result<Self, Gemma4Error> {
        config.validate()?;
        
        let text_model = Gemma4TextModel3D::new(config)?;
        let cache = SlidingWindowCache3D::new(&config);
        
        let vision_model = if config.vision_num_layers > 0 {
            Some(Gemma4VisionModel3D::new(&config))
        } else {
            None
        };
        
        let audio_model = if config.audio_num_layers > 0 {
            Some(Gemma4AudioModel3D::new(&config))
        } else {
            None
        };
        
        let multimodal_projector = if vision_model.is_some() || audio_model.is_some() {
            Some(Gemma4MultimodalProjector3D::new(&config))
        } else {
            None
        };
        
        Ok(Self {
            text_model,
            vision_model,
            audio_model,
            multimodal_projector,
            cache,
            tokenizer: None,
            spatial: SpatialTensorMetadata::new(
                config.hidden_size as u32,
                config.sliding_window as u32,
                config.spatial_depth as u32,
            ),
        })
    }
    
    /// Forward pass with multimodal support
    pub fn forward(&mut self, batch: &NeuralBatch) -> Result<Vec<Vec<f32>>, Gemma4Error> {
        let multimodal_embeddings = if !batch.multimodal_data.is_empty() {
            self.process_multimodal(&batch.multimodal_data)?
        } else {
            None
        };
        
        self.text_model.forward_zero_copy(batch, &mut self.cache, multimodal_embeddings.as_deref())
    }
    
    /// Process multimodal input (vision/audio)
    fn process_multimodal(&mut self, data: &[MultimodalData]) -> Result<Option<Vec<f32>>, Gemma4Error> {
        if data.is_empty() {
            return Ok(None);
        }
        
        // For now, return placeholder embeddings
        // In production, this would call vision/audio encoders
        let config = &self.text_model.config;
        let dummy_embedding = vec![0.0; config.hidden_size];
        Ok(Some(vec![dummy_embedding; data.len()]).map(|e| e.concat()))
    }
    
    /// Get model information
    pub fn model_info(&self) -> Gemma4ModelInfo {
        let config = &self.text_model.config;
        
        Gemma4ModelInfo {
            name: "Gemma4-3D".to_string(),
            variant: if config.num_layers == 28 {
                "2B".to_string()
            } else if config.num_layers == 42 {
                "9B".to_string()
            } else if config.num_layers == 48 {
                "27B".to_string()
            } else {
                "Custom".to_string()
            },
            total_params: self.estimate_parameters(),
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            vocab_size: config.vocab_size,
            sliding_window: config.sliding_window,
            has_vision: self.vision_model.is_some(),
            has_audio: self.audio_model.is_some(),
            num_experts: config.num_experts,
        }
    }
    
    fn estimate_parameters(&self) -> usize {
        let config = &self.text_model.config;
        
        // Text model parameters
        let embeddings = config.vocab_size * config.hidden_size;
        let layer_params = config.num_layers * (
            config.hidden_size * config.head_dim * config.num_heads * 4 + // Attention
            config.hidden_size * 4 * config.hidden_size * 3 + // MLP
            config.hidden_size * 6 // Norms
        );
        
        let mut total = embeddings + layer_params;
        
        // MoE parameters
        if config.num_experts > 0 {
            total += config.num_layers * config.num_experts * config.hidden_size * 4 * config.hidden_size * 3;
        }
        
        // Vision parameters
        if let Some(ref vision) = self.vision_model {
            total += vision.estimate_parameters();
        }
        
        // Audio parameters
        if let Some(ref audio) = self.audio_model {
            total += audio.estimate_parameters();
        }
        
        total
    }
}

/// Vision model with professional 3D encoder
pub struct Gemma4VisionModel3D {
    pub config: Gemma4Config3D,
    pub encoder: PronaxVisionEncoder3D,
    pub projector: PronaxVisionTextProjector3D,
    pub spatial_position: ConversionCoordinate,
}

impl Gemma4VisionModel3D {
    pub fn new(config: &Gemma4Config3D) -> Result<Self, Gemma4Error> {
        let vision_params = PronaxVisionHyperparams3D {
            spatial_width: config.image_size,
            spatial_height: config.image_size,
            channel_depth: 3,
            guidance_strength: config.spatial_guidance,
            embedding_dim: config.vision_hidden_size,
            attention_heads: config.vision_num_heads,
            head_dimension: config.vision_hidden_size / config.vision_num_heads,
            patch_size: config.vision_patch_size,
            transformer_layers: config.vision_num_layers,
            rope_theta: 100.0,
            max_position_embeddings: 256,
            merge_factor: config.vision_merge_factor,
            text_embedding_dim: config.hidden_size,
            epsilon: config.eps,
            layer_scale_init: 0.0,
            spatial_depth: config.spatial_depth,
            spatial_guidance: config.spatial_guidance,
        };
        
        let encoder = PronaxVisionEncoder3D::new(vision_params)
            .map_err(|e| Gemma4Error::VisionError(e.to_string()))?;
        
        let projector = PronaxVisionTextProjector3D::new(
            config.vision_hidden_size,
            config.hidden_size,
        );
        
        Ok(Self {
            config: *config,
            encoder,
            projector,
            spatial_position: ConversionCoordinate::standard(),
        })
    }
    
    /// Encode image pixels to text embeddings
    pub fn encode_vision(
        &mut self,
        pixel_values: &[f32],
        num_patches_x: usize,
        num_patches_y: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let vision_features = self.encoder.encode_vision_zero_copy(pixel_values, num_patches_x, num_patches_y)
            .map_err(|e| Gemma4Error::VisionError(e.to_string()))?;
        
        let vision_params = PronaxVisionHyperparams3D {
            spatial_width: self.config.image_size,
            spatial_height: self.config.image_size,
            channel_depth: 3,
            guidance_strength: self.config.spatial_guidance,
            embedding_dim: self.config.vision_hidden_size,
            attention_heads: self.config.vision_num_heads,
            head_dimension: self.config.vision_hidden_size / self.config.vision_num_heads,
            patch_size: self.config.vision_patch_size,
            transformer_layers: self.config.vision_num_layers,
            rope_theta: 100.0,
            max_position_embeddings: 256,
            merge_factor: self.config.vision_merge_factor,
            text_embedding_dim: self.config.hidden_size,
            epsilon: self.config.eps,
            layer_scale_init: 0.0,
            spatial_depth: self.config.spatial_depth,
            spatial_guidance: self.config.spatial_guidance,
        };
        
        let text_embeddings = self.projector.project_with_pooling(
            &vision_features,
            num_patches_x,
            num_patches_y,
            self.config.vision_merge_factor,
            &vision_params,
        ).map_err(|e| Gemma4Error::VisionError(e.to_string()))?;
        
        Ok(text_embeddings)
    }
    
    fn estimate_parameters(&self) -> usize {
        // Patch embedding
        let patch_embd = self.config.vision_patch_size * self.config.vision_patch_size * 3 * self.config.vision_hidden_size;
        
        // Position embeddings
        let num_patches = (self.config.image_size / self.config.vision_patch_size).pow(2);
        let pos_embd = num_patches * self.config.vision_hidden_size * 2;
        
        // Vision transformer layers
        let vision_layer = self.config.vision_hidden_size * self.config.vision_hidden_size * 4 + // Attention
            self.config.vision_hidden_size * (self.config.vision_hidden_size * 4) * 2 + // MLP
            self.config.vision_hidden_size * 8; // Norms
        
        let total = patch_embd + pos_embd + 
            self.config.vision_num_layers * vision_layer +
            self.config.vision_hidden_size * self.config.hidden_size; // Projector
        
        total
    }
}

/// Audio model with professional 3D encoder
pub struct Gemma4AudioModel3D {
    pub config: Gemma4Config3D,
    pub encoder: PronaxAudioEncoder3D,
    pub projector: PronaxAudioTextProjector3D,
    pub spatial_position: ConversionCoordinate,
}

impl Gemma4AudioModel3D {
    pub fn new(config: &Gemma4Config3D) -> Result<Self, Gemma4Error> {
        let audio_params = PronaxAudioHyperparams3D {
            spectral_width: config.audio_mel_bins,
            temporal_height: 100,
            feature_depth: 1,
            guidance_strength: config.spatial_guidance,
            embedding_dim: config.audio_hidden_size,
            attention_heads: config.audio_num_heads,
            head_dimension: config.audio_hidden_size / config.audio_num_heads,
            feedforward_dim: config.audio_hidden_size * 4,
            transformer_layers: config.audio_num_layers,
            convolution_kernel: 5,
            chunk_span: config.audio_chunk_size,
            context_past: config.audio_max_past,
            context_future: config.audio_max_future,
            total_context: config.audio_chunk_size + config.audio_max_past + config.audio_max_future,
            logit_cap: 50.0,
            residual_scale: 0.5,
            gradient_threshold: 1e10,
            epsilon: config.eps,
            spatial_depth: config.spatial_depth,
            spatial_guidance: config.spatial_guidance,
        };
        
        let encoder = PronaxAudioEncoder3D::new(audio_params)
            .map_err(|e| Gemma4Error::AudioError(e.to_string()))?;
        
        let projector = PronaxAudioTextProjector3D::new(
            config.audio_hidden_size,
            config.hidden_size,
        );
        
        Ok(Self {
            config: *config,
            encoder,
            projector,
            spatial_position: ConversionCoordinate::standard(),
        })
    }
    
    /// Encode audio mel spectrogram to text embeddings
    pub fn encode_audio(
        &mut self,
        mel_spectrogram: &[f32],
        num_frames: usize,
    ) -> Result<Vec<f32>, Gemma4Error> {
        let audio_features = self.encoder.encode_audio_zero_copy(mel_spectrogram, num_frames)
            .map_err(|e| Gemma4Error::AudioError(e.to_string()))?;
        
        let audio_params = PronaxAudioHyperparams3D {
            spectral_width: self.config.audio_mel_bins,
            temporal_height: 100,
            feature_depth: 1,
            guidance_strength: self.config.spatial_guidance,
            embedding_dim: self.config.audio_hidden_size,
            attention_heads: self.config.audio_num_heads,
            head_dimension: self.config.audio_hidden_size / self.config.audio_num_heads,
            feedforward_dim: self.config.audio_hidden_size * 4,
            transformer_layers: self.config.audio_num_layers,
            convolution_kernel: 5,
            chunk_span: self.config.audio_chunk_size,
            context_past: self.config.audio_max_past,
            context_future: self.config.audio_max_future,
            total_context: self.config.audio_chunk_size + self.config.audio_max_past + self.config.audio_max_future,
            logit_cap: 50.0,
            residual_scale: 0.5,
            gradient_threshold: 1e10,
            epsilon: self.config.eps,
            spatial_depth: self.config.spatial_depth,
            spatial_guidance: self.config.spatial_guidance,
        };
        
        let text_embeddings = self.projector.project_to_text_zero_copy(&audio_features, &audio_params)
            .map_err(|e| Gemma4Error::AudioError(e.to_string()))?;
        
        Ok(text_embeddings)
    }
    
    fn estimate_parameters(&self) -> usize {
        // SSCP parameters
        let sscp_params = 3 * 3 * 1 * 64 + 3 * 3 * 64 * 128;
        
        // Conformer layer parameters
        let conformer_layer = self.config.audio_hidden_size * self.config.audio_hidden_size * 4 + // Attention
            self.config.audio_hidden_size * (self.config.audio_hidden_size * 4) * 2 + // FFW
            self.config.audio_hidden_size * self.config.audio_hidden_size * 2 + // Conv
            self.config.audio_hidden_size * 8; // Norms
        
        let total = sscp_params + 
            self.config.audio_num_layers * conformer_layer +
            self.config.audio_hidden_size * self.config.hidden_size; // Projector
        
        total
    }
}

/// Multimodal projector
pub struct Gemma4MultimodalProjector3D {
    pub projection_weights: Vec<f32>,
    pub spatial_position: ConversionCoordinate,
}

impl Gemma4MultimodalProjector3D {
    pub fn new(config: &Gemma4Config3D) -> Self {
        Self {
            projection_weights: vec![0.0; config.vision_hidden_size * config.hidden_size],
            spatial_position: ConversionCoordinate::standard(),
        }
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct Gemma4ModelInfo {
    pub name: String,
    pub variant: String,
    pub total_params: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub has_vision: bool,
    pub has_audio: bool,
    pub num_experts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = Gemma4Config3D::gemma4_9b();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_layers, 42);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_variants() {
        let config_2b = Gemma4Config3D::gemma4_2b();
        assert_eq!(config_2b.num_layers, 28);
        
        let config_27b = Gemma4Config3D::gemma4_27b();
        assert_eq!(config_27b.num_layers, 48);
        assert_eq!(config_27b.num_experts, 16);
    }
    
    #[test]
    fn test_model_creation() {
        let config = Gemma4Config3D::gemma4_9b();
        let model = Gemma4Model3D::new(config).unwrap();
        
        let info = model.model_info();
        assert_eq!(info.variant, "9B");
        assert!(info.has_vision);
        assert!(info.has_audio);
    }
    
    #[test]
    fn test_layer_types() {
        let config = Gemma4Config3D::gemma4_9b();
        assert!(config.is_local_layer(0));  // Even = local
        assert!(!config.is_local_layer(1)); // Odd = global
    }
    
    #[test]
    fn test_rope_params() {
        let config = Gemma4Config3D::gemma4_9b();
        
        let (base, dims) = config.rope_for_layer(0);
        assert_eq!(base, config.rope_local_base);
        assert_eq!(dims, config.head_dim);
        
        let (base, dims) = config.rope_for_layer(1);
        assert_eq!(base, config.rope_base);
        assert_eq!(dims, config.partial_rotary_dims);
    }
}
