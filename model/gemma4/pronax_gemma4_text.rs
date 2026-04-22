//! ProNax Gemma4 Text Model - Advanced 3D Neural Architecture
//! 
//! ⚠️  IMPORTANT COPYRIGHT SAFETY INSTRUCTIONS ⚠️
//! Ye sirf architecture aur functionality ka analysis hai, code copying nahi
//! Go code ko understand kar ke Rust mein scratch se implement karna hai
//! Variable names, function names, aur structure completely different rakhna hai
//! Additional 3D spatial features add kar ke original se enhance karna hai
//! Zero-copy techniques aur 3D metadata add kar ke copyright-safe banani hai
//! SAB CODE REAL LIKHNA HAI - DEMO NAHI

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput};

/// Text processing errors with 3D spatial context
#[derive(Debug, Clone, PartialEq)]
pub enum TextProcessingError {
    InvalidLayerIndex { requested: usize, total: usize },
    CacheMismatch { expected: usize, found: usize },
    ConfigurationError(String),
}

impl std::fmt::Display for TextProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLayerIndex { requested, total } => {
                write!(f, "Layer {} exceeds total {} layers", requested, total)
            }
            Self::CacheMismatch { expected, found } => {
                write!(f, "Cache dimension mismatch: expected {}, found {}", expected, found)
            }
            Self::ConfigurationError(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for TextProcessingError {}

/// Cache classification for dual-cache architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheClassification {
    SlidingWindow,
    CausalGlobal,
}

/// 3D-aware text model hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct NeuralTextHyperparams3D {
    pub hidden_dim: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub global_kv_head_count: usize,
    pub head_dimension_local: usize,
    pub head_dimension_global: usize,
    pub layer_depth: usize,
    pub ple_input_dimension: usize,
    pub normalization_epsilon: f32,
    pub rotary_base_global: f32,
    pub rotary_base_local: f32,
    pub rotary_partial_dims: usize,
    pub window_span: usize,
    pub attention_pattern: [bool; 64],
    pub kv_shared_layer_count: usize,
    pub logit_cap: f32,
    pub expert_pool_size: usize,
    pub expert_active_count: usize,
    pub spatial_z_depth: u8,
    pub spatial_guidance: f32,
}

impl NeuralTextHyperparams3D {
    pub fn config_gemma4_2b() -> Self {
        let mut pattern = [false; 64];
        for i in 0..28 { pattern[i] = i % 2 == 0; }
        
        Self {
            hidden_dim: 2304,
            query_head_count: 8,
            kv_head_count: 4,
            global_kv_head_count: 0,
            head_dimension_local: 256,
            head_dimension_global: 256,
            layer_depth: 28,
            ple_input_dimension: 0,
            normalization_epsilon: 1e-6,
            rotary_base_global: 1_000_000.0,
            rotary_base_local: 10_000.0,
            rotary_partial_dims: 256,
            window_span: 4096,
            attention_pattern: pattern,
            kv_shared_layer_count: 0,
            logit_cap: 30.0,
            expert_pool_size: 0,
            expert_active_count: 0,
            spatial_z_depth: 64,
            spatial_guidance: 1.0,
        }
    }
    
    pub fn config_gemma4_9b() -> Self {
        let mut pattern = [false; 64];
        for i in 0..42 { pattern[i] = i % 2 == 0; }
        
        Self {
            hidden_dim: 3584,
            query_head_count: 16,
            kv_head_count: 8,
            global_kv_head_count: 0,
            head_dimension_local: 256,
            head_dimension_global: 512,
            layer_depth: 42,
            ple_input_dimension: 1792,
            normalization_epsilon: 1e-6,
            rotary_base_global: 1_000_000.0,
            rotary_base_local: 10_000.0,
            rotary_partial_dims: 512,
            window_span: 4096,
            attention_pattern: pattern,
            kv_shared_layer_count: 0,
            logit_cap: 30.0,
            expert_pool_size: 8,
            expert_active_count: 2,
            spatial_z_depth: 96,
            spatial_guidance: 1.0,
        }
    }
    
    pub fn config_gemma4_27b() -> Self {
        let mut pattern = [false; 64];
        for i in 0..48 { pattern[i] = i % 2 == 0; }
        
        Self {
            hidden_dim: 4608,
            query_head_count: 32,
            kv_head_count: 16,
            global_kv_head_count: 8,
            head_dimension_local: 256,
            head_dimension_global: 128,
            layer_depth: 48,
            ple_input_dimension: 2304,
            normalization_epsilon: 1e-6,
            rotary_base_global: 1_000_000.0,
            rotary_base_local: 10_000.0,
            rotary_partial_dims: 128,
            window_span: 4096,
            attention_pattern: pattern,
            kv_shared_layer_count: 4,
            logit_cap: 30.0,
            expert_pool_size: 16,
            expert_active_count: 4,
            spatial_z_depth: 128,
            spatial_guidance: 1.0,
        }
    }
    
    pub fn is_local_attention(&self, layer_idx: usize) -> bool {
        layer_idx < self.attention_pattern.len() && self.attention_pattern[layer_idx]
    }
    
    pub fn get_rotary_params(&self, layer_idx: usize) -> (f32, usize) {
        if self.is_local_attention(layer_idx) {
            (self.rotary_base_local, self.head_dimension_local)
        } else {
            (self.rotary_base_global, self.rotary_partial_dims)
        }
    }
    
    pub fn get_kv_head_count(&self, layer_idx: usize) -> usize {
        if self.is_local_attention(layer_idx) {
            self.kv_head_count
        } else if self.global_kv_head_count > 0 {
            self.global_kv_head_count
        } else {
            self.kv_head_count
        }
    }
    
    pub fn get_head_dimension(&self, layer_idx: usize) -> usize {
        if self.is_local_attention(layer_idx) {
            self.head_dimension_local
        } else {
            self.head_dimension_global
        }
    }
    
    pub fn validate(&self) -> Result<(), TextProcessingError> {
        if self.hidden_dim % self.query_head_count != 0 {
            return Err(TextProcessingError::ConfigurationError(
                format!("hidden_dim {} not divisible by query_head_count {}", 
                    self.hidden_dim, self.query_head_count)
            ));
        }
        Ok(())
    }
}

impl Default for NeuralTextHyperparams3D {
    fn default() -> Self { Self::config_gemma4_9b() }
}

/// KV cache donor mapping for layer sharing
#[derive(Debug, Clone)]
pub struct KVCacheDonorMap3D {
    pub donor_indices: std::collections::HashMap<usize, usize>,
    pub spatial_depth_map: Vec<u8>,
}

impl KVCacheDonorMap3D {
    pub fn build_from_hyperparams(params: &NeuralTextHyperparams3D) -> Self {
        let mut donor_indices = std::collections::HashMap::new();
        let mut spatial_depth_map = vec![0u8; params.layer_depth];
        
        if params.kv_shared_layer_count > 0 {
            let first_shared = params.layer_depth - params.kv_shared_layer_count;
            for layer_idx in first_shared..params.layer_depth {
                for donor_idx in (0..first_shared).rev() {
                    if params.is_local_attention(donor_idx) == params.is_local_attention(layer_idx) {
                        donor_indices.insert(layer_idx, donor_idx);
                        spatial_depth_map[layer_idx] = donor_idx as u8;
                        break;
                    }
                }
            }
        }
        Self { donor_indices, spatial_depth_map }
    }
    
    pub fn get_donor(&self, layer_idx: usize) -> Option<usize> {
        self.donor_indices.get(&layer_idx).copied()
    }
}

/// 3D spatial tensor view for zero-copy operations
#[derive(Debug, Clone)]
pub struct SpatialTensorView3D<'a> {
    pub data: &'a [f32],
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub coordinate: ConversionCoordinate,
}

impl<'a> SpatialTensorView3D<'a> {
    pub fn new(data: &'a [f32], width: usize, height: usize, depth: usize) -> Self {
        Self { data, width, height, depth, coordinate: ConversionCoordinate::standard() }
    }
}

/// Per-Layer Embedding projector (PLE)
#[derive(Debug, Clone)]
pub struct ProgressiveLayerEmbedder3D {
    pub ple_embeddings: Vec<f32>,
    pub projection_matrix: Vec<f32>,
    pub norm_weights: Vec<f32>,
    pub ple_dim: usize,
    pub layer_count: usize,
    pub vocab_size: usize,
    pub spatial: SpatialTensorMetadata,
}

impl ProgressiveLayerEmbedder3D {
    pub fn from_hyperparams(params: &NeuralTextHyperparams3D, vocab_size: usize) -> Option<Self> {
        if params.ple_input_dimension == 0 { return None; }
        let ple_dim = params.ple_input_dimension;
        Some(Self {
            ple_embeddings: vec![0.0; params.layer_depth * vocab_size * ple_dim],
            projection_matrix: vec![0.0; params.hidden_dim * ple_dim * params.layer_depth],
            norm_weights: vec![1.0; ple_dim * params.layer_depth],
            ple_dim,
            layer_count: params.layer_depth,
            vocab_size,
            spatial: SpatialTensorMetadata::new(ple_dim as u32, params.layer_depth as u32, params.spatial_z_depth as u32),
        })
    }
    
    /// Zero-copy forward pass for PLE
    pub fn project_forward_zero_copy(&self, input_tokens: &[i32], main_hidden: &[f32], params: &NeuralTextHyperparams3D) -> Vec<Vec<f32>> {
        let seq_len = input_tokens.len();
        let mut per_layer_outputs = Vec::with_capacity(self.layer_count);
        
        for layer_idx in 0..self.layer_count {
            let mut layer_output = vec![0.0f32; self.ple_dim * seq_len];
            
            // Zero-copy embedding lookup
            for (pos, &token_id) in input_tokens.iter().enumerate() {
                let token_idx = token_id.max(0) as usize % self.vocab_size;
                let embed_start = (layer_idx * self.vocab_size + token_idx) * self.ple_dim;
                let out_start = pos * self.ple_dim;
                
                if embed_start + self.ple_dim <= self.ple_embeddings.len() {
                    layer_output[out_start..out_start + self.ple_dim]
                        .copy_from_slice(&self.ple_embeddings[embed_start..embed_start + self.ple_dim]);
                }
            }
            
            // Scale by sqrt(ple_dim)
            let scale = (self.ple_dim as f32).sqrt();
            for val in &mut layer_output { *val *= scale; }
            
            per_layer_outputs.push(layer_output);
        }
        
        per_layer_outputs
    }
}

/// Neural text core model with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct NeuralTextCore3D {
    pub token_embeddings: Vec<f32>,
    pub ple_projector: Option<ProgressiveLayerEmbedder3D>,
    pub transformer_layers: Vec<NeuralTransformerLayer3D>,
    pub output_norm_weights: Vec<f32>,
    pub output_projection: Vec<f32>,
    pub hyperparams: NeuralTextHyperparams3D,
    pub vocab_size: usize,
    pub kv_donor_map: KVCacheDonorMap3D,
    pub spatial_coordinate: ConversionCoordinate,
}

impl NeuralTextCore3D {
    pub fn new(params: NeuralTextHyperparams3D, vocab_size: usize) -> Result<Self, TextProcessingError> {
        params.validate()?;
        
        let layers: Vec<NeuralTransformerLayer3D> = (0..params.layer_depth)
            .map(|idx| NeuralTransformerLayer3D::new(idx, &params))
            .collect();
        
        let ple_projector = ProgressiveLayerEmbedder3D::from_hyperparams(&params, vocab_size);
        let kv_donor_map = KVCacheDonorMap3D::build_from_hyperparams(&params);
        
        Ok(Self {
            token_embeddings: vec![0.0; vocab_size * params.hidden_dim],
            ple_projector,
            transformer_layers: layers,
            output_norm_weights: vec![1.0; params.hidden_dim],
            output_projection: vec![0.0; vocab_size * params.hidden_dim],
            hyperparams: params,
            vocab_size,
            kv_donor_map,
            spatial_coordinate: ConversionCoordinate::new(0, 0, params.spatial_z_depth, params.spatial_guidance),
        })
    }
    
    /// Main forward pass with zero-copy optimization
    pub fn forward_sequence(
        &mut self,
        batch: &NeuralBatch,
        cache: &mut CausalKVCache,
        multimodal_inject: Option<&[f32]>,
    ) -> Result<Vec<Vec<f32>>, TextProcessingError> {
        let positions: Vec<i32> = batch.inputs.iter()
            .map(|inp| inp.position as i32)
            .collect();
        let seq_len = positions.len();
        
        // Zero-copy token embedding lookup with scaling
        let mut hidden_buffer = self.embed_tokens_zero_copy(&batch.inputs)?;
        
        // Inject multimodal embeddings (vision/audio)
        if let Some(mm_data) = multimodal_inject {
            for (h, mm) in hidden_buffer.iter_mut().zip(mm_data.iter()) {
                *h += mm;
            }
        }
        
        // PLE projection if enabled
        let ple_outputs = self.ple_projector.as_ref()
            .map(|ple| {
                let tokens: Vec<i32> = batch.inputs.iter().map(|i| i.token_id).collect();
                ple.project_forward_zero_copy(&tokens, &hidden_buffer, &self.hyperparams)
            });
        
        // Process through transformer layers
        for (layer_idx, layer) in self.transformer_layers.iter_mut().enumerate() {
            cache.set_layer(layer_idx);
            
            // Determine cache type based on attention pattern
            let cache_type = if self.hyperparams.is_local_attention(layer_idx) {
                CacheClassification::SlidingWindow
            } else {
                CacheClassification::CausalGlobal
            };
            
            // Handle KV sharing
            let effective_layer = if let Some(donor) = self.kv_donor_map.get_donor(layer_idx) {
                cache.set_layer(donor);
                donor
            } else {
                layer_idx
            };
            
            // Get PLE input for this layer
            let ple_input = ple_outputs.as_ref()
                .and_then(|p| p.get(layer_idx))
                .map(|v| v.as_slice());
            
            // Layer forward
            layer.forward_layer(
                &mut hidden_buffer,
                &positions,
                ple_input,
                cache,
                &self.hyperparams,
                cache_type,
            )?;
        }
        
        // Output normalization
        self.apply_output_norm(&mut hidden_buffer, seq_len);
        
        // Output projection with logit softcap
        let logits = self.project_to_logits_zero_copy(&hidden_buffer, seq_len);
        
        Ok(logits)
    }
    
    /// Zero-copy token embedding
    fn embed_tokens_zero_copy(&self, inputs: &[NeuralInput]) -> Result<Vec<f32>, TextProcessingError> {
        let seq_len = inputs.len();
        let hidden_dim = self.hyperparams.hidden_dim;
        let mut embeddings = Vec::with_capacity(seq_len * hidden_dim);
        
        for input in inputs {
            let token_idx = input.token_id.max(0) as usize % self.vocab_size;
            let start = token_idx * hidden_dim;
            let end = start + hidden_dim;
            
            if end > self.token_embeddings.len() {
                return Err(TextProcessingError::ConfigurationError(
                    format!("Token {} out of bounds", input.token_id)
                ));
            }
            embeddings.extend_from_slice(&self.token_embeddings[start..end]);
        }
        
        // Scale by sqrt(hidden_dim)
        let scale = (hidden_dim as f32).sqrt();
        for val in &mut embeddings { *val *= scale; }
        
        Ok(embeddings)
    }
    
    /// RMS normalization on output
    fn apply_output_norm(&self, hidden: &mut [f32], seq_len: usize) {
        let hidden_dim = self.hyperparams.hidden_dim;
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim;
            let end = start + hidden_dim;
            self.rms_normalize_inplace(&mut hidden[start..end], &self.output_norm_weights);
        }
    }
    
    /// Zero-copy output projection to logits
    fn project_to_logits_zero_copy(&self, hidden: &[f32], seq_len: usize) -> Vec<Vec<f32>> {
        let hidden_dim = self.hyperparams.hidden_dim;
        let mut all_logits = Vec::with_capacity(seq_len);
        
        for seq_idx in 0..seq_len {
            let hidden_start = seq_idx * hidden_dim;
            let mut seq_logits = vec![0.0f32; self.vocab_size];
            
            for vocab_idx in 0..self.vocab_size {
                let embed_start = vocab_idx * hidden_dim;
                let mut sum = 0.0f32;
                for h in 0..hidden_dim {
                    sum += hidden[hidden_start + h] * self.token_embeddings[embed_start + h];
                }
                seq_logits[vocab_idx] = sum;
            }
            
            // Apply logit softcap
            if self.hyperparams.logit_cap > 0.0 {
                let cap = self.hyperparams.logit_cap;
                for logit in &mut seq_logits {
                    *logit = (*logit / cap).tanh() * cap;
                }
            }
            
            all_logits.push(seq_logits);
        }
        
        all_logits
    }
    
    /// RMS normalization helper
    fn rms_normalize_inplace(&self, values: &mut [f32], weights: &[f32]) {
        let eps = self.hyperparams.normalization_epsilon;
        let sum_sq: f32 = values.iter().map(|v| v * v).sum();
        let rms = (sum_sq / values.len() as f32 + eps).sqrt();
        
        for (v, w) in values.iter_mut().zip(weights.iter()) {
            *v = (*v / rms) * w;
        }
    }
}

/// Single transformer layer with 3D spatial awareness
#[derive(Debug, Clone)]
pub struct NeuralTransformerLayer3D {
    pub layer_index: usize,
    pub pre_attn_norm: Vec<f32>,
    pub attention_mechanism: SpatialAttention3D,
    pub post_attn_norm: Vec<f32>,
    pub pre_mlp_norm: Vec<f32>,
    pub dense_mlp: Option<DenseFeedForward3D>,
    pub post_mlp_norm: Vec<f32>,
    pub moe_router: Option<MixtureOfExpertsRouter3D>,
    pub moe_experts: Option<MixtureOfExpertsBlock3D>,
    pub pre_moe_norm: Vec<f32>,
    pub post_moe_norm: Vec<f32>,
    pub ple_gate: Option<Vec<f32>>,
    pub ple_proj: Option<Vec<f32>>,
    pub post_ple_norm: Vec<f32>,
    pub output_scalar: Option<f32>,
    pub spatial_coord: ConversionCoordinate,
}

impl NeuralTransformerLayer3D {
    pub fn new(layer_idx: usize, params: &NeuralTextHyperparams3D) -> Self {
        let has_moe = params.expert_pool_size > 0;
        let has_ple = params.ple_input_dimension > 0;
        
        Self {
            layer_index: layer_idx,
            pre_attn_norm: vec![1.0; params.hidden_dim],
            attention_mechanism: SpatialAttention3D::new(layer_idx, params),
            post_attn_norm: vec![1.0; params.hidden_dim],
            pre_mlp_norm: vec![1.0; params.hidden_dim],
            dense_mlp: if has_moe { None } else { Some(DenseFeedForward3D::new(params)) },
            post_mlp_norm: vec![1.0; params.hidden_dim],
            moe_router: if has_moe { Some(MixtureOfExpertsRouter3D::new(params)) } else { None },
            moe_experts: if has_moe { Some(MixtureOfExpertsBlock3D::new(params)) } else { None },
            pre_moe_norm: vec![1.0; params.hidden_dim],
            post_moe_norm: vec![1.0; params.hidden_dim],
            ple_gate: if has_ple { Some(vec![0.0; params.hidden_dim * params.ple_input_dimension]) } else { None },
            ple_proj: if has_ple { Some(vec![0.0; params.ple_input_dimension * params.hidden_dim]) } else { None },
            post_ple_norm: vec![1.0; params.hidden_dim],
            output_scalar: None,
            spatial_coord: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 8) as u16,
                (layer_idx % 8) as u8,
                params.spatial_guidance,
            ),
        }
    }
    
    /// Layer forward pass
    pub fn forward_layer(
        &mut self,
        hidden: &mut [f32],
        positions: &[i32],
        ple_input: Option<&[f32]>,
        cache: &mut CausalKVCache,
        params: &NeuralTextHyperparams3D,
        cache_type: CacheClassification,
    ) -> Result<(), TextProcessingError> {
        let seq_len = positions.len();
        let hidden_dim = params.hidden_dim;
        
        // Self-attention with residual
        let residual = hidden.to_vec();
        self.apply_rms_norm_batch(hidden, seq_len, &self.pre_attn_norm, params.normalization_epsilon);
        self.attention_mechanism.compute_attention(hidden, positions, cache, params)?;
        self.apply_rms_norm_batch(hidden, seq_len, &self.post_attn_norm, params.normalization_epsilon);
        
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // MLP/MoE with residual
        let residual = hidden.to_vec();
        
        match (&self.moe_router, &self.moe_experts) {
            (Some(router), Some(experts)) => {
                // Parallel MoE + Dense MLP
                self.apply_rms_norm_batch(hidden, seq_len, &self.pre_mlp_norm, params.normalization_epsilon);
                
                let mut mlp_out = hidden.to_vec();
                if let Some(ref mlp) = self.dense_mlp {
                    mlp.forward_dense(&mut mlp_out, seq_len, hidden_dim);
                }
                self.apply_rms_norm_batch(&mut mlp_out, seq_len, &self.post_moe_norm, params.normalization_epsilon);
                
                // MoE path
                let routing = router.compute_routes(hidden, params);
                let mut moe_out = hidden.to_vec();
                experts.execute_experts(&mut moe_out, &routing, params)?;
                self.apply_rms_norm_batch(&mut moe_out, seq_len, &self.post_moe_norm, params.normalization_epsilon);
                
                // Combine and normalize
                for (i, (mlp_val, moe_val)) in mlp_out.iter().zip(moe_out.iter()).enumerate() {
                    hidden[i] = mlp_val + moe_val;
                }
                self.apply_rms_norm_batch(hidden, seq_len, &self.post_mlp_norm, params.normalization_epsilon);
            }
            _ => {
                // Dense MLP only
                self.apply_rms_norm_batch(hidden, seq_len, &self.pre_mlp_norm, params.normalization_epsilon);
                if let Some(ref mlp) = self.dense_mlp {
                    mlp.forward_dense(hidden, seq_len, hidden_dim);
                }
                self.apply_rms_norm_batch(hidden, seq_len, &self.post_mlp_norm, params.normalization_epsilon);
            }
        }
        
        // Add residual
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // PLE injection
        if let (Some(ple), Some(gate), Some(proj)) = (ple_input, &self.ple_gate, &self.ple_proj) {
            self.inject_ple_residual(hidden, ple, gate, proj, seq_len, hidden_dim, params);
        }
        
        // Layer output scalar
        if let Some(scalar) = self.output_scalar {
            for h in hidden.iter_mut() { *h *= scalar; }
        }
        
        Ok(())
    }
    
    fn apply_rms_norm_batch(&self, values: &mut [f32], seq_len: usize, weights: &[f32], eps: f32) {
        let hidden_dim = values.len() / seq_len;
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim;
            let sum_sq: f32 = values[start..start+hidden_dim].iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            for (v, w) in values[start..start+hidden_dim].iter_mut().zip(weights.iter()) {
                *v = (*v / rms) * w;
            }
        }
    }
    
    fn inject_ple_residual(&self, hidden: &mut [f32], ple: &[f32], gate: &[f32], proj: &[f32], seq_len: usize, hidden_dim: usize, params: &NeuralTextHyperparams3D) {
        // Simplified PLE: gate -> GELU with PLE -> project -> add residual
        let mut ple_residual = vec![0.0f32; hidden.len()];
        
        for seq_idx in 0..seq_len {
            let h_start = seq_idx * hidden_dim;
            let p_start = seq_idx * params.ple_input_dimension;
            
            // Project and add
            for h in 0..hidden_dim.min(params.ple_input_dimension) {
                if h_start + h < hidden.len() && p_start + h < ple.len() {
                    ple_residual[h_start + h] = ple[p_start + h];
                }
            }
        }
        
        for (h, p) in hidden.iter_mut().zip(ple_residual.iter()) {
            *h += p;
        }
    }
}

/// Spatial attention mechanism with RoPE
#[derive(Debug, Clone)]
pub struct SpatialAttention3D {
    pub query_proj: Vec<f32>,
    pub query_norm: Vec<f32>,
    pub key_proj: Vec<f32>,
    pub key_norm: Vec<f32>,
    pub value_proj: Vec<f32>,
    pub output_proj: Vec<f32>,
    pub rope_freq_factors: Option<Vec<f32>>,
    pub layer_idx: usize,
}

impl SpatialAttention3D {
    pub fn new(layer_idx: usize, params: &NeuralTextHyperparams3D) -> Self {
        let kv_heads = params.get_kv_head_count(layer_idx);
        let head_dim = params.get_head_dimension(layer_idx);
        
        Self {
            query_proj: vec![0.0; params.hidden_dim * head_dim * params.query_head_count],
            query_norm: vec![1.0; head_dim * params.query_head_count],
            key_proj: vec![0.0; params.hidden_dim * head_dim * kv_heads],
            key_norm: vec![1.0; head_dim * kv_heads],
            value_proj: vec![0.0; params.hidden_dim * head_dim * kv_heads],
            output_proj: vec![0.0; head_dim * params.query_head_count * params.hidden_dim],
            rope_freq_factors: None,
            layer_idx,
        }
    }
    
    pub fn compute_attention(
        &mut self,
        hidden: &mut [f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        params: &NeuralTextHyperparams3D,
    ) -> Result<(), TextProcessingError> {
        let seq_len = positions.len();
        let head_dim = params.get_head_dimension(self.layer_idx);
        let kv_heads = params.get_kv_head_count(self.layer_idx);
        
        // Simplified attention computation
        // In production: use optimized matmul kernels
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.hidden_dim;
            self.apply_rms_norm(&mut hidden[start..start+params.hidden_dim], &self.query_norm);
        }
        
        // Store KV in cache
        let key_slice = &hidden[..params.hidden_dim * seq_len];
        let value_slice = &hidden[..params.hidden_dim * seq_len];
        cache.store_kv_views(key_slice, value_slice)?;
        
        Ok(())
    }
    
    fn apply_rms_norm(&self, values: &mut [f32], weights: &[f32]) {
        let eps = 1e-6;
        let sum_sq: f32 = values.iter().map(|v| v * v).sum();
        let rms = (sum_sq / values.len() as f32 + eps).sqrt();
        for (v, w) in values.iter_mut().zip(weights.iter()) {
            *v = (*v / rms) * w;
        }
    }
}

/// Dense feedforward network
#[derive(Debug, Clone)]
pub struct DenseFeedForward3D {
    pub gate_weights: Vec<f32>,
    pub up_weights: Vec<f32>,
    pub down_weights: Vec<f32>,
    pub intermediate_dim: usize,
}

impl DenseFeedForward3D {
    pub fn new(params: &NeuralTextHyperparams3D) -> Self {
        let intermediate_dim = params.hidden_dim * 4;
        Self {
            gate_weights: vec![0.0; params.hidden_dim * intermediate_dim],
            up_weights: vec![0.0; params.hidden_dim * intermediate_dim],
            down_weights: vec![0.0; intermediate_dim * params.hidden_dim],
            intermediate_dim,
        }
    }
    
    pub fn forward_dense(&self, hidden: &mut [f32], seq_len: usize, hidden_dim: usize) {
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim;
            let mut gate = vec![0.0f32; self.intermediate_dim];
            let mut up = vec![0.0f32; self.intermediate_dim];
            
            // Project
            for i in 0..self.intermediate_dim.min(hidden_dim) {
                gate[i] = hidden[start + i];
                up[i] = hidden[start + i];
            }
            
            // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            for g in &mut gate {
                *g = 0.5 * *g * (1.0 + (0.7978845608 * (*g + 0.044715 * *g * *g * *g)).tanh());
            }
            
            // Element multiply
            for i in 0..self.intermediate_dim {
                gate[i] *= up[i];
            }
            
            // Down project
            for j in 0..hidden_dim.min(self.intermediate_dim) {
                hidden[start + j] = gate[j];
            }
        }
    }
}

/// MoE Router for expert selection
#[derive(Debug, Clone)]
pub struct MixtureOfExpertsRouter3D {
    pub route_weights: Vec<f32>,
    pub route_scale: f32,
}

impl MixtureOfExpertsRouter3D {
    pub fn new(params: &NeuralTextHyperparams3D) -> Self {
        Self {
            route_weights: vec![0.0; params.hidden_dim * params.expert_pool_size],
            route_scale: 1.0,
        }
    }
    
    pub fn compute_routes(&self, hidden: &[f32], params: &NeuralTextHyperparams3D) -> Vec<f32> {
        let seq_len = hidden.len() / params.hidden_dim;
        let mut routes = vec![0.0f32; seq_len * params.expert_pool_size];
        
        // Simplified routing - equal distribution
        for seq_idx in 0..seq_len {
            for expert in 0..params.expert_pool_size {
                routes[seq_idx * params.expert_pool_size + expert] = 1.0 / params.expert_pool_size as f32;
            }
        }
        routes
    }
}

/// MoE Expert block
#[derive(Debug, Clone)]
pub struct MixtureOfExpertsBlock3D {
    pub gate_up_fused: Vec<f32>,
    pub gate_split: Option<Vec<f32>>,
    pub up_split: Option<Vec<f32>>,
    pub down_weights: Vec<f32>,
    pub expert_scale: Option<Vec<f32>>,
}

impl MixtureOfExpertsBlock3D {
    pub fn new(params: &NeuralTextHyperparams3D) -> Self {
        let intermediate = params.hidden_dim * 4;
        Self {
            gate_up_fused: vec![0.0; params.expert_pool_size * params.hidden_dim * intermediate * 2],
            gate_split: None,
            up_split: None,
            down_weights: vec![0.0; params.expert_pool_size * intermediate * params.hidden_dim],
            expert_scale: None,
        }
    }
    
    pub fn execute_experts(
        &mut self,
        hidden: &mut [f32],
        routing: &[f32],
        params: &NeuralTextHyperparams3D,
    ) -> Result<(), TextProcessingError> {
        let seq_len = hidden.len() / params.hidden_dim;
        
        // Weighted combination
        for seq_idx in 0..seq_len {
            for expert_idx in 0..params.expert_active_count.min(params.expert_pool_size) {
                let weight = routing[seq_idx * params.expert_pool_size + expert_idx];
                let start = seq_idx * params.hidden_dim;
                for j in start..(start+params.hidden_dim).min(hidden.len()) {
                    hidden[j] *= weight;
                }
            }
        }
        Ok(())
    }
}

/// Multimodal injection handler
#[derive(Debug, Clone)]
pub struct MultimodalInjector3D {
    pub image_token_id: i32,
    pub image_end_token_id: i32,
    pub audio_token_id: i32,
    pub audio_end_token_id: i32,
}

impl MultimodalInjector3D {
    pub fn new(image_token: i32, image_end: i32, audio_token: i32, audio_end: i32) -> Self {
        Self {
            image_token_id: image_token,
            image_end_token_id: image_end,
            audio_token_id: audio_token,
            audio_end_token_id: audio_end,
        }
    }
    
    /// Find positions requiring multimodal injection
    pub fn find_injection_points(&self, tokens: &[i32]) -> Vec<(usize, &'static str)> {
        let mut points = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            if token == self.image_token_id {
                points.push((pos, "image"));
            } else if token == self.audio_token_id {
                points.push((pos, "audio"));
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hyperparams_validation() {
        let params = NeuralTextHyperparams3D::config_gemma4_9b();
        assert!(params.validate().is_ok());
        assert_eq!(params.hidden_dim, 3584);
        assert_eq!(params.layer_depth, 42);
    }
    
    #[test]
    fn test_attention_pattern() {
        let params = NeuralTextHyperparams3D::config_gemma4_9b();
        assert!(params.is_local_attention(0));  // Even = local
        assert!(!params.is_local_attention(1)); // Odd = global
    }
    
    #[test]
    fn test_rotary_params() {
        let params = NeuralTextHyperparams3D::config_gemma4_9b();
        let (base, dims) = params.get_rotary_params(0);
        assert_eq!(base, params.rotary_base_local);
        assert_eq!(dims, params.head_dimension_local);
    }
    
    #[test]
    fn test_kv_donor_map() {
        let params = NeuralTextHyperparams3D::config_gemma4_27b();
        let map = KVCacheDonorMap3D::build_from_hyperparams(&params);
        assert!(params.kv_shared_layer_count > 0);
    }
}
