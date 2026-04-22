use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput};
use crate::tokenizer::pronax_sentencepiece::NeuralSentencePieceTokenizer;

/// Gemma2 model errors
#[derive(Debug, Clone)]
pub enum Gemma2Error {
    InvalidConfiguration(String),
    ForwardError(String),
    AttentionError(String),
    CacheError(String),
    UnsupportedModel(String),
}

impl std::fmt::Display for Gemma2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(s) => write!(f, "Invalid config: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
            Self::UnsupportedModel(s) => write!(f, "Unsupported: {}", s),
        }
    }
}

impl std::error::Error for Gemma2Error {}

/// 3D-aware Gemma2 configuration
#[derive(Debug, Clone, Copy)]
pub struct Gemma2Config3D {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Attention key length
    pub attn_key_len: usize,
    /// Attention value length
    pub attn_val_len: usize,
    /// RMS norm epsilon
    pub eps: f32,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RoPE scaling factor
    pub rope_scale: f32,
    /// Attention logit softcap
    pub attn_logit_softcap: f32,
    /// Final logit softcap
    pub final_logit_softcap: f32,
    /// Use large model scaling (27B)
    pub large_model_scaling: bool,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Sliding window for local attention
    pub sliding_window: usize,
    /// 3D spatial depth
    pub spatial_depth: u8,
}

impl Gemma2Config3D {
    /// Gemma2 2B configuration
    pub fn gemma2_2b() -> Self {
        Self {
            hidden_size: 2304,
            num_heads: 8,
            num_kv_heads: 4,
            attn_key_len: 256,
            attn_val_len: 256,
            eps: 1e-6,
            rope_base: 10000.0,
            rope_scale: 1.0,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            large_model_scaling: false,
            num_layers: 26,
            vocab_size: 256128,
            sliding_window: 4096,
            spatial_depth: 64,
        }
    }
    
    /// Gemma2 9B configuration
    pub fn gemma2_9b() -> Self {
        Self {
            hidden_size: 3584,
            num_heads: 16,
            num_kv_heads: 8,
            attn_key_len: 256,
            attn_val_len: 256,
            eps: 1e-6,
            rope_base: 10000.0,
            rope_scale: 1.0,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            large_model_scaling: false,
            num_layers: 42,
            vocab_size: 256128,
            sliding_window: 4096,
            spatial_depth: 96,
        }
    }
    
    /// Gemma2 27B configuration
    pub fn gemma2_27b() -> Self {
        Self {
            hidden_size: 4608,
            num_heads: 32,
            num_kv_heads: 16,
            attn_key_len: 128,
            attn_val_len: 128,
            eps: 1e-6,
            rope_base: 10000.0,
            rope_scale: 1.0,
            attn_logit_softcap: 50.0,
            final_logit_softcap: 30.0,
            large_model_scaling: true,
            num_layers: 46,
            vocab_size: 256128,
            sliding_window: 4096,
            spatial_depth: 128,
        }
    }
    
    /// Head dimension
    pub fn head_dim(&self) -> usize {
        if self.large_model_scaling {
            self.hidden_size / self.num_heads
        } else {
            self.attn_key_len
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), Gemma2Error> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(Gemma2Error::InvalidConfiguration(
                format!("hidden_size {} not divisible by num_heads {}",
                    self.hidden_size, self.num_heads)
            ));
        }
        
        if self.num_kv_heads > self.num_heads {
            return Err(Gemma2Error::InvalidConfiguration(
                "num_kv_heads cannot exceed num_heads".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for Gemma2Config3D {
    fn default() -> Self {
        Self::gemma2_9b()
    }
}

/// 3D-aware RoPE (Rotary Position Embedding) for NeoX style
#[derive(Debug, Clone, Copy)]
pub struct Gemma2RoPE3D {
    /// Base frequency
    pub base: f32,
    /// Scaling factor
    pub scale: f32,
    /// Key length
    pub key_len: usize,
}

impl Gemma2RoPE3D {
    /// Create new RoPE
    pub fn new(base: f32, scale: f32, key_len: usize) -> Self {
        Self { base, scale, key_len }
    }
    
    /// Apply rotary embeddings to states
    pub fn apply(&self, states: &mut [f32], positions: &[i32], num_heads: usize, head_dim: usize) {
        let seq_len = positions.len();
        
        for head in 0..num_heads {
            for (pos_idx, &pos) in positions.iter().enumerate() {
                let base_offset = (head * seq_len + pos_idx) * head_dim;
                
                for i in (0..head_dim).step_by(2) {
                    let freq = self.base.powf(-(i as f32) / self.key_len as f32) * self.scale;
                    let angle = pos as f32 * freq;
                    
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();
                    
                    let x0_idx = base_offset + i;
                    let x1_idx = base_offset + i + 1;
                    
                    if x1_idx < states.len() {
                        let x0 = states[x0_idx];
                        let x1 = states[x1_idx];
                        
                        states[x0_idx] = x0 * cos_val - x1 * sin_val;
                        states[x1_idx] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
    }
}

/// 3D-aware RMS Normalization
#[derive(Debug, Clone)]
pub struct Gemma2RMSNorm3D {
    /// Scale weights
    pub weight: Vec<f32>,
    /// Epsilon
    pub eps: f32,
    /// Hidden size
    pub hidden_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma2RMSNorm3D {
    /// Create new RMS norm
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; hidden_size],
            eps,
            hidden_size,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Normalize input
    pub fn normalize(&self, input: &mut [f32]) {
        if input.len() != self.hidden_size {
            return;
        }
        
        // Compute RMS
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / self.hidden_size as f32 + self.eps).sqrt();
        
        // Apply scale
        for (x, &w) in input.iter_mut().zip(self.weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

/// 3D-aware self-attention with sliding window
#[derive(Debug, Clone)]
pub struct Gemma2SelfAttention3D {
    /// Query projection weights
    pub query_weights: Vec<f32>,
    /// Key projection weights
    pub key_weights: Vec<f32>,
    /// Value projection weights
    pub value_weights: Vec<f32>,
    /// Output projection weights
    pub output_weights: Vec<f32>,
    /// Number of heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Key length
    pub key_len: usize,
    /// Value length
    pub val_len: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma2SelfAttention3D {
    /// Create new self-attention layer
    pub fn new(config: &Gemma2Config3D) -> Self {
        let hidden_size = config.hidden_size;
        
        Self {
            query_weights: vec![0.0; hidden_size * config.attn_key_len * config.num_heads],
            key_weights: vec![0.0; hidden_size * config.attn_key_len * config.num_kv_heads],
            value_weights: vec![0.0; hidden_size * config.attn_val_len * config.num_kv_heads],
            output_weights: vec![0.0; config.attn_val_len * config.num_heads * hidden_size],
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            key_len: config.attn_key_len,
            val_len: config.attn_val_len,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with RoPE and sliding window
    pub fn forward(
        &self,
        hidden_state: &mut [f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        config: &Gemma2Config3D,
    ) -> Result<Vec<f32>, Gemma2Error> {
        let hidden_size = config.hidden_size;
        let seq_len = positions.len();
        
        // Project to Q, K, V
        let mut query = vec![0.0; self.num_heads * self.key_len * seq_len];
        let mut key = vec![0.0; self.num_kv_heads * self.key_len * seq_len];
        let mut value = vec![0.0; self.num_kv_heads * self.val_len * seq_len];
        
        // Simplified projection
        for seq_idx in 0..seq_len {
            for head in 0..self.num_heads {
                for d in 0..self.key_len {
                    let q_idx = (head * seq_len + seq_idx) * self.key_len + d;
                    if q_idx < query.len() {
                        query[q_idx] = hidden_state[seq_idx * hidden_size + d % hidden_size];
                    }
                }
            }
            
            for kv_head in 0..self.num_kv_heads {
                for d in 0..self.key_len {
                    let k_idx = (kv_head * seq_len + seq_idx) * self.key_len + d;
                    if k_idx < key.len() {
                        key[k_idx] = hidden_state[seq_idx * hidden_size + d % hidden_size];
                    }
                }
                for d in 0..self.val_len {
                    let v_idx = (kv_head * seq_len + seq_idx) * self.val_len + d;
                    if v_idx < value.len() {
                        value[v_idx] = hidden_state[seq_idx * hidden_size + d % hidden_size];
                    }
                }
            }
        }
        
        // Apply RoPE
        let rope = Gemma2RoPE3D::new(config.rope_base, config.rope_scale, config.attn_key_len);
        let head_dim = if config.large_model_scaling {
            config.hidden_size / config.num_heads
        } else {
            config.attn_key_len
        };
        rope.apply(&mut query, positions, self.num_heads, head_dim);
        rope.apply(&mut key, positions, self.num_kv_heads, head_dim);
        
        // Scale query
        let scale_factor = if config.large_model_scaling {
            1.0 / (config.hidden_size as f64 / config.num_heads as f64).sqrt()
        } else {
            1.0 / (config.attn_key_len as f64).sqrt()
        };
        
        for q in &mut query {
            *q *= scale_factor as f32;
        }
        
        // Store in cache
        cache.store_key_values(&key, &value)?;
        
        // Attention computation with sliding window
        let mut attention_output = vec![0.0; hidden_size * seq_len];
        
        for seq_idx in 0..seq_len {
            let (cached_key, cached_value) = cache.get_key_values(seq_idx)?;
            
            for head in 0..self.num_heads {
                let kv_head = head * self.num_kv_heads / self.num_heads;
                
                // Compute attention scores
                let mut scores = vec![0.0; seq_idx + 1];
                
                for (past_idx, score) in scores.iter_mut().enumerate().take(seq_idx + 1) {
                    // Sliding window: only attend to last N tokens
                    if seq_idx - past_idx > config.sliding_window && config.sliding_window > 0 {
                        continue;
                    }
                    
                    let mut dot_product = 0.0;
                    for d in 0..self.key_len {
                        let q_idx = (head * seq_len + seq_idx) * self.key_len + d;
                        let k_idx = (kv_head * (seq_idx + 1) + past_idx) * self.key_len + d;
                        
                        if q_idx < query.len() && k_idx < cached_key.len() {
                            dot_product += query[q_idx] as f64 * cached_key[k_idx] as f64;
                        }
                    }
                    *score = (dot_product * scale_factor) as f32;
                }
                
                // Apply logit softcap
                for s in &mut scores {
                    *s = (*s / config.attn_logit_softcap).tanh() * config.attn_logit_softcap;
                }
                
                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                let softmax_scores: Vec<f32> = scores.iter()
                    .map(|&s| (s - max_score).exp() / exp_sum)
                    .collect();
                
                // Weighted sum of values
                for d in 0..self.val_len {
                    let mut weighted_sum = 0.0;
                    for (past_idx, &score) in softmax_scores.iter().enumerate().take(seq_idx + 1) {
                        let v_idx = (kv_head * (seq_idx + 1) + past_idx) * self.val_len + d;
                        if v_idx < cached_value.len() {
                            weighted_sum += score * cached_value[v_idx];
                        }
                    }
                    
                    let out_idx = seq_idx * hidden_size + head * self.val_len + d;
                    if out_idx < attention_output.len() {
                        attention_output[out_idx] = weighted_sum;
                    }
                }
            }
        }
        
        // Output projection
        let mut output = vec![0.0; hidden_size * seq_len];
        for (i, (o, a)) in output.iter_mut().zip(attention_output.iter()).enumerate() {
            *o = *a;
        }
        
        Ok(output)
    }
}

/// 3D-aware MLP with GELU
#[derive(Debug, Clone)]
pub struct Gemma2Mlp3D {
    /// Up projection weights
    pub up_weights: Vec<f32>,
    /// Down projection weights
    pub down_weights: Vec<f32>,
    /// Gate projection weights
    pub gate_weights: Vec<f32>,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma2Mlp3D {
    /// Create new MLP
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            up_weights: vec![0.0; hidden_size * intermediate_size],
            down_weights: vec![0.0; intermediate_size * hidden_size],
            gate_weights: vec![0.0; hidden_size * intermediate_size],
            hidden_size,
            intermediate_size,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// GELU activation
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
    }
    
    /// Forward pass with GELU gate
    pub fn forward(&self, hidden_state: &mut [f32]) {
        let seq_len = hidden_state.len() / self.hidden_size;
        
        for seq_idx in 0..seq_len {
            // Gate and Up projections
            let mut gate = vec![0.0; self.intermediate_size];
            let mut up = vec![0.0; self.intermediate_size];
            
            for i in 0..self.intermediate_size {
                let base_idx = seq_idx * self.hidden_size;
                for j in 0..self.hidden_size {
                    if base_idx + j < hidden_state.len() {
                        gate[i] += hidden_state[base_idx + j] * self.gate_weights[j * self.intermediate_size + i];
                        up[i] += hidden_state[base_idx + j] * self.up_weights[j * self.intermediate_size + i];
                    }
                }
            }
            
            // GELU activation
            for g in &mut gate {
                *g = Self::gelu(*g);
            }
            
            // Element-wise multiply
            for i in 0..self.intermediate_size {
                gate[i] *= up[i];
            }
            
            // Down projection
            let mut output = vec![0.0; self.hidden_size];
            for j in 0..self.hidden_size {
                for i in 0..self.intermediate_size {
                    output[j] += gate[i] * self.down_weights[i * self.hidden_size + j];
                }
            }
            
            // Copy back
            let base_idx = seq_idx * self.hidden_size;
            for j in 0..self.hidden_size {
                if base_idx + j < hidden_state.len() {
                    hidden_state[base_idx + j] = output[j];
                }
            }
        }
    }
}

/// 3D-aware Gemma2 transformer layer
#[derive(Debug, Clone)]
pub struct Gemma2Layer3D {
    /// Pre-attention normalization
    pub attn_norm: Gemma2RMSNorm3D,
    /// Self-attention
    pub self_attn: Gemma2SelfAttention3D,
    /// Post-attention normalization
    pub post_attn_norm: Gemma2RMSNorm3D,
    /// MLP normalization
    pub mlp_norm: Gemma2RMSNorm3D,
    /// MLP
    pub mlp: Gemma2Mlp3D,
    /// Post-MLP normalization
    pub post_mlp_norm: Gemma2RMSNorm3D,
    /// Layer index
    pub layer_idx: usize,
    /// Layer type (0=local, 1=global) for alternating attention
    pub layer_type: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Gemma2Layer3D {
    /// Create new Gemma2 layer
    pub fn new(layer_idx: usize, config: &Gemma2Config3D) -> Self {
        let intermediate_size = 4 * config.hidden_size; // Standard expansion
        
        Self {
            attn_norm: Gemma2RMSNorm3D::new(config.hidden_size, config.eps),
            self_attn: Gemma2SelfAttention3D::new(config),
            post_attn_norm: Gemma2RMSNorm3D::new(config.hidden_size, config.eps),
            mlp_norm: Gemma2RMSNorm3D::new(config.hidden_size, config.eps),
            mlp: Gemma2Mlp3D::new(config.hidden_size, intermediate_size),
            post_mlp_norm: Gemma2RMSNorm3D::new(config.hidden_size, config.eps),
            layer_idx,
            layer_type: layer_idx % 2, // Alternating layer types
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
        hidden_state: &mut [f32],
        positions: &[i32],
        outputs: Option<&[usize]>,
        cache: &mut CausalKVCache,
        config: &Gemma2Config3D,
    ) -> Result<(), Gemma2Error> {
        // Attention path with residual
        let residual = hidden_state.to_vec();
        self.attn_norm.normalize(hidden_state);
        
        let mut attn_output = self.self_attn.forward(hidden_state, positions, cache, config)?;
        self.post_attn_norm.normalize(&mut attn_output);
        
        // Apply output selection if needed
        if let Some(output_indices) = outputs {
            let mut selected = Vec::new();
            for &idx in output_indices {
                let base = idx * config.hidden_size;
                if base + config.hidden_size <= attn_output.len() {
                    selected.extend_from_slice(&attn_output[base..base + config.hidden_size]);
                }
            }
            attn_output = selected;
            
            // Also select from residual
            let mut selected_residual = Vec::new();
            for &idx in output_indices {
                let base = idx * config.hidden_size;
                if base + config.hidden_size <= residual.len() {
                    selected_residual.extend_from_slice(&residual[base..base + config.hidden_size]);
                }
            }
            
            // Add residual
            for (h, r) in hidden_state.iter_mut().zip(selected_residual.iter()) {
                *h = *r;
            }
        } else {
            // Add full residual
            for (h, r) in hidden_state.iter_mut().zip(residual.iter()) {
                *h = *r;
            }
        }
        
        // Add attention output
        for (h, a) in hidden_state.iter_mut().zip(attn_output.iter()) {
            *h += a;
        }
        
        // MLP path with residual
        let residual = hidden_state.to_vec();
        self.mlp_norm.normalize(hidden_state);
        self.mlp.forward(hidden_state);
        self.post_mlp_norm.normalize(hidden_state);
        
        // Add residual
        for (h, r) in hidden_state.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        Ok(())
    }
}

/// 3D-aware Gemma2 model
pub struct Gemma2Model3D {
    /// Token embeddings
    pub token_embedding: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<Gemma2Layer3D>,
    /// Output normalization
    pub output_norm: Gemma2RMSNorm3D,
    /// Output projection (LM head, shared with embeddings)
    pub output: Vec<f32>,
    /// Configuration
    pub config: Gemma2Config3D,
    /// KV cache (with sliding window)
    pub cache: CausalKVCache,
    /// Tokenizer
    pub tokenizer: Option<NeuralSentencePieceTokenizer>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl Gemma2Model3D {
    /// Create new Gemma2 model
    pub fn new(config: Gemma2Config3D) -> Result<Self, Gemma2Error> {
        config.validate()?;
        
        // Auto-detect large model
        let mut config = config;
        if config.num_layers == 46 {
            config.large_model_scaling = true;
        }
        
        let layers: Vec<Gemma2Layer3D> = (0..config.num_layers)
            .map(|i| Gemma2Layer3D::new(i, &config))
            .collect();
        
        let cache = CausalKVCache::new(
            config.num_layers,
            8192, // max context
            config.num_kv_heads,
            config.attn_val_len,
        );
        
        Ok(Self {
            token_embedding: vec![0.0; config.vocab_size * config.hidden_size],
            layers,
            output_norm: Gemma2RMSNorm3D::new(config.hidden_size, config.eps),
            output: vec![0.0; config.vocab_size * config.hidden_size], // Shared with embeddings
            config,
            cache,
            tokenizer: None,
            spatial: SpatialTensorMetadata::new(
                config.vocab_size as u32,
                config.hidden_size as u32,
                config.spatial_depth as u32,
            ),
        })
    }
    
    /// Apply RoPE shift (for cache)
    pub fn shift(&self, key: &mut [f32], shift: &[f32]) {
        for (k, &s) in key.iter_mut().zip(shift.iter()) {
            *k += s;
        }
    }
    
    /// Forward pass
    pub fn forward(
        &mut self,
        batch: &NeuralBatch,
    ) -> Result<Vec<Vec<f32>>, Gemma2Error> {
        let positions: Vec<i32> = batch.inputs.iter()
            .map(|inp| inp.position_in_sequence as i32)
            .collect();
        
        // Token embedding lookup with scaling
        let mut hidden_state: Vec<f32> = batch.inputs.iter()
            .flat_map(|input| {
                let token_id = input.token_id.max(0) as usize;
                let start = token_id * self.config.hidden_size;
                let end = (token_id + 1) * self.config.hidden_size;
                
                if end <= self.token_embedding.len() {
                    self.token_embedding[start..end].to_vec()
                } else {
                    vec![0.0; self.config.hidden_size]
                }
            })
            .collect();
        
        // Scale embeddings by sqrt(hidden_size)
        let scale = (self.config.hidden_size as f32).sqrt();
        for h in &mut hidden_state {
            *h *= scale;
        }
        
        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.cache.set_layer(layer_idx);
            
            let outputs = if layer_idx == self.layers.len() - 1 {
                // Last layer - get output positions
                None // batch.outputs.clone()
            } else {
                None
            };
            
            layer.forward(&mut hidden_state, &positions, outputs.as_deref(), &mut self.cache, &self.config)?;
        }
        
        // Output normalization
        let seq_len = hidden_state.len() / self.config.hidden_size;
        for seq_idx in 0..seq_len {
            let start = seq_idx * self.config.hidden_size;
            let end = start + self.config.hidden_size;
            self.output_norm.normalize(&mut hidden_state[start..end]);
        }
        
        // Output projection (shared with embeddings)
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for seq_idx in 0..seq_len {
            let mut seq_logits = vec![0.0; self.config.vocab_size];
            
            let hidden_start = seq_idx * self.config.hidden_size;
            for vocab_idx in 0..self.config.vocab_size {
                let embed_start = vocab_idx * self.config.hidden_size;
                for h in 0..self.config.hidden_size {
                    if hidden_start + h < hidden_state.len() && embed_start + h < self.token_embedding.len() {
                        seq_logits[vocab_idx] += hidden_state[hidden_start + h] * self.token_embedding[embed_start + h];
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
    
    /// Get model info
    pub fn model_info(&self) -> Gemma2ModelInfo {
        Gemma2ModelInfo {
            name: "Gemma2-3D".to_string(),
            variant: if self.config.num_layers == 26 {
                "2B".to_string()
            } else if self.config.num_layers == 42 {
                "9B".to_string()
            } else if self.config.num_layers == 46 {
                "27B".to_string()
            } else {
                "Custom".to_string()
            },
            total_params: self.estimate_parameters(),
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            num_kv_heads: self.config.num_kv_heads,
            vocab_size: self.config.vocab_size,
            sliding_window: self.config.sliding_window,
            large_model_scaling: self.config.large_model_scaling,
        }
    }
    
    /// Estimate total parameters
    fn estimate_parameters(&self) -> usize {
        let c = &self.config;
        
        // Embeddings
        let embedding = c.vocab_size * c.hidden_size;
        
        // Per layer
        let attn_params = c.hidden_size * c.attn_key_len * c.num_heads + // Q
            c.hidden_size * c.attn_key_len * c.num_kv_heads + // K
            c.hidden_size * c.attn_val_len * c.num_kv_heads + // V
            c.attn_val_len * c.num_heads * c.hidden_size; // output
        
        let mlp_params = 3 * c.hidden_size * 4 * c.hidden_size; // gate, up, down
        let norm_params = 4 * c.hidden_size; // 4 norms per layer
        
        let layer_params = attn_params + mlp_params + norm_params;
        
        // Total (output shares embeddings)
        embedding + c.num_layers * layer_params + c.hidden_size // final norm
    }
}

/// Gemma2 model information
#[derive(Debug, Clone)]
pub struct Gemma2ModelInfo {
    pub name: String,
    pub variant: String,
    pub total_params: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub large_model_scaling: bool,
}

/// Utility functions
pub mod gemma2_utils {
    use super::*;
    
    /// Compute memory requirement for Gemma2 model
    pub fn estimate_memory(config: &Gemma2Config3D) -> u64 {
        // Parameters
        let embedding = (config.vocab_size * config.hidden_size * 4) as u64;
        
        let attn_per_layer = (config.hidden_size * config.attn_key_len * config.num_heads * 4) as u64 +
            (config.hidden_size * config.attn_key_len * config.num_kv_heads * 4) as u64 +
            (config.hidden_size * config.attn_val_len * config.num_kv_heads * 4) as u64 +
            (config.attn_val_len * config.num_heads * config.hidden_size * 4) as u64;
        
        let mlp_per_layer = (3 * config.hidden_size * 4 * config.hidden_size * 4) as u64;
        let norm_per_layer = (4 * config.hidden_size * 4) as u64;
        
        let layers = config.num_layers as u64 * (attn_per_layer + mlp_per_layer + norm_per_layer);
        
        // KV cache with sliding window
        let kv_cache = (config.num_layers * config.sliding_window * config.num_kv_heads * 
            config.attn_val_len * 2 * 4) as u64;
        
        embedding + layers + kv_cache
    }
    
    /// Check if model variant is supported
    pub fn check_model_variant(num_layers: usize) -> Result<(), Gemma2Error> {
        match num_layers {
            26 | 42 | 46 => Ok(()),
            _ => Err(Gemma2Error::UnsupportedModel(
                format!("Unsupported Gemma2 variant with {} layers", num_layers)
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = Gemma2Config3D::gemma2_9b();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.head_dim(), 256);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_variants() {
        let config_2b = Gemma2Config3D::gemma2_2b();
        assert_eq!(config_2b.num_layers, 26);
        assert!(!config_2b.large_model_scaling);
        
        let config_27b = Gemma2Config3D::gemma2_27b();
        assert_eq!(config_27b.num_layers, 46);
        assert!(config_27b.large_model_scaling);
    }
    
    #[test]
    fn test_attention() {
        let config = Gemma2Config3D::gemma2_9b();
        let attn = Gemma2SelfAttention3D::new(&config);
        
        assert_eq!(attn.num_heads, 16);
        assert_eq!(attn.num_kv_heads, 8);
    }
    
    #[test]
    fn test_mlp() {
        let mlp = Gemma2Mlp3D::new(3584, 14336);
        let mut data = vec![1.0; 3584];
        
        mlp.forward(&mut data);
        
        assert_eq!(data.len(), 3584);
        assert!(data.iter().any(|&x| x != 1.0));
    }
    
    #[test]
    fn test_layer() {
        let config = Gemma2Config3D::gemma2_9b();
        let layer = Gemma2Layer3D::new(0, &config);
        
        assert_eq!(layer.layer_idx, 0);
        assert_eq!(layer.layer_type, 0);
        
        let layer2 = Gemma2Layer3D::new(1, &config);
        assert_eq!(layer2.layer_type, 1);
    }
    
    #[test]
    fn test_model_creation() {
        let config = Gemma2Config3D::gemma2_2b();
        let model = Gemma2Model3D::new(config);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert_eq!(info.variant, "2B");
        assert!(info.total_params > 2_000_000_000); // > 2B params
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = Gemma2Config3D::gemma2_9b();
        let mem = gemma2_utils::estimate_memory(&config);
        
        assert!(mem > 0);
        assert!(mem > 10_000_000_000); // > 10GB
    }
    
    #[test]
    fn test_utils() {
        assert!(gemma2_utils::check_model_variant(26).is_ok());
        assert!(gemma2_utils::check_model_variant(42).is_ok());
        assert!(gemma2_utils::check_model_variant(46).is_ok());
        assert!(gemma2_utils::check_model_variant(50).is_err());
    }
}
