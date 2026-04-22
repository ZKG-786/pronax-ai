use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;

/// DeepSeekOCR text model errors
#[derive(Debug, Clone)]
pub enum TextModelError {
    InvalidDimensions(String),
    ForwardError(String),
    AttentionError(String),
    ConfigurationError(String),
    CacheError(String),
}

impl std::fmt::Display for TextModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::ConfigurationError(s) => write!(f, "Config error: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
        }
    }
}

impl std::error::Error for TextModelError {}

/// 3D-aware text model configuration
#[derive(Debug, Clone, Copy)]
pub struct TextConfig3D {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Number of experts (for MoE)
    pub num_experts: usize,
    /// Number of experts used per token
    pub num_experts_used: usize,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RoPE scaling factor
    pub rope_scale: f32,
    /// RMS norm epsilon
    pub eps: f32,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// 3D spatial depth
    pub spatial_depth: u8,
}

impl TextConfig3D {
    /// Default DeepSeekOCR text configuration
    pub fn default_ocr_text() -> Self {
        Self {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            num_experts: 64,
            num_experts_used: 6,
            rope_base: 10000.0,
            rope_scale: 1.0,
            eps: 1e-6,
            num_layers: 26,
            vocab_size: 129280,
            spatial_depth: 128,
        }
    }
    
    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), TextModelError> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(TextModelError::ConfigurationError(
                format!("hidden_size {} not divisible by num_heads {}",
                    self.hidden_size, self.num_heads)
            ));
        }
        
        if self.num_kv_heads > self.num_heads {
            return Err(TextModelError::ConfigurationError(
                "num_kv_heads cannot exceed num_heads".to_string()
            ));
        }
        
        if self.num_experts_used > self.num_experts {
            return Err(TextModelError::ConfigurationError(
                "num_experts_used cannot exceed num_experts".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for TextConfig3D {
    fn default() -> Self {
        Self::default_ocr_text()
    }
}

/// 3D-aware RoPE (Rotary Position Embedding) for NeoX style
#[derive(Debug, Clone, Copy)]
pub struct RoPE3D {
    /// Base frequency
    pub base: f32,
    /// Scaling factor
    pub scale: f32,
    /// Head dimension
    pub head_dim: usize,
}

impl RoPE3D {
    /// Create new RoPE
    pub fn new(base: f32, scale: f32, head_dim: usize) -> Self {
        Self { base, scale, head_dim }
    }
    
    /// Apply rotary embeddings to states
    pub fn apply(&self, states: &mut [f32], positions: &[i32], num_heads: usize) {
        let head_dim = self.head_dim;
        let seq_len = positions.len();
        
        for head in 0..num_heads {
            for (pos_idx, &pos) in positions.iter().enumerate() {
                let base_offset = (head * seq_len + pos_idx) * head_dim;
                
                for i in (0..head_dim).step_by(2) {
                    let freq = self.base.powf(-(i as f32) / head_dim as f32) * self.scale;
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
pub struct RMSNorm3D {
    /// Scale weights [hidden_size]
    pub weight: Vec<f32>,
    /// Epsilon
    pub eps: f32,
    /// Hidden size
    pub hidden_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl RMSNorm3D {
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

/// 3D-aware attention mechanism with GQA
#[derive(Debug, Clone)]
pub struct TextAttention3D {
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
    /// Head dimension
    pub head_dim: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl TextAttention3D {
    /// Create new attention layer
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        
        Self {
            query_weights: vec![0.0; hidden_size * hidden_size],
            key_weights: vec![0.0; num_kv_heads * head_dim * hidden_size],
            value_weights: vec![0.0; num_kv_heads * head_dim * hidden_size],
            output_weights: vec![0.0; hidden_size * hidden_size],
            num_heads,
            num_kv_heads,
            head_dim,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with RoPE and cache
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
        positions: &[i32],
        cache: &mut CausalKVCache,
        config: &TextConfig3D,
    ) -> Result<Vec<f32>, TextModelError> {
        let hidden_size = config.hidden_size;
        let seq_len = positions.len();
        
        // Project to Q, K, V
        let mut query = vec![0.0; self.num_heads * self.head_dim * seq_len];
        let mut key = vec![0.0; self.num_kv_heads * self.head_dim * seq_len];
        let mut value = vec![0.0; self.num_kv_heads * self.head_dim * seq_len];
        
        // Simplified projection
        for (seq_idx, _) in positions.iter().enumerate() {
            // Query projection
            for head in 0..self.num_heads {
                for d in 0..self.head_dim {
                    let q_idx = (head * seq_len + seq_idx) * self.head_dim + d;
                    if q_idx < query.len() {
                        query[q_idx] = hidden_states[seq_idx * hidden_size + d % hidden_size];
                    }
                }
            }
            
            // Key/Value projection (GQA - fewer KV heads)
            for kv_head in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    let kv_idx = (kv_head * seq_len + seq_idx) * self.head_dim + d;
                    if kv_idx < key.len() {
                        key[kv_idx] = hidden_states[seq_idx * hidden_size + d % hidden_size];
                        value[kv_idx] = hidden_states[seq_idx * hidden_size + d % hidden_size];
                    }
                }
            }
        }
        
        // Apply RoPE
        let rope = RoPE3D::new(config.rope_base, config.rope_scale, self.head_dim);
        rope.apply(&mut query, positions, self.num_heads);
        rope.apply(&mut key, positions, self.num_kv_heads);
        
        // Store in cache
        cache.store_key_values(&key, &value)?;
        
        // Attention computation
        let mut attention_output = vec![0.0; hidden_size * seq_len];
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        
        for seq_idx in 0..seq_len {
            // Get cached keys and values up to current position
            let (cached_key, cached_value) = cache.get_key_values(seq_idx)?;
            
            for head in 0..self.num_heads {
                // Determine which KV head to use (GQA)
                let kv_head = head * self.num_kv_heads / self.num_heads;
                
                // Compute attention scores
                let mut scores = vec![0.0; seq_idx + 1];
                
                for (past_idx, score) in scores.iter_mut().enumerate().take(seq_idx + 1) {
                    let mut dot_product = 0.0;
                    for d in 0..self.head_dim {
                        let q_idx = (head * seq_len + seq_idx) * self.head_dim + d;
                        let k_idx = (kv_head * (seq_idx + 1) + past_idx) * self.head_dim + d;
                        
                        if q_idx < query.len() && k_idx < cached_key.len() {
                            dot_product += query[q_idx] as f64 * cached_key[k_idx] as f64;
                        }
                    }
                    *score = (dot_product * scale) as f32;
                }
                
                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                let softmax_scores: Vec<f32> = scores.iter()
                    .map(|&s| (s - max_score).exp() / exp_sum)
                    .collect();
                
                // Weighted sum of values
                for d in 0..self.head_dim {
                    let mut weighted_sum = 0.0;
                    for (past_idx, &score) in softmax_scores.iter().enumerate().take(seq_idx + 1) {
                        let v_idx = (kv_head * (seq_idx + 1) + past_idx) * self.head_dim + d;
                        if v_idx < cached_value.len() {
                            weighted_sum += score * cached_value[v_idx];
                        }
                    }
                    
                    let out_idx = seq_idx * hidden_size + head * self.head_dim + d;
                    if out_idx < attention_output.len() {
                        attention_output[out_idx] = weighted_sum;
                    }
                }
            }
        }
        
        // Output projection (simplified)
        let mut output = vec![0.0; hidden_size * seq_len];
        for (i, (o, a)) in output.iter_mut().zip(attention_output.iter()).enumerate() {
            *o = *a;
        }
        
        Ok(output)
    }
}

/// 3D-aware dense MLP
#[derive(Debug, Clone)]
pub struct TextMlp3D {
    /// Gate projection weights
    pub gate_weights: Vec<f32>,
    /// Up projection weights
    pub up_weights: Vec<f32>,
    /// Down projection weights
    pub down_weights: Vec<f32>,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl TextMlp3D {
    /// Create new MLP
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_weights: vec![0.0; hidden_size * intermediate_size],
            up_weights: vec![0.0; hidden_size * intermediate_size],
            down_weights: vec![0.0; intermediate_size * hidden_size],
            hidden_size,
            intermediate_size,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with SiLU gate
    pub fn forward(&self, hidden_states: &mut [f32]) {
        let seq_len = hidden_states.len() / self.hidden_size;
        
        for seq_idx in 0..seq_len {
            // Gate and Up projections
            let mut gate = vec![0.0; self.intermediate_size];
            let mut up = vec![0.0; self.intermediate_size];
            
            for i in 0..self.intermediate_size {
                let base_idx = seq_idx * self.hidden_size;
                for j in 0..self.hidden_size {
                    if base_idx + j < hidden_states.len() {
                        gate[i] += hidden_states[base_idx + j] * self.gate_weights[j * self.intermediate_size + i];
                        up[i] += hidden_states[base_idx + j] * self.up_weights[j * self.intermediate_size + i];
                    }
                }
            }
            
            // SiLU activation: x * sigmoid(x)
            for g in &mut gate {
                *g = *g * (1.0 / (1.0 + (-*g).exp()));
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
                if base_idx + j < hidden_states.len() {
                    hidden_states[base_idx + j] = output[j];
                }
            }
        }
    }
}

/// 3D-aware MoE (Mixture of Experts) for text
#[derive(Debug, Clone)]
pub struct TextMoe3D {
    /// Router weights
    pub router_weights: Vec<f32>,
    /// Expert gate weights
    pub expert_gate_weights: Vec<f32>,
    /// Expert up weights
    pub expert_up_weights: Vec<f32>,
    /// Expert down weights
    pub expert_down_weights: Vec<f32>,
    /// Shared experts
    pub shared_experts: TextMlp3D,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts used
    pub num_experts_used: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// 3D spatial positions for experts
    pub expert_spatial_positions: Vec<ConversionCoordinate>,
}

impl TextMoe3D {
    /// Create new MoE layer
    pub fn new(hidden_size: usize, intermediate_size: usize, num_experts: usize, num_experts_used: usize) -> Self {
        // Create 3D spatial positions for each expert
        let expert_spatial_positions: Vec<ConversionCoordinate> = (0..num_experts)
            .map(|i| {
                let x = (i % 8) as u64;
                let y = ((i / 8) % 8) as u16;
                let z = (i / 64) as u8;
                ConversionCoordinate::new(x, y, z, 1.0)
            })
            .collect();
        
        Self {
            router_weights: vec![0.0; num_experts * hidden_size],
            expert_gate_weights: vec![0.0; num_experts * hidden_size * intermediate_size],
            expert_up_weights: vec![0.0; num_experts * hidden_size * intermediate_size],
            expert_down_weights: vec![0.0; num_experts * intermediate_size * hidden_size],
            shared_experts: TextMlp3D::new(hidden_size, intermediate_size),
            num_experts,
            num_experts_used,
            hidden_size,
            intermediate_size,
            expert_spatial_positions,
        }
    }
    
    /// Forward pass with top-K routing
    pub fn forward(&self, hidden_states: &mut [f32], config: &TextConfig3D) {
        let seq_len = hidden_states.len() / self.hidden_size;
        
        for seq_idx in 0..seq_len {
            let base_idx = seq_idx * self.hidden_size;
            
            // Compute router scores
            let mut scores = vec![0.0; self.num_experts];
            for expert in 0..self.num_experts {
                for j in 0..self.hidden_size {
                    if base_idx + j < hidden_states.len() {
                        scores[expert] += hidden_states[base_idx + j] * self.router_weights[j * self.num_experts + expert];
                    }
                }
            }
            
            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            let mut softmax_scores: Vec<f32> = scores.iter()
                .map(|&s| (s - max_score).exp() / exp_sum)
                .collect();
            
            // Select top-K experts
            let mut expert_indices: Vec<usize> = (0..self.num_experts).collect();
            expert_indices.sort_by(|&a, &b| softmax_scores[b].partial_cmp(&softmax_scores[a]).unwrap());
            let topk = &expert_indices[..self.num_experts_used];
            
            // Normalize top-K weights
            let topk_sum: f32 = topk.iter().map(|&i| softmax_scores[i]).sum();
            for &i in topk {
                softmax_scores[i] /= topk_sum;
            }
            
            // Compute expert outputs
            let mut expert_output = vec![0.0; self.hidden_size];
            
            for &expert_idx in topk {
                let weight = softmax_scores[expert_idx];
                
                // Gate and Up for this expert
                let mut gate = vec![0.0; self.intermediate_size];
                let mut up = vec![0.0; self.intermediate_size];
                
                for i in 0..self.intermediate_size {
                    for j in 0..self.hidden_size {
                        if base_idx + j < hidden_states.len() {
                            let gate_idx = expert_idx * self.hidden_size * self.intermediate_size + j * self.intermediate_size + i;
                            let up_idx = expert_idx * self.hidden_size * self.intermediate_size + j * self.intermediate_size + i;
                            
                            if gate_idx < self.expert_gate_weights.len() {
                                gate[i] += hidden_states[base_idx + j] * self.expert_gate_weights[gate_idx];
                            }
                            if up_idx < self.expert_up_weights.len() {
                                up[i] += hidden_states[base_idx + j] * self.expert_up_weights[up_idx];
                            }
                        }
                    }
                }
                
                // SiLU
                for g in &mut gate {
                    *g = *g * (1.0 / (1.0 + (-*g).exp()));
                }
                
                // Multiply
                for i in 0..self.intermediate_size {
                    gate[i] *= up[i];
                }
                
                // Down projection
                for j in 0..self.hidden_size {
                    for i in 0..self.intermediate_size {
                        let down_idx = expert_idx * self.intermediate_size * self.hidden_size + i * self.hidden_size + j;
                        if down_idx < self.expert_down_weights.len() {
                            expert_output[j] += weight * gate[i] * self.expert_down_weights[down_idx];
                        }
                    }
                }
            }
            
            // Shared experts
            let mut shared_input: Vec<f32> = (0..self.hidden_size)
                .map(|j| if base_idx + j < hidden_states.len() { hidden_states[base_idx + j] } else { 0.0 })
                .collect();
            self.shared_experts.forward(&mut shared_input);
            
            // Add shared + routed
            for j in 0..self.hidden_size {
                if base_idx + j < hidden_states.len() {
                    hidden_states[base_idx + j] = expert_output[j] + shared_input[j];
                }
            }
        }
    }
    
    /// Get 3D spatial position of an expert
    pub fn get_expert_spatial(&self, expert_idx: usize) -> Option<&ConversionCoordinate> {
        self.expert_spatial_positions.get(expert_idx)
    }
}

/// Feed-forward enum (dense or MoE)
#[derive(Debug, Clone)]
pub enum TextFeedForward3D {
    Dense(TextMlp3D),
    MoE(TextMoe3D),
}

impl TextFeedForward3D {
    /// Forward pass
    pub fn forward(&self, hidden_states: &mut [f32], config: &TextConfig3D) {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states),
            Self::MoE(moe) => moe.forward(hidden_states, config),
        }
    }
    
    /// Check if MoE
    pub fn is_moe(&self) -> bool {
        matches!(self, Self::MoE(_))
    }
}

/// 3D-aware transformer block
#[derive(Debug, Clone)]
pub struct TextBlock3D {
    /// Attention normalization
    pub attn_norm: RMSNorm3D,
    /// Attention layer
    pub attention: TextAttention3D,
    /// FFN normalization
    pub ffn_norm: RMSNorm3D,
    /// Feed-forward (dense or MoE)
    pub feed_forward: TextFeedForward3D,
    /// Layer index
    pub layer_idx: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl TextBlock3D {
    /// Create new transformer block
    pub fn new(layer_idx: usize, config: &TextConfig3D, use_moe: bool) -> Self {
        let feed_forward = if use_moe && layer_idx >= 1 { // Dense first layer
            TextFeedForward3D::MoE(TextMoe3D::new(
                config.hidden_size,
                4 * config.hidden_size,
                config.num_experts,
                config.num_experts_used,
            ))
        } else {
            TextFeedForward3D::Dense(TextMlp3D::new(
                config.hidden_size,
                4 * config.hidden_size,
            ))
        };
        
        Self {
            attn_norm: RMSNorm3D::new(config.hidden_size, config.eps),
            attention: TextAttention3D::new(config.hidden_size, config.num_heads, config.num_kv_heads),
            ffn_norm: RMSNorm3D::new(config.hidden_size, config.eps),
            feed_forward,
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
        outputs: Option<&[usize]>,
        cache: &mut CausalKVCache,
        config: &TextConfig3D,
    ) -> Result<(), TextModelError> {
        // Attention path with residual
        let residual = hidden_states.to_vec();
        self.attn_norm.normalize(hidden_states);
        
        let mut attn_output = self.attention.forward(hidden_states, positions, cache, config)?;
        
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
            for (h, r) in hidden_states.iter_mut().zip(selected_residual.iter()) {
                *h = *r;
            }
        } else {
            // Add full residual
            for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
                *h = *r;
            }
        }
        
        // Add attention output
        for (h, a) in hidden_states.iter_mut().zip(attn_output.iter()) {
            *h += a;
        }
        
        // FFN path with residual
        let residual = hidden_states.to_vec();
        self.ffn_norm.normalize(hidden_states);
        self.feed_forward.forward(hidden_states, config);
        
        // Add residual
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        Ok(())
    }
}

/// 3D-aware DeepSeekOCR text model
pub struct DeepSeekOcrTextModel3D {
    /// Token embeddings
    pub token_embedding: Vec<f32>,
    /// Transformer blocks
    pub blocks: Vec<TextBlock3D>,
    /// Output normalization
    pub output_norm: RMSNorm3D,
    /// Output projection (LM head)
    pub output: Vec<f32>,
    /// Configuration
    pub config: TextConfig3D,
    /// KV cache
    pub cache: CausalKVCache,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl DeepSeekOcrTextModel3D {
    /// Create new text model
    pub fn new(config: TextConfig3D) -> Result<Self, TextModelError> {
        config.validate()?;
        
        let blocks: Vec<TextBlock3D> = (0..config.num_layers)
            .map(|i| TextBlock3D::new(i, &config, config.num_experts > 1))
            .collect();
        
        let cache = CausalKVCache::new(
            config.num_layers,
            4096, // max context
            config.num_kv_heads,
            config.head_dim(),
        );
        
        Ok(Self {
            token_embedding: vec![0.0; config.vocab_size * config.hidden_size],
            blocks,
            output_norm: RMSNorm3D::new(config.hidden_size, config.eps),
            output: vec![0.0; config.hidden_size * config.vocab_size],
            config,
            cache,
            spatial: SpatialTensorMetadata::new(
                config.vocab_size as u32,
                config.hidden_size as u32,
                config.spatial_depth as u32,
            ),
        })
    }
    
    /// Apply RoPE shift (for cache)
    pub fn shift(&self, key: &mut [f32], shift: &[f32]) {
        // Simplified shift
        for (k, &s) in key.iter_mut().zip(shift.iter()) {
            *k += s;
        }
    }
    
    /// Forward pass
    pub fn forward(
        &mut self,
        token_ids: &[i32],
        positions: &[i32],
        outputs: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>, TextModelError> {
        // Token embedding lookup
        let mut hidden_states: Vec<f32> = token_ids.iter()
            .flat_map(|&token_id| {
                let idx = token_id.max(0) as usize;
                let start = idx * self.config.hidden_size;
                let end = (idx + 1) * self.config.hidden_size;
                
                if end <= self.token_embedding.len() {
                    self.token_embedding[start..end].to_vec()
                } else {
                    vec![0.0; self.config.hidden_size]
                }
            })
            .collect();
        
        // Process through blocks
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            self.cache.set_layer(layer_idx);
            block.forward(&mut hidden_states, positions, outputs, &mut self.cache, &self.config)?;
        }
        
        // Output normalization
        let seq_len = hidden_states.len() / self.config.hidden_size;
        for seq_idx in 0..seq_len {
            let start = seq_idx * self.config.hidden_size;
            let end = start + self.config.hidden_size;
            self.output_norm.normalize(&mut hidden_states[start..end]);
        }
        
        // Output projection (simplified)
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for seq_idx in 0..seq_len {
            let mut seq_logits = vec![0.0; self.config.vocab_size];
            
            let hidden_start = seq_idx * self.config.hidden_size;
            for vocab_idx in 0..self.config.vocab_size {
                let output_start = vocab_idx * self.config.hidden_size;
                for h in 0..self.config.hidden_size {
                    if hidden_start + h < hidden_states.len() && output_start + h < self.output.len() {
                        seq_logits[vocab_idx] += hidden_states[hidden_start + h] * self.output[output_start + h];
                    }
                }
            }
            
            logits.push(seq_logits);
        }
        
        Ok(logits)
    }
    
    /// Get model info
    pub fn model_info(&self) -> TextModelInfo {
        TextModelInfo {
            name: "DeepSeekOCR-Text-3D".to_string(),
            total_params: self.estimate_parameters(),
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            num_kv_heads: self.config.num_kv_heads,
            vocab_size: self.config.vocab_size,
            use_moe: self.config.num_experts > 1,
            num_experts: self.config.num_experts,
        }
    }
    
    /// Estimate total parameters
    fn estimate_parameters(&self) -> usize {
        let c = &self.config;
        
        // Embeddings
        let embedding = c.vocab_size * c.hidden_size;
        
        // Per layer
        let attn_params = c.hidden_size * c.hidden_size * 3 // QKV
            + c.hidden_size * c.hidden_size; // output
        
        let ffn_params = if c.num_experts > 1 {
            // MoE
            c.num_experts * 3 * c.hidden_size * 4 * c.hidden_size + c.hidden_size * c.num_experts
        } else {
            // Dense
            3 * c.hidden_size * 4 * c.hidden_size
        };
        
        let layer_params = attn_params + ffn_params + 2 * c.hidden_size; // norms
        
        // Total
        embedding + c.num_layers * layer_params + embedding // output
    }
}

/// Text model information
#[derive(Debug, Clone)]
pub struct TextModelInfo {
    pub name: String,
    pub total_params: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub use_moe: bool,
    pub num_experts: usize,
}

/// Utility functions
pub mod text_utils {
    use super::*;
    
    /// Compute memory requirement for text model
    pub fn estimate_memory(config: &TextConfig3D) -> u64 {
        // Parameters
        let embedding = (config.vocab_size * config.hidden_size * 4) as u64;
        
        let attn_per_layer = (config.hidden_size * config.hidden_size * 4 * 4) as u64;
        let ffn_per_layer = if config.num_experts > 1 {
            (config.num_experts * config.hidden_size * 4 * config.hidden_size * 3 * 4) as u64
        } else {
            (config.hidden_size * 4 * config.hidden_size * 3 * 4) as u64
        };
        
        let layers = config.num_layers as u64 * (attn_per_layer + ffn_per_layer);
        
        // KV cache
        let kv_cache = (config.num_layers * 4096 * config.num_kv_heads * config.head_dim() * 2 * 4) as u64;
        
        embedding + layers + kv_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = TextConfig3D::default();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim(), 128);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_rope() {
        let rope = RoPE3D::new(10000.0, 1.0, 128);
        let mut states = vec![1.0; 128 * 4]; // 4 heads, 128 head_dim
        let positions = vec![0, 1, 2, 3];
        
        rope.apply(&mut states, &positions, 4);
        
        // States should have been rotated
        assert!(states.iter().any(|&x| x != 1.0));
    }
    
    #[test]
    fn test_attention() {
        let config = TextConfig3D::default();
        let attn = TextAttention3D::new(4096, 32, 32);
        
        assert_eq!(attn.num_heads, 32);
        assert_eq!(attn.head_dim, 128);
    }
    
    #[test]
    fn test_mlp() {
        let mlp = TextMlp3D::new(4096, 16384);
        let mut data = vec![1.0; 4096];
        
        mlp.forward(&mut data);
        
        assert_eq!(data.len(), 4096);
        assert!(data.iter().any(|&x| x != 1.0));
    }
    
    #[test]
    fn test_moe() {
        let moe = TextMoe3D::new(4096, 16384, 64, 6);
        
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.expert_spatial_positions.len(), 64);
        
        let mut data = vec![1.0; 4096];
        let config = TextConfig3D::default();
        moe.forward(&mut data, &config);
        
        assert_eq!(data.len(), 4096);
    }
    
    #[test]
    fn test_block() {
        let config = TextConfig3D::default();
        let block = TextBlock3D::new(0, &config, true);
        
        assert!(matches!(block.feed_forward, TextFeedForward3D::Dense(_)));
        
        let moe_block = TextBlock3D::new(1, &config, true);
        assert!(matches!(moe_block.feed_forward, TextFeedForward3D::MoE(_)));
    }
    
    #[test]
    fn test_model_creation() {
        let config = TextConfig3D::default();
        let model = DeepSeekOcrTextModel3D::new(config);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert!(info.total_params > 1_000_000_000); // > 1B params
        assert!(info.use_moe);
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = TextConfig3D::default();
        let mem = text_utils::estimate_memory(&config);
        
        assert!(mem > 0);
        assert!(mem > 2_000_000_000); // > 2GB
    }
}
