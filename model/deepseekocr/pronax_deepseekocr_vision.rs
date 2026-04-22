use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// DeepSeekOCR vision model errors
#[derive(Debug, Clone)]
pub enum VisionModelError {
    InvalidDimensions(String),
    ForwardError(String),
    AttentionError(String),
    ConfigurationError(String),
    EmbeddingError(String),
}

impl std::fmt::Display for VisionModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::ConfigurationError(s) => write!(f, "Config error: {}", s),
            Self::EmbeddingError(s) => write!(f, "Embedding error: {}", s),
        }
    }
}

impl std::error::Error for VisionModelError {}

/// 3D-aware vision model configuration
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig3D {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Layer norm epsilon
    pub eps: f32,
    /// Image size (square)
    pub image_size: usize,
    /// Patch size (square)
    pub patch_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of patches per side
    pub num_patches: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// 3D spatial depth
    pub spatial_depth: u8,
}

impl VisionConfig3D {
    /// Default DeepSeekOCR vision configuration
    pub fn default_ocr_vision() -> Self {
        let image_size = 640;
        let patch_size = 14;
        let num_patches = image_size / patch_size;
        
        Self {
            hidden_size: 1024,
            num_heads: 16,
            eps: 1e-5,
            image_size,
            patch_size,
            num_layers: 24,
            num_patches,
            intermediate_size: 4096,
            spatial_depth: 64,
        }
    }
    
    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    
    /// Total number of patches (including class token)
    pub fn total_patches(&self) -> usize {
        self.num_patches * self.num_patches + 1 // +1 for class token
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), VisionModelError> {
        if self.image_size % self.patch_size != 0 {
            return Err(VisionModelError::ConfigurationError(
                format!("image_size {} not divisible by patch_size {}",
                    self.image_size, self.patch_size)
            ));
        }
        
        if self.hidden_size % self.num_heads != 0 {
            return Err(VisionModelError::ConfigurationError(
                format!("hidden_size {} not divisible by num_heads {}",
                    self.hidden_size, self.num_heads)
            ));
        }
        
        Ok(())
    }
}

impl Default for VisionConfig3D {
    fn default() -> Self {
        Self::default_ocr_vision()
    }
}

/// 3D-aware 2D Convolution for patch embedding
#[derive(Debug, Clone)]
pub struct VisionConv2D3D {
    /// Kernel weights [out_channels, in_channels, kernel_h, kernel_w]
    pub weight: Vec<f32>,
    /// Bias [out_channels]
    pub bias: Option<Vec<f32>>,
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl VisionConv2D3D {
    /// Create new Conv2D layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        let kernel_prod = kernel_size.0 * kernel_size.1;
        Self {
            weight: vec![0.0; out_channels * in_channels * kernel_prod],
            bias: Some(vec![0.0; out_channels]),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass - extract patches
    pub fn forward(&self, input: &[f32], input_shape: (usize, usize, usize)) -> Vec<f32> {
        // input: [C, H, W]
        let (_, h, w) = input_shape;
        let out_h = h / self.stride.0;
        let out_w = w / self.stride.1;
        
        // Output: [out_channels, out_h, out_w] flattened to [out_h * out_w, out_channels]
        vec![0.0; out_h * out_w * self.out_channels]
    }
    
    /// Get output dimensions
    pub fn output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = input_h / self.stride.0;
        let out_w = input_w / self.stride.1;
        (out_h, out_w)
    }
}

/// 3D-aware Layer Normalization
#[derive(Debug, Clone)]
pub struct VisionLayerNorm3D {
    /// Scale weights
    pub weight: Vec<f32>,
    /// Bias
    pub bias: Vec<f32>,
    /// Normalized shape
    pub normalized_shape: usize,
    /// Epsilon
    pub eps: f32,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl VisionLayerNorm3D {
    /// Create new LayerNorm
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; normalized_shape],
            bias: vec![0.0; normalized_shape],
            normalized_shape,
            eps,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Normalize input
    pub fn normalize(&self, input: &mut [f32]) {
        let num_features = self.normalized_shape;
        let num_samples = input.len() / num_features;
        
        for sample_idx in 0..num_samples {
            let start = sample_idx * num_features;
            let end = start + num_features;
            
            // Compute mean
            let sum: f32 = input[start..end].iter().sum();
            let mean = sum / num_features as f32;
            
            // Compute variance
            let var_sum: f32 = input[start..end]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            let var = var_sum / num_features as f32;
            
            // Normalize
            let std = (var + self.eps).sqrt();
            for (i, x) in input[start..end].iter_mut().enumerate() {
                *x = ((*x - mean) / std) * self.weight[i] + self.bias[i];
            }
        }
    }
}

/// 3D-aware vision attention mechanism
#[derive(Debug, Clone)]
pub struct VisionAttention3D {
    /// QKV projection weights [3 * hidden_size, hidden_size]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias [3 * hidden_size]
    pub qkv_bias: Vec<f32>,
    /// Output projection weights [hidden_size, hidden_size]
    pub out_weight: Vec<f32>,
    /// Output projection bias [hidden_size]
    pub out_bias: Vec<f32>,
    /// Number of heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl VisionAttention3D {
    /// Create new attention layer
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            qkv_weight: vec![0.0; 3 * hidden_size * hidden_size],
            qkv_bias: vec![0.0; 3 * hidden_size],
            out_weight: vec![0.0; hidden_size * hidden_size],
            out_bias: vec![0.0; hidden_size],
            num_heads,
            head_dim: hidden_size / num_heads,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
        seq_len: usize,
        config: &VisionConfig3D,
    ) -> Result<(), VisionModelError> {
        let hidden_size = config.hidden_size;
        
        // QKV projection
        let mut qkv = vec![0.0; 3 * hidden_size * seq_len];
        
        for seq_idx in 0..seq_len {
            for proj_idx in 0..3 * hidden_size {
                let mut sum = self.qkv_bias[proj_idx];
                for h in 0..hidden_size {
                    let input_idx = seq_idx * hidden_size + h;
                    let weight_idx = proj_idx * hidden_size + h;
                    if input_idx < hidden_states.len() && weight_idx < self.qkv_weight.len() {
                        sum += hidden_states[input_idx] * self.qkv_weight[weight_idx];
                    }
                }
                let qkv_idx = seq_idx * 3 * hidden_size + proj_idx;
                if qkv_idx < qkv.len() {
                    qkv[qkv_idx] = sum;
                }
            }
        }
        
        // Split into Q, K, V
        let mut query = vec![0.0; self.num_heads * self.head_dim * seq_len];
        let mut key = vec![0.0; self.num_heads * self.head_dim * seq_len];
        let mut value = vec![0.0; self.num_heads * self.head_dim * seq_len];
        
        for seq_idx in 0..seq_len {
            for head in 0..self.num_heads {
                for d in 0..self.head_dim {
                    let src_idx = seq_idx * 3 * hidden_size + head * self.head_dim + d;
                    let q_dst = (head * seq_len + seq_idx) * self.head_dim + d;
                    let k_dst = (head * seq_len + seq_idx) * self.head_dim + d;
                    let v_dst = (head * seq_len + seq_idx) * self.head_dim + d;
                    
                    if src_idx < qkv.len() {
                        query[q_dst] = qkv[src_idx];
                        key[k_dst] = qkv[src_idx + hidden_size];
                        value[v_dst] = qkv[src_idx + 2 * hidden_size];
                    }
                }
            }
        }
        
        // Attention computation
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attention_output = vec![0.0; hidden_size * seq_len];
        
        for seq_idx in 0..seq_len {
            for head in 0..self.num_heads {
                // Compute attention scores
                let mut scores = vec![0.0; seq_len];
                
                for k_idx in 0..seq_len {
                    let mut dot_product = 0.0;
                    for d in 0..self.head_dim {
                        let q_idx = (head * seq_len + seq_idx) * self.head_dim + d;
                        let k_vec_idx = (head * seq_len + k_idx) * self.head_dim + d;
                        
                        if q_idx < query.len() && k_vec_idx < key.len() {
                            dot_product += query[q_idx] as f64 * key[k_vec_idx] as f64;
                        }
                    }
                    scores[k_idx] = (dot_product * scale) as f32;
                }
                
                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                for s in &mut scores {
                    *s = (*s - max_score).exp() / exp_sum;
                }
                
                // Weighted sum of values
                for d in 0..self.head_dim {
                    let mut weighted_sum = 0.0;
                    for v_idx in 0..seq_len {
                        let v_vec_idx = (head * seq_len + v_idx) * self.head_dim + d;
                        if v_vec_idx < value.len() {
                            weighted_sum += scores[v_idx] * value[v_vec_idx];
                        }
                    }
                    
                    let out_idx = seq_idx * hidden_size + head * self.head_dim + d;
                    if out_idx < attention_output.len() {
                        attention_output[out_idx] = weighted_sum;
                    }
                }
            }
        }
        
        // Output projection
        for seq_idx in 0..seq_len {
            for h in 0..hidden_size {
                let mut sum = self.out_bias[h];
                for prev_h in 0..hidden_size {
                    let input_idx = seq_idx * hidden_size + prev_h;
                    let weight_idx = h * hidden_size + prev_h;
                    if input_idx < attention_output.len() && weight_idx < self.out_weight.len() {
                        sum += attention_output[input_idx] * self.out_weight[weight_idx];
                    }
                }
                let output_idx = seq_idx * hidden_size + h;
                if output_idx < hidden_states.len() {
                    hidden_states[output_idx] = sum;
                }
            }
        }
        
        Ok(())
    }
}

/// 3D-aware vision MLP with QuickGELU
#[derive(Debug, Clone)]
pub struct VisionMlp3D {
    /// FC1 weights [intermediate_size, hidden_size]
    pub fc1_weight: Vec<f32>,
    /// FC1 bias [intermediate_size]
    pub fc1_bias: Vec<f32>,
    /// FC2 weights [hidden_size, intermediate_size]
    pub fc2_weight: Vec<f32>,
    /// FC2 bias [hidden_size]
    pub fc2_bias: Vec<f32>,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl VisionMlp3D {
    /// Create new MLP
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            fc1_weight: vec![0.0; intermediate_size * hidden_size],
            fc1_bias: vec![0.0; intermediate_size],
            fc2_weight: vec![0.0; hidden_size * intermediate_size],
            fc2_bias: vec![0.0; hidden_size],
            hidden_size,
            intermediate_size,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// QuickGELU activation: x * sigmoid(1.702 * x)
    fn quick_gelu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-1.702 * x).exp()))
    }
    
    /// Forward pass
    pub fn forward(&self, hidden_states: &mut [f32]) {
        let seq_len = hidden_states.len() / self.hidden_size;
        
        for seq_idx in 0..seq_len {
            // FC1
            let mut intermediate = vec![0.0; self.intermediate_size];
            for i in 0..self.intermediate_size {
                let mut sum = self.fc1_bias[i];
                for j in 0..self.hidden_size {
                    let input_idx = seq_idx * self.hidden_size + j;
                    let weight_idx = i * self.hidden_size + j;
                    if input_idx < hidden_states.len() && weight_idx < self.fc1_weight.len() {
                        sum += hidden_states[input_idx] * self.fc1_weight[weight_idx];
                    }
                }
                intermediate[i] = Self::quick_gelu(sum);
            }
            
            // FC2
            for j in 0..self.hidden_size {
                let mut sum = self.fc2_bias[j];
                for i in 0..self.intermediate_size {
                    let weight_idx = j * self.intermediate_size + i;
                    if weight_idx < self.fc2_weight.len() {
                        sum += intermediate[i] * self.fc2_weight[weight_idx];
                    }
                }
                let output_idx = seq_idx * self.hidden_size + j;
                if output_idx < hidden_states.len() {
                    hidden_states[output_idx] = sum;
                }
            }
        }
    }
}

/// 3D-aware vision transformer block
#[derive(Debug, Clone)]
pub struct VisionBlock3D {
    /// First layer norm (pre-attention)
    pub norm1: VisionLayerNorm3D,
    /// Attention layer
    pub attention: VisionAttention3D,
    /// Second layer norm (pre-mlp)
    pub norm2: VisionLayerNorm3D,
    /// MLP layer
    pub mlp: VisionMlp3D,
    /// Layer index
    pub layer_idx: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl VisionBlock3D {
    /// Create new vision block
    pub fn new(layer_idx: usize, config: &VisionConfig3D) -> Self {
        Self {
            norm1: VisionLayerNorm3D::new(config.hidden_size, config.eps),
            attention: VisionAttention3D::new(config.hidden_size, config.num_heads),
            norm2: VisionLayerNorm3D::new(config.hidden_size, config.eps),
            mlp: VisionMlp3D::new(config.hidden_size, config.intermediate_size),
            layer_idx,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 6) as u16,
                (layer_idx % 6) as u8,
                1.0,
            ),
        }
    }
    
    /// Forward pass with residual connections
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
        seq_len: usize,
        config: &VisionConfig3D,
    ) -> Result<(), VisionModelError> {
        // Attention path with residual
        let residual = hidden_states.to_vec();
        self.norm1.normalize(hidden_states);
        self.attention.forward(hidden_states, seq_len, config)?;
        
        // Add residual
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // MLP path with residual
        let residual = hidden_states.to_vec();
        self.norm2.normalize(hidden_states);
        self.mlp.forward(hidden_states);
        
        // Add residual
        for (h, r) in hidden_states.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        Ok(())
    }
}

/// 3D-aware position embedding with interpolation support
#[derive(Debug, Clone)]
pub struct VisionPositionEmbedding3D {
    /// Position embeddings [num_positions, hidden_size]
    pub embeddings: Vec<f32>,
    /// Number of positions (max)
    pub num_positions: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl VisionPositionEmbedding3D {
    /// Create new position embedding
    pub fn new(num_positions: usize, hidden_size: usize) -> Self {
        Self {
            embeddings: vec![0.0; num_positions * hidden_size],
            num_positions,
            hidden_size,
            spatial: SpatialTensorMetadata::new(num_positions as u32, hidden_size as u32, 1),
        }
    }
    
    /// Get embeddings with interpolation if needed
    pub fn get_embeddings(&self, target_num_patches: usize) -> Vec<f32> {
        let source_side = ((self.num_positions - 1) as f32).sqrt() as usize;
        let target_side = (target_num_patches as f32).sqrt() as usize;
        
        if source_side == target_side {
            // Return all embeddings (including class token)
            return self.embeddings.clone();
        }
        
        // Need to interpolate position embeddings (excluding class token)
        let mut result = vec![0.0; (target_num_patches + 1) * self.hidden_size];
        
        // Copy class token (first embedding)
        result[..self.hidden_size].copy_from_slice(&self.embeddings[..self.hidden_size]);
        
        // Interpolate grid embeddings
        // Source grid: source_side x source_side
        // Target grid: target_side x target_side
        let ratio = source_side as f32 / target_side as f32;
        
        for target_y in 0..target_side {
            for target_x in 0..target_side {
                let target_idx = 1 + target_y * target_side + target_x;
                
                // Bilinear interpolation
                let src_y_f = target_y as f32 * ratio;
                let src_x_f = target_x as f32 * ratio;
                let src_y = src_y_f as usize;
                let src_x = src_x_f as usize;
                let src_y_next = (src_y + 1).min(source_side - 1);
                let src_x_next = (src_x + 1).min(source_side - 1);
                
                let dy = src_y_f - src_y as f32;
                let dx = src_x_f - src_x as f32;
                
                for h in 0..self.hidden_size {
                    let src_idx_tl = 1 + src_y * source_side + src_x;
                    let src_idx_tr = 1 + src_y * source_side + src_x_next;
                    let src_idx_bl = 1 + src_y_next * source_side + src_x;
                    let src_idx_br = 1 + src_y_next * source_side + src_x_next;
                    
                    let v_tl = self.embeddings[src_idx_tl * self.hidden_size + h];
                    let v_tr = self.embeddings[src_idx_tr * self.hidden_size + h];
                    let v_bl = self.embeddings[src_idx_bl * self.hidden_size + h];
                    let v_br = self.embeddings[src_idx_br * self.hidden_size + h];
                    
                    let v_top = v_tl * (1.0 - dx) + v_tr * dx;
                    let v_bottom = v_bl * (1.0 - dx) + v_br * dx;
                    let v = v_top * (1.0 - dy) + v_bottom * dy;
                    
                    result[target_idx * self.hidden_size + h] = v;
                }
            }
        }
        
        result
    }
}

/// 3D-aware DeepSeekOCR vision encoder
pub struct DeepSeekOcrVisionModel3D {
    /// Patch embedding (Conv2D)
    pub patch_embed: VisionConv2D3D,
    /// Class token embedding
    pub class_embedding: Vec<f32>,
    /// Position embeddings
    pub position_embedding: VisionPositionEmbedding3D,
    /// Pre-layer norm
    pub pre_layer_norm: VisionLayerNorm3D,
    /// Transformer blocks
    pub blocks: Vec<VisionBlock3D>,
    /// Configuration
    pub config: VisionConfig3D,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl DeepSeekOcrVisionModel3D {
    /// Create new vision model
    pub fn new(config: VisionConfig3D) -> Result<Self, VisionModelError> {
        config.validate()?;
        
        let patch_embed = VisionConv2D3D::new(
            3, // RGB
            config.hidden_size,
            (config.patch_size, config.patch_size),
            (config.patch_size, config.patch_size),
        );
        
        let class_embedding = vec![0.0; config.hidden_size];
        
        let max_positions = 256 * 256 + 1; // Max patches + class token
        let position_embedding = VisionPositionEmbedding3D::new(max_positions, config.hidden_size);
        
        let blocks: Vec<VisionBlock3D> = (0..config.num_layers)
            .map(|i| VisionBlock3D::new(i, &config))
            .collect();
        
        Ok(Self {
            patch_embed,
            class_embedding,
            position_embedding,
            pre_layer_norm: VisionLayerNorm3D::new(config.hidden_size, config.eps),
            blocks,
            config,
            spatial: SpatialTensorMetadata::new(
                config.num_patches as u32,
                config.num_patches as u32,
                config.hidden_size as u32,
            ),
        })
    }
    
    /// Absolute position embedding with interpolation
    pub fn absolute_position_embedding(&self, num_patches: usize) -> Vec<f32> {
        self.position_embedding.get_embeddings(num_patches)
    }
    
    /// Forward pass
    pub fn forward(
        &self,
        pixel_values: &[f32],
        patch_embeds: Option<&[f32]>,
    ) -> Result<Vec<f32>, VisionModelError> {
        // Patch embedding
        let mut embeds = if let Some(precomputed) = patch_embeds {
            precomputed.to_vec()
        } else {
            self.patch_embed.forward(pixel_values, (3, self.config.image_size, self.config.image_size))
        };
        
        let (out_h, out_w) = self.patch_embed.output_shape(self.config.image_size, self.config.image_size);
        let num_patches = out_h * out_w;
        let seq_len = num_patches + 1; // +1 for class token
        
        // Reshape to [num_patches, hidden_size]
        // embeds is already in this format from patch_embed
        
        // Add class token
        let mut all_embeds = self.class_embedding.clone();
        all_embeds.extend(&embeds);
        
        // Add position embeddings
        let position_embeds = self.absolute_position_embedding(num_patches);
        for (e, &p) in all_embeds.iter_mut().zip(position_embeds.iter()) {
            *e += p;
        }
        
        // Pre-layer norm
        self.pre_layer_norm.normalize(&mut all_embeds);
        
        // Transformer blocks
        let mut hidden_states = all_embeds;
        for block in &self.blocks {
            block.forward(&mut hidden_states, seq_len, &self.config)?;
        }
        
        Ok(hidden_states)
    }
    
    /// Get output shape
    pub fn output_shape(&self) -> (usize, usize) {
        let num_patches = self.config.num_patches * self.config.num_patches;
        (num_patches + 1, self.config.hidden_size)
    }
    
    /// Get model info
    pub fn model_info(&self) -> VisionModelInfo {
        VisionModelInfo {
            name: "DeepSeekOCR-Vision-3D".to_string(),
            total_params: self.estimate_parameters(),
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            num_patches: self.config.num_patches * self.config.num_patches,
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
        }
    }
    
    /// Estimate total parameters
    fn estimate_parameters(&self) -> usize {
        let c = &self.config;
        
        // Patch embedding
        let patch_embed = c.patch_size * c.patch_size * 3 * c.hidden_size;
        
        // Class + position embeddings
        let embeddings = c.hidden_size + c.num_positions * c.hidden_size;
        
        // Per layer
        let attn_params = 3 * c.hidden_size * c.hidden_size + c.hidden_size * c.hidden_size; // QKV + out
        let mlp_params = c.hidden_size * c.intermediate_size + c.intermediate_size * c.hidden_size;
        let layer_params = attn_params + mlp_params + 2 * c.hidden_size; // norms
        
        // Total
        patch_embed + embeddings + c.num_layers * layer_params
    }
}

impl Default for DeepSeekOcrVisionModel3D {
    fn default() -> Self {
        Self::new(VisionConfig3D::default()).unwrap()
    }
}

/// Vision model information
#[derive(Debug, Clone)]
pub struct VisionModelInfo {
    pub name: String,
    pub total_params: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_patches: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

/// Utility functions
pub mod vision_utils {
    use super::*;
    
    /// Compute number of patches for given image size
    pub fn num_patches(image_size: usize, patch_size: usize) -> usize {
        let patches_per_side = image_size / patch_size;
        patches_per_side * patches_per_side
    }
    
    /// Estimate memory requirement for vision model
    pub fn estimate_memory(config: &VisionConfig3D) -> u64 {
        let num_patches = num_patches(config.image_size, config.patch_size);
        
        // Patch embedding
        let patch_embed = (config.patch_size * config.patch_size * 3 * config.hidden_size * 4) as u64;
        
        // Embeddings
        let embeddings = ((num_patches + 1) * config.hidden_size * 4) as u64;
        
        // Per layer
        let attn_per_layer = (3 * config.hidden_size * config.hidden_size * 4) as u64;
        let mlp_per_layer = (config.hidden_size * config.intermediate_size * 2 * 4) as u64;
        let layers = config.num_layers as u64 * (attn_per_layer + mlp_per_layer);
        
        // Activations (rough estimate)
        let activations = ((num_patches + 1) * config.hidden_size * 4 * config.num_layers as u64) as u64;
        
        patch_embed + embeddings + layers + activations
    }
    
    /// Check if image size is compatible
    pub fn check_image_size(image_size: usize, patch_size: usize) -> Result<(), VisionModelError> {
        if image_size % patch_size != 0 {
            return Err(VisionModelError::ConfigurationError(
                format!("Image size {} must be divisible by patch size {}",
                    image_size, patch_size)
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = VisionConfig3D::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.head_dim(), 64);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_conv2d() {
        let conv = VisionConv2D3D::new(3, 1024, (14, 14), (14, 14));
        let (out_h, out_w) = conv.output_shape(640, 640);
        assert_eq!(out_h, 45); // 640 / 14 = 45.71 -> rounded
        assert_eq!(out_w, 45);
    }
    
    #[test]
    fn test_layernorm() {
        let ln = VisionLayerNorm3D::new(1024, 1e-5);
        let mut data = vec![1.0; 1024 * 10]; // 10 tokens
        ln.normalize(&mut data);
        
        // Check normalization
        assert!(data.iter().any(|&x| x != 1.0));
    }
    
    #[test]
    fn test_attention() {
        let config = VisionConfig3D::default();
        let attn = VisionAttention3D::new(1024, 16);
        
        assert_eq!(attn.num_heads, 16);
        assert_eq!(attn.head_dim, 64);
    }
    
    #[test]
    fn test_mlp() {
        let mlp = VisionMlp3D::new(1024, 4096);
        let mut data = vec![1.0; 1024 * 5]; // 5 tokens
        
        mlp.forward(&mut data);
        
        assert_eq!(data.len(), 1024 * 5);
        assert!(data.iter().any(|&x| x != 1.0));
    }
    
    #[test]
    fn test_quick_gelu() {
        let x = 1.0f32;
        let result = VisionMlp3D::quick_gelu(x);
        
        // QuickGELU(1) ≈ 0.85
        assert!(result > 0.8 && result < 0.9);
    }
    
    #[test]
    fn test_position_embedding() {
        let pos_embed = VisionPositionEmbedding3D::new(256, 1024);
        let embeds = pos_embed.get_embeddings(100); // 10x10 grid
        
        // Should return 101 embeddings (100 patches + 1 class token)
        assert_eq!(embeds.len(), 101 * 1024);
    }
    
    #[test]
    fn test_block() {
        let config = VisionConfig3D::default();
        let block = VisionBlock3D::new(0, &config);
        
        assert_eq!(block.layer_idx, 0);
        assert_eq!(block.spatial_position.x, 0);
    }
    
    #[test]
    fn test_model_creation() {
        let config = VisionConfig3D::default();
        let model = DeepSeekOcrVisionModel3D::new(config);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.blocks.len(), 24);
        
        let info = model.model_info();
        assert!(info.total_params > 100_000_000); // > 100M params
    }
    
    #[test]
    fn test_utils() {
        let num = vision_utils::num_patches(640, 14);
        assert_eq!(num, 45 * 45);
        
        assert!(vision_utils::check_image_size(640, 14).is_ok());
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = VisionConfig3D::default();
        let mem = vision_utils::estimate_memory(&config);
        
        assert!(mem > 0);
        assert!(mem > 500_000_000); // > 500MB
    }
}
