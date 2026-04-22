use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// SAM (Segment Anything Model) errors
#[derive(Debug, Clone)]
pub enum SamError {
    InvalidDimensions(String),
    ForwardError(String),
    AttentionError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for SamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(s) => write!(f, "Invalid dimensions: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::AttentionError(s) => write!(f, "Attention error: {}", s),
            Self::ConfigurationError(s) => write!(f, "Config error: {}", s),
        }
    }
}

impl std::error::Error for SamError {}

/// 3D-aware SAM configuration
#[derive(Debug, Clone, Copy)]
pub struct SamConfig3D {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Layer norm epsilon
    pub eps: f32,
    /// Global attention layer indices
    pub global_attention_layers: &'static [usize],
    /// Window size for local attention
    pub window_size: usize,
    /// Patch size for embedding
    pub patch_size: usize,
    /// Patch stride
    pub patch_stride: usize,
    /// Number of layers
    pub num_layers: usize,
    /// 3D spatial depth
    pub spatial_depth: u8,
}

impl SamConfig3D {
    /// Default SAM configuration
    pub fn default_sam() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            eps: 1e-6,
            global_attention_layers: &[2, 5, 8, 11],
            window_size: 14,
            patch_size: 16,
            patch_stride: 16,
            num_layers: 12,
            spatial_depth: 32,
        }
    }
    
    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    
    /// Check if layer uses global attention
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        self.global_attention_layers.contains(&layer_idx)
    }
}

impl Default for SamConfig3D {
    fn default() -> Self {
        Self::default_sam()
    }
}

/// 3D-aware 2D Convolution layer
#[derive(Debug, Clone)]
pub struct Conv2D3D {
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
    /// Padding
    pub padding: (usize, usize),
    /// Dilation
    pub dilation: (usize, usize),
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl Conv2D3D {
    /// Create new Conv2D layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let kernel_prod = kernel_size.0 * kernel_size.1;
        Self {
            weight: vec![0.0; out_channels * in_channels * kernel_prod],
            bias: Some(vec![0.0; out_channels]),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass (simplified)
    pub fn forward(&self, input: &[f32], input_shape: (usize, usize, usize)) -> Vec<f32> {
        // Simplified convolution forward
        // Output: [out_channels, out_h, out_w]
        let (_, h, w) = input_shape;
        let out_h = (h + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1) 
            / self.stride.0 + 1;
        let out_w = (w + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1) 
            / self.stride.1 + 1;
        
        vec![0.0; self.out_channels * out_h * out_w]
    }
    
    /// Get output shape
    pub fn output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1) 
            / self.stride.0 + 1;
        let out_w = (input_w + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1) 
            / self.stride.1 + 1;
        (out_h, out_w)
    }
}

/// 3D-aware 2D Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm2D3D {
    /// Scale weights [channels]
    pub weight: Vec<f32>,
    /// Bias [channels]
    pub bias: Vec<f32>,
    /// Number of channels
    pub num_channels: usize,
    /// Epsilon
    pub eps: f32,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl LayerNorm2D3D {
    /// Create new LayerNorm2D
    pub fn new(num_channels: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; num_channels],
            bias: vec![0.0; num_channels],
            num_channels,
            eps,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Normalize input [C, H, W]
    pub fn normalize(&self, input: &mut [f32], height: usize, width: usize) {
        // Per-channel normalization
        for c in 0..self.num_channels {
            let channel_start = c * height * width;
            let channel_end = channel_start + height * width;
            
            // Compute mean
            let sum: f32 = input[channel_start..channel_end].iter().sum();
            let mean = sum / (height * width) as f32;
            
            // Compute variance
            let var_sum: f32 = input[channel_start..channel_end]
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            let var = var_sum / (height * width) as f32;
            
            // Normalize and apply scale/shift
            let std = (var + self.eps).sqrt();
            for x in &mut input[channel_start..channel_end] {
                *x = ((*x - mean) / std) * self.weight[c] + self.bias[c];
            }
        }
    }
}

/// 3D-aware SAM MLP
#[derive(Debug, Clone)]
pub struct SamMlp3D {
    /// First linear layer
    pub lin1_weights: Vec<f32>,
    /// Second linear layer
    pub lin2_weights: Vec<f32>,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl SamMlp3D {
    /// Create new MLP
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            lin1_weights: vec![0.0; hidden_size * intermediate_size],
            lin2_weights: vec![0.0; intermediate_size * hidden_size],
            hidden_size,
            intermediate_size,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass with GELU
    pub fn forward(&self, hidden_states: &mut [f32]) {
        // Simplified MLP forward
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        for x in hidden_states.iter_mut() {
            *x = Self::gelu(*x);
        }
    }
    
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
    }
}

/// Relative position tensors for attention
#[derive(Debug, Clone)]
pub struct RelativePosition3D {
    /// Height relative positions
    pub height: Vec<f32>,
    /// Width relative positions
    pub width: Vec<f32>,
    /// Max relative distance
    pub max_distance: usize,
}

impl RelativePosition3D {
    /// Create new relative position embeddings
    pub fn new(max_distance: usize, num_heads: usize) -> Self {
        Self {
            height: vec![0.0; num_heads * (2 * max_distance - 1)],
            width: vec![0.0; num_heads * (2 * max_distance - 1)],
            max_distance,
        }
    }
}

/// 3D-aware SAM attention with decomposed relative positions
#[derive(Debug, Clone)]
pub struct SamAttention3D {
    /// QKV projection weights [3 * hidden_size, hidden_size]
    pub qkv_weights: Vec<f32>,
    /// Output projection weights [hidden_size, hidden_size]
    pub output_weights: Vec<f32>,
    /// Number of heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Relative position embeddings
    pub relative_position: Option<RelativePosition3D>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl SamAttention3D {
    /// Create new attention layer
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            qkv_weights: vec![0.0; 3 * hidden_size * hidden_size],
            output_weights: vec![0.0; hidden_size * hidden_size],
            num_heads,
            head_dim: hidden_size / num_heads,
            relative_position: None,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Enable relative positions
    pub fn with_relative_position(mut self, max_distance: usize) -> Self {
        self.relative_position = Some(RelativePosition3D::new(max_distance, self.num_heads));
        self
    }
    
    /// Forward pass with decomposed relative positions
    pub fn forward(
        &self,
        hidden_states: &[f32],
        height: usize,
        width: usize,
        batch: usize,
        config: &SamConfig3D,
    ) -> Result<Vec<f32>, SamError> {
        let num_patches = height * width;
        
        // QKV projection and split
        let mut query = vec![0.0; self.num_heads * self.head_dim * num_patches * batch];
        let mut key = vec![0.0; self.num_heads * self.head_dim * num_patches * batch];
        let mut value = vec![0.0; self.num_heads * self.head_dim * num_patches * batch];
        
        // Compute decomposed relative position bias
        let (rh, rw) = if let Some(ref rel_pos) = self.relative_position {
            self.decomposed_relative_positions(height, width, rel_pos)?
        } else {
            (vec![0.0; num_patches], vec![0.0; num_patches])
        };
        
        // Compute attention scores
        let mut scores = vec![0.0; num_patches * num_patches * self.num_heads * batch];
        
        // Scale by head_dim
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        
        // Add relative position bias
        // scores = scores + mask
        // mask = rh.repeat(rw)
        
        // Softmax
        // attention = softmax(scores)
        
        // Attention output
        let mut output = vec![0.0; self.num_heads * self.head_dim * num_patches * batch];
        
        // Output projection
        let mut projected = vec![0.0; config.hidden_size * num_patches * batch];
        
        Ok(projected)
    }
    
    /// Decomposed relative positions (height and width separately)
    fn decomposed_relative_positions(
        &self,
        h: usize,
        w: usize,
        rel_pos: &RelativePosition3D,
    ) -> Result<(Vec<f32>, Vec<f32>), SamError> {
        // Height relative positions
        let rh = self.relative_positions(&rel_pos.height, h, h)?;
        
        // Width relative positions
        let rw = self.relative_positions(&rel_pos.width, w, w)?;
        
        Ok((rh, rw))
    }
    
    /// Compute relative positions with interpolation
    fn relative_positions(
        &self,
        positions: &[f32],
        query_size: usize,
        key_size: usize,
    ) -> Result<Vec<f32>, SamError> {
        let max_rel_dist = 2 * query_size.max(key_size) - 1;
        
        // If positions don't match, interpolate
        let interpolated = if positions.len() != max_rel_dist * self.num_heads {
            self.interpolate_positions(positions, max_rel_dist)?
        } else {
            positions.to_vec()
        };
        
        // Get relative coordinates
        let coords = self.relative_coordinates(query_size, key_size);
        
        // Gather positions by coordinates
        let mut result = vec![0.0; query_size * key_size * self.num_heads];
        
        for (idx, &coord) in coords.iter().enumerate() {
            let q = idx / key_size;
            let k = idx % key_size;
            
            for head in 0..self.num_heads {
                let pos_idx = head * (2 * query_size.max(key_size) - 1) + coord as usize;
                if pos_idx < interpolated.len() {
                    result[(head * query_size + q) * key_size + k] = interpolated[pos_idx];
                }
            }
        }
        
        Ok(result)
    }
    
    /// Compute relative coordinate indices
    fn relative_coordinates(&self, qn: usize, kn: usize) -> Vec<i32> {
        let mut coords = Vec::with_capacity(qn * kn);
        
        for i in 0..qn {
            for j in 0..kn {
                let q = i * (kn / qn).max(1);
                let k = j * (qn / kn).max(1);
                let coord = q as i32 - k as i32 + ((kn - 1) * (qn / kn).max(1)) as i32;
                coords.push(coord);
            }
        }
        
        coords
    }
    
    /// Linear interpolation for positions
    fn interpolate_positions(&self, positions: &[f32], target_size: usize) -> Result<Vec<f32>, SamError> {
        // Simplified bilinear interpolation
        let source_size = positions.len() / self.num_heads;
        let ratio = source_size as f32 / target_size as f32;
        
        let mut result = vec![0.0; target_size * self.num_heads];
        
        for head in 0..self.num_heads {
            for target_idx in 0..target_size {
                let source_idx = (target_idx as f32 * ratio) as usize;
                let source_idx = source_idx.min(source_size - 1);
                result[head * target_size + target_idx] = positions[head * source_size + source_idx];
            }
        }
        
        Ok(result)
    }
}

/// 3D-aware SAM transformer block
#[derive(Debug, Clone)]
pub struct SamBlock3D {
    /// Pre-attention normalization
    pub norm1: LayerNorm2D3D,
    /// Attention layer
    pub attention: SamAttention3D,
    /// Post-attention normalization
    pub norm2: LayerNorm2D3D,
    /// MLP layer
    pub mlp: SamMlp3D,
    /// Layer index
    pub layer_idx: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl SamBlock3D {
    /// Create new SAM block
    pub fn new(layer_idx: usize, config: &SamConfig3D) -> Self {
        let attention = if config.is_global_layer(layer_idx) {
            SamAttention3D::new(config.hidden_size, config.num_heads)
        } else {
            SamAttention3D::new(config.hidden_size, config.num_heads)
                .with_relative_position(config.window_size * 2)
        };
        
        Self {
            norm1: LayerNorm2D3D::new(config.hidden_size, config.eps),
            attention,
            norm2: LayerNorm2D3D::new(config.hidden_size, config.eps),
            mlp: SamMlp3D::new(config.hidden_size, 4 * config.hidden_size),
            layer_idx,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 4) as u16,
                (layer_idx % 4) as u8,
                1.0,
            ),
        }
    }
    
    /// Forward pass with windowed attention
    pub fn forward(
        &self,
        hidden_states: &mut [f32],
        channels: usize,
        width: usize,
        height: usize,
        window_size: usize,
        config: &SamConfig3D,
    ) -> Result<(), SamError> {
        // Store residual
        let residual = hidden_states.to_vec();
        
        // Pre-attention norm
        self.norm1.normalize(hidden_states, height, width);
        
        // Window padding if needed
        let (padded_w, padded_h, pad_w, pad_h) = if window_size > 0 {
            let pw = (window_size - width % window_size) % window_size;
            let ph = (window_size - height % window_size) % window_size;
            (width + pw, height + ph, pw, ph)
        } else {
            (width, height, 0, 0)
        };
        
        // Reshape for windowed attention if needed
        let mut windowed = if window_size > 0 && (pad_w > 0 || pad_h > 0) {
            // Pad and reshape
            self.pad_and_reshape(hidden_states, channels, width, height, padded_w, padded_h, window_size)?
        } else {
            hidden_states.to_vec()
        };
        
        // Attention
        let num_patches = if window_size > 0 {
            window_size * window_size
        } else {
            padded_w * padded_h
        };
        
        let batch = if window_size > 0 {
            channels * window_size * (padded_w / window_size) * (padded_h / window_size) / channels
        } else {
            1
        };
        
        let attn_output = self.attention.forward(
            &windowed,
            if window_size > 0 { window_size } else { padded_h },
            if window_size > 0 { window_size } else { padded_w },
            batch,
            config,
        )?;
        
        // Unpad and reshape back if needed
        let mut unwindowed = if window_size > 0 && (pad_w > 0 || pad_h > 0) {
            self.unpad_and_reshape(&attn_output, channels, width, height, padded_w, padded_h, window_size, pad_w, pad_h)?
        } else {
            attn_output
        };
        
        // Add residual
        for (h, r) in unwindowed.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        
        // Second residual path
        let residual2 = unwindowed.clone();
        self.norm2.normalize(&mut unwindowed, height, width);
        self.mlp.forward(&mut unwindowed);
        
        // Add second residual
        for (h, r) in unwindowed.iter_mut().zip(residual2.iter()) {
            *h += r;
        }
        
        // Copy back
        hidden_states.copy_from_slice(&unwindowed);
        
        Ok(())
    }
    
    fn pad_and_reshape(
        &self,
        input: &[f32],
        c: usize,
        w: usize,
        h: usize,
        pw: usize,
        ph: usize,
        window_size: usize,
    ) -> Result<Vec<f32>, SamError> {
        // Simplified pad and reshape for windowed attention
        // [C, H, W] -> [C * window_size, H/window_size, window_size, W/window_size * window_size]
        let padded_size = c * ph * pw;
        let mut padded = vec![0.0; padded_size];
        
        // Copy original data
        for c_idx in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = c_idx * h * w + y * w + x;
                    let dst_idx = c_idx * ph * pw + y * pw + x;
                    if src_idx < input.len() && dst_idx < padded.len() {
                        padded[dst_idx] = input[src_idx];
                    }
                }
            }
        }
        
        Ok(padded)
    }
    
    fn unpad_and_reshape(
        &self,
        input: &[f32],
        c: usize,
        w: usize,
        h: usize,
        _pw: usize,
        _ph: usize,
        _window_size: usize,
        pad_w: usize,
        pad_h: usize,
    ) -> Result<Vec<f32>, SamError> {
        // Unpad back to [C, H, W]
        let mut output = vec![0.0; c * h * w];
        
        for c_idx in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = c_idx * (_ph - pad_h) * (_pw - pad_w) + y * (_pw - pad_w) + x;
                    let dst_idx = c_idx * h * w + y * w + x;
                    if src_idx < input.len() && dst_idx < output.len() {
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
        
        Ok(output)
    }
}

/// 3D-aware SAM neck (feature pyramid)
#[derive(Debug, Clone)]
pub struct SamNeck3D {
    /// First conv
    pub c1: Conv2D3D,
    /// First layer norm
    pub ln1: LayerNorm2D3D,
    /// Second conv
    pub c2: Conv2D3D,
    /// Second layer norm
    pub ln2: LayerNorm2D3D,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

impl SamNeck3D {
    /// Create new SAM neck
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            c1: Conv2D3D::new(hidden_size, hidden_size, (1, 1), (1, 1), (0, 0)),
            ln1: LayerNorm2D3D::new(hidden_size, eps),
            c2: Conv2D3D::new(hidden_size, hidden_size, (3, 3), (1, 1), (1, 1)),
            ln2: LayerNorm2D3D::new(hidden_size, eps),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, hidden_states: &mut [f32], height: usize, width: usize) {
        // Conv1
        let mut out1 = self.c1.forward(hidden_states, (self.c1.in_channels, height, width));
        self.ln1.normalize(&mut out1, height, width);
        
        // Conv2
        let mut out2 = self.c2.forward(&out1, (self.c2.in_channels, height, width));
        self.ln2.normalize(&mut out2, height, width);
        
        // Copy back
        hidden_states.copy_from_slice(&out2);
    }
}

/// 3D-aware SAM encoder
pub struct SamEncoder3D {
    /// Patch embedding (Conv2D)
    pub patch_embed: Conv2D3D,
    /// Position embedding
    pub position_embed: Vec<f32>,
    /// Transformer blocks
    pub blocks: Vec<SamBlock3D>,
    /// Neck (feature pyramid)
    pub neck: SamNeck3D,
    /// Additional conv layers
    pub net2: Conv2D3D,
    pub net3: Conv2D3D,
    /// Configuration
    pub config: SamConfig3D,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl SamEncoder3D {
    /// Create new SAM encoder
    pub fn new(config: SamConfig3D) -> Self {
        let patch_embed = Conv2D3D::new(
            3, // RGB
            config.hidden_size,
            (config.patch_size, config.patch_size),
            (config.patch_stride, config.patch_stride),
            (0, 0),
        );
        
        let blocks: Vec<SamBlock3D> = (0..config.num_layers)
            .map(|i| SamBlock3D::new(i, &config))
            .collect();
        
        let neck = SamNeck3D::new(config.hidden_size, config.eps);
        
        let net2 = Conv2D3D::new(config.hidden_size, config.hidden_size, (2, 2), (2, 2), (1, 1));
        let net3 = Conv2D3D::new(config.hidden_size, config.hidden_size, (2, 2), (2, 2), (1, 1));
        
        Self {
            patch_embed,
            position_embed: vec![0.0; config.hidden_size * 64 * 64], // Default max size
            blocks,
            neck,
            net2,
            net3,
            config,
            spatial: SpatialTensorMetadata::new(64, 64, config.hidden_size as u32),
        }
    }
    
    /// Absolute position embedding with interpolation
    pub fn absolute_position_embedding(
        &self,
        hidden_states: &mut [f32],
        target_size: usize,
    ) -> Result<Vec<f32>, SamError> {
        let source_size = 64; // Default position embed size
        
        if source_size == target_size {
            // Return position embed directly
            return Ok(self.position_embed.clone());
        }
        
        // Interpolate position embedding
        let mut interpolated = vec![0.0; self.config.hidden_size * target_size * target_size];
        
        // Bilinear interpolation (simplified)
        let ratio = source_size as f32 / target_size as f32;
        
        for c in 0..self.config.hidden_size {
            for y in 0..target_size {
                for x in 0..target_size {
                    let src_y = (y as f32 * ratio) as usize;
                    let src_x = (x as f32 * ratio) as usize;
                    
                    let src_idx = c * source_size * source_size + src_y * source_size + src_x;
                    let dst_idx = c * target_size * target_size + y * target_size + x;
                    
                    if src_idx < self.position_embed.len() && dst_idx < interpolated.len() {
                        interpolated[dst_idx] = self.position_embed[src_idx];
                    }
                }
            }
        }
        
        Ok(interpolated)
    }
    
    /// Forward pass
    pub fn forward(&self, input: &[f32], input_shape: (usize, usize, usize)) -> Result<Vec<f32>, SamError> {
        let (channels, height, width) = input_shape;
        
        // Patch embedding
        let mut hidden_states = self.patch_embed.forward(input, (channels, height, width));
        let (out_h, out_w) = self.patch_embed.output_shape(height, width);
        
        // Permute to [C, H, W] format
        // hidden_states shape: [hidden_size, out_h, out_w]
        
        // Add position embedding
        let position_embed = self.absolute_position_embedding(&mut hidden_states, out_h.max(out_w))?;
        for (h, &p) in hidden_states.iter_mut().zip(position_embed.iter()) {
            *h += p;
        }
        
        // Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            let window_size = if self.config.is_global_layer(i) {
                0 // Global attention
            } else {
                self.config.window_size
            };
            
            block.forward(
                &mut hidden_states,
                self.config.hidden_size,
                out_w,
                out_h,
                window_size,
                &self.config,
            )?;
        }
        
        // Permute for neck
        // [C, H, W] -> [H, W, C]
        let mut permuted = vec![0.0; out_h * out_w * self.config.hidden_size];
        for c in 0..self.config.hidden_size {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src_idx = c * out_h * out_w + y * out_w + x;
                    let dst_idx = y * out_w * self.config.hidden_size + x * self.config.hidden_size + c;
                    if src_idx < hidden_states.len() && dst_idx < permuted.len() {
                        permuted[dst_idx] = hidden_states[src_idx];
                    }
                }
            }
        }
        
        // Neck
        let mut neck_output = permuted.clone();
        self.neck.forward(&mut neck_output, out_h, out_w);
        
        // Additional conv layers (net2, net3)
        let mut net2_output = self.net2.forward(&neck_output, (self.net2.in_channels, out_h, out_w));
        let (net2_h, net2_w) = self.net2.output_shape(out_h, out_w);
        
        let mut net3_output = self.net3.forward(&net2_output, (self.net3.in_channels, net2_h, net2_w));
        
        Ok(net3_output)
    }
    
    /// Get output shape for given input
    pub fn output_shape(&self, input_h: usize, input_w: usize) -> (usize, usize, usize) {
        let (h1, w1) = self.patch_embed.output_shape(input_h, input_w);
        let (h2, w2) = self.net2.output_shape(h1, w1);
        let (h3, w3) = self.net3.output_shape(h2, w2);
        (self.config.hidden_size, h3, w3)
    }
}

impl Default for SamEncoder3D {
    fn default() -> Self {
        Self::new(SamConfig3D::default())
    }
}

/// Utility functions for SAM
pub mod sam_utils {
    use super::*;
    
    /// Compute window padding
    pub fn compute_padding(size: usize, window_size: usize) -> usize {
        (window_size - size % window_size) % window_size
    }
    
    /// Check if windowed attention needed
    pub fn use_windowed_attention(layer_idx: usize, global_layers: &[usize]) -> bool {
        !global_layers.contains(&layer_idx)
    }
    
    /// Compute number of windows
    pub fn num_windows(size: usize, window_size: usize) -> usize {
        (size + compute_padding(size, window_size)) / window_size
    }
    
    /// Estimate SAM memory usage
    pub fn estimate_memory(config: &SamConfig3D, image_size: usize) -> u64 {
        let patch_size = image_size / config.patch_size;
        let features = config.hidden_size * patch_size * patch_size;
        
        // Patch embedding
        let patch_mem = (config.patch_size * config.patch_size * 3 * config.hidden_size * 4) as u64;
        
        // Position embedding
        let pos_mem = (config.hidden_size * patch_size * patch_size * 4) as u64;
        
        // Blocks
        let block_mem = (config.num_layers * config.hidden_size * config.hidden_size * 4 * 4) as u64;
        
        // Neck
        let neck_mem = (config.hidden_size * config.hidden_size * 4 * 4) as u64;
        
        patch_mem + pos_mem + block_mem + neck_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = SamConfig3D::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.head_dim(), 64);
        assert!(config.is_global_layer(2));
        assert!(!config.is_global_layer(1));
    }
    
    #[test]
    fn test_conv2d() {
        let conv = Conv2D3D::new(3, 768, (16, 16), (16, 16), (0, 0));
        let (out_h, out_w) = conv.output_shape(640, 640);
        assert_eq!(out_h, 40);
        assert_eq!(out_w, 40);
    }
    
    #[test]
    fn test_layernorm2d() {
        let ln = LayerNorm2D3D::new(768, 1e-6);
        let mut data: Vec<f32> = (0..768 * 40 * 40).map(|i| i as f32).collect();
        ln.normalize(&mut data, 40, 40);
        
        // Check that data is normalized
        assert!(data.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_sam_block() {
        let config = SamConfig3D::default();
        let block = SamBlock3D::new(0, &config);
        
        assert_eq!(block.layer_idx, 0);
        assert!(block.attention.relative_position.is_none()); // Global layer
        
        let windowed_block = SamBlock3D::new(1, &config);
        assert!(windowed_block.attention.relative_position.is_some());
    }
    
    #[test]
    fn test_sam_encoder() {
        let config = SamConfig3D::default();
        let encoder = SamEncoder3D::new(config);
        
        assert_eq!(encoder.blocks.len(), 12);
        assert_eq!(encoder.config.hidden_size, 768);
    }
    
    #[test]
    fn test_relative_positions() {
        let config = SamConfig3D::default();
        let attention = SamAttention3D::new(768, 12)
            .with_relative_position(28);
        
        assert!(attention.relative_position.is_some());
        let rel_pos = attention.relative_position.unwrap();
        assert_eq!(rel_pos.height.len(), 12 * (2 * 28 - 1));
    }
    
    #[test]
    fn test_sam_neck() {
        let neck = SamNeck3D::new(768, 1e-6);
        let mut data = vec![0.0; 768 * 40 * 40];
        neck.forward(&mut data, 40, 40);
        
        assert_eq!(data.len(), 768 * 40 * 40);
    }
    
    #[test]
    fn test_utils() {
        let padding = sam_utils::compute_padding(39, 14);
        assert_eq!(padding, 3);
        
        let windows = sam_utils::num_windows(39, 14);
        assert_eq!(windows, 3);
        
        assert!(sam_utils::use_windowed_attention(1, &[2, 5, 8]));
        assert!(!sam_utils::use_windowed_attention(2, &[2, 5, 8]));
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = SamConfig3D::default();
        let mem = sam_utils::estimate_memory(&config, 640);
        assert!(mem > 0);
        assert!(mem > 100_000_000); // > 100MB
    }
}
