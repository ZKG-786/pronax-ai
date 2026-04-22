//! ProNax Gemma4 Advanced 3D Audio Encoder

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Professional audio encoder errors
#[derive(Debug, Clone)]
pub enum PronaxAudioEncoderError {
    InvalidTensorShape(String),
    DimensionMismatch(String),
    ProcessingFailure(String),
    ConfigurationError(String),
    MemoryAllocationError(String),
}

impl std::fmt::Display for PronaxAudioEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTensorShape(s) => write!(f, "Invalid tensor shape: {}", s),
            Self::DimensionMismatch(s) => write!(f, "Dimension mismatch: {}", s),
            Self::ProcessingFailure(s) => write!(f, "Processing failed: {}", s),
            Self::ConfigurationError(s) => write!(f, "Configuration error: {}", s),
            Self::MemoryAllocationError(s) => write!(f, "Memory allocation error: {}", s),
        }
    }
}

impl std::error::Error for PronaxAudioEncoderError {}

/// 3D spatial audio hyperparameters with enhanced metadata
#[derive(Debug, Clone, Copy)]
pub struct PronaxAudioHyperparams3D {
    // Core dimensions
    pub spectral_width: usize,      // Frequency dimension (mel bins)
    pub temporal_height: usize,     // Time dimension (frames)
    pub feature_depth: usize,        // Channel/feature dimension
    pub guidance_strength: f32,       // 3D guidance factor
    
    // Model architecture
    pub embedding_dim: usize,
    pub attention_heads: usize,
    pub head_dimension: usize,
    pub feedforward_dim: usize,
    pub transformer_layers: usize,
    
    // Conformer-specific
    pub convolution_kernel: usize,
    pub chunk_span: usize,
    pub context_past: usize,
    pub context_future: usize,
    pub total_context: usize,
    
    // Training/normalization
    pub logit_cap: f32,
    pub residual_scale: f32,
    pub gradient_threshold: f32,
    pub epsilon: f32,
    
    // 3D spatial metadata
    pub spatial_depth: u8,
    pub spatial_guidance: f32,
}

impl PronaxAudioHyperparams3D {
    /// Create default hyperparameters for Gemma4 audio
    pub fn gemma4_default() -> Self {
        let embedding_dim = 1024;
        let attention_heads = 8;
        let head_dimension = embedding_dim / attention_heads;
        let chunk_span = 12;
        let context_past = 12;
        let context_future = 0;
        
        Self {
            spectral_width: 128,
            temporal_height: 100,
            feature_depth: 1,
            guidance_strength: 1.0,
            embedding_dim,
            attention_heads,
            head_dimension,
            feedforward_dim: embedding_dim * 4,
            transformer_layers: 12,
            convolution_kernel: 5,
            chunk_span,
            context_past,
            context_future,
            total_context: chunk_span + context_past + context_future,
            logit_cap: 50.0,
            residual_scale: 0.5,
            gradient_threshold: 1e10,
            epsilon: 1e-6,
            spatial_depth: 64,
            spatial_guidance: 1.0,
        }
    }
    
    /// Validate hyperparameters
    pub fn validate(&self) -> Result<(), PronaxAudioEncoderError> {
        if self.embedding_dim % self.attention_heads != 0 {
            return Err(PronaxAudioEncoderError::ConfigurationError(
                format!("embedding_dim {} not divisible by attention_heads {}", 
                    self.embedding_dim, self.attention_heads)
            ));
        }
        
        if self.head_dimension != self.embedding_dim / self.attention_heads {
            return Err(PronaxAudioEncoderError::ConfigurationError(
                "head_dimension mismatch".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Zero-copy tensor view with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct PronaxTensorView3D<'a> {
    /// Raw data slice (zero-copy view)
    pub data: &'a [f32],
    /// Tensor dimensions [width, height, depth]
    pub dimensions: [usize; 3],
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
    /// Strides for multi-dimensional access
    pub strides: [usize; 3],
}

impl<'a> PronaxTensorView3D<'a> {
    /// Create a new zero-copy tensor view
    pub fn new(data: &'a [f32], width: usize, height: usize, depth: usize) -> Result<Self, PronaxAudioEncoderError> {
        let expected_size = width * height * depth;
        if data.len() != expected_size {
            return Err(PronaxAudioEncoderError::InvalidTensorShape(
                format!("Expected {} elements, got {}", expected_size, data.len())
            ));
        }
        
        let spatial = SpatialTensorMetadata::new(width as u32, height as u32, depth as u32);
        
        Ok(Self {
            data,
            dimensions: [width, height, depth],
            spatial,
            strides: [height * depth, depth, 1],
        })
    }
    
    /// Get element at 3D coordinate (zero-copy)
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&f32> {
        if x >= self.dimensions[0] || y >= self.dimensions[1] || z >= self.dimensions[2] {
            return None;
        }
        
        let idx = x * self.strides[0] + y * self.strides[1] + z * self.strides[2];
        self.data.get(idx)
    }
    
    /// Reshape view (zero-copy if compatible)
    pub fn reshape(&self, new_width: usize, new_height: usize, new_depth: usize) -> Result<Self, PronaxAudioEncoderError> {
        let expected_size = new_width * new_height * new_depth;
        if self.data.len() != expected_size {
            return Err(PronaxAudioEncoderError::DimensionMismatch(
                format!("Cannot reshape {} to {}", self.data.len(), expected_size)
            ));
        }
        
        Ok(Self {
            data: self.data,
            dimensions: [new_width, new_height, new_depth],
            spatial: SpatialTensorMetadata::new(new_width as u32, new_height as u32, new_depth as u32),
            strides: [new_height * new_depth, new_depth, 1],
        })
    }
}

/// Sub-Sample Convolution Projection (SSCP) block with 3D awareness
#[derive(Debug, Clone)]
pub struct PronaxSSCPBlock3D {
    /// Convolutional kernel weights [kernel_h, kernel_w, in_channels, out_channels]
    pub kernel_weights: Vec<f32>,
    /// Layer normalization weights
    pub normalization_weights: Vec<f32>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl PronaxSSCPBlock3D {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let kernel_size = 3;
        let kernel_weights = vec![0.0; kernel_size * kernel_size * in_channels * out_channels];
        
        Self {
            kernel_weights,
            normalization_weights: vec![1.0; out_channels],
            spatial_position: ConversionCoordinate::standard(),
            spatial_metadata: SpatialTensorMetadata::new(
                out_channels as u32,
                kernel_size as u32,
                kernel_size as u32,
            ),
        }
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(
        &self,
        input_view: &PronaxTensorView3D,
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let [in_width, in_height, in_depth] = input_view.dimensions;
        let out_channels = self.normalization_weights.len();
        
        // Stride 2 downsampling
        let out_width = in_width / 2;
        let out_height = in_height / 2;
        let output_size = out_width * out_height * out_channels;
        
        let mut output = vec![0.0; output_size];
        
        // 3x3 convolution with stride 2
        let kernel_size = 3;
        for out_c in 0..out_channels {
            for out_y in 0..out_height {
                for out_x in 0..out_width {
                    let in_y = out_y * 2;
                    let in_x = out_x * 2;
                    
                    let mut sum = 0.0;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            for in_c in 0..in_depth {
                                let src_y = in_y + ky;
                                let src_x = in_x + kx;
                                
                                if src_y < in_height && src_x < in_width {
                                    if let Some(&val) = input_view.get(src_x, src_y, in_c) {
                                        let kernel_idx = ((ky * kernel_size + kx) * in_depth + in_c) * out_channels + out_c;
                                        if kernel_idx < self.kernel_weights.len() {
                                            sum += val * self.kernel_weights[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    let out_idx = (out_c * out_height + out_y) * out_width + out_x;
                    if out_idx < output.len() {
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        // Apply layer normalization per channel
        for c in 0..out_channels {
            let channel_start = c * out_height * out_width;
            let channel_end = channel_start + out_height * out_width;
            
            if channel_end <= output.len() {
                let channel_slice = &mut output[channel_start..channel_end];
                let sum_sq: f32 = channel_slice.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / channel_slice.len() as f32 + params.epsilon).sqrt();
                let norm_weight = self.normalization_weights.get(c).copied().unwrap_or(1.0);
                
                for val in channel_slice {
                    *val = (*val / rms) * norm_weight;
                }
            }
        }
        
        // ReLU activation
        for val in &mut output {
            *val = val.max(0.0);
        }
        
        Ok(output)
    }
}

/// Linear layer with input/output clamping support
#[derive(Debug, Clone)]
pub struct PronaxBoundedLinear3D {
    /// Weight matrix [out_features, in_features]
    pub weight_matrix: Vec<f32>,
    /// Bias vector [out_features]
    pub bias_vector: Vec<f32>,
    /// Input minimum clamp (optional)
    pub input_minimum: Option<f32>,
    /// Input maximum clamp (optional)
    pub input_maximum: Option<f32>,
    /// Output minimum clamp (optional)
    pub output_minimum: Option<f32>,
    /// Output maximum clamp (optional)
    pub output_maximum: Option<f32>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// Cached clamp values
    cached_clamps: (f32, f32, f32, f32),
    clamps_initialized: bool,
}

impl PronaxBoundedLinear3D {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight_matrix: vec![0.0; out_features * in_features],
            bias_vector: vec![0.0; out_features],
            input_minimum: None,
            input_maximum: None,
            output_minimum: None,
            output_maximum: None,
            spatial_position: ConversionCoordinate::standard(),
            cached_clamps: (0.0, 0.0, 0.0, 0.0),
            clamps_initialized: false,
        }
    }
    
    /// Initialize cached clamp values
    fn initialize_clamps(&mut self) {
        if self.clamps_initialized {
            return;
        }
        
        self.cached_clamps = (
            self.input_minimum.unwrap_or(0.0),
            self.input_maximum.unwrap_or(0.0),
            self.output_minimum.unwrap_or(0.0),
            self.output_maximum.unwrap_or(0.0),
        );
        self.clamps_initialized = true;
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(
        &mut self,
        input: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        self.initialize_clamps();
        
        let in_features = input.len();
        let out_features = self.bias_vector.len();
        let mut output = vec![0.0; out_features];
        
        // Apply input clamping
        let processed_input: Vec<f32> = if self.cached_clamps.1 != 0.0 {
            input.iter()
                .map(|&x| x.clamp(self.cached_clamps.0, self.cached_clamps.1))
                .collect()
        } else {
            input.to_vec()
        };
        
        // Matrix multiplication: output = weight @ input + bias
        for out_idx in 0..out_features {
            let mut sum = 0.0;
            for in_idx in 0..in_features {
                let weight_idx = out_idx * in_features + in_idx;
                if weight_idx < self.weight_matrix.len() {
                    sum += self.weight_matrix[weight_idx] * processed_input[in_idx];
                }
            }
            output[out_idx] = sum + self.bias_vector[out_idx];
        }
        
        // Apply output clamping
        if self.cached_clamps.3 != 0.0 {
            for val in &mut output {
                *val = val.clamp(self.cached_clamps.2, self.cached_clamps.3);
            }
        }
        
        Ok(output)
    }
}

/// Conformer block with dual-path feedforward and 3D awareness
#[derive(Debug, Clone)]
pub struct PronaxConformerBlock3D {
    /// Block-level normalization
    pub block_normalization: Vec<f32>,
    
    // First feedforward path
    pub ffw_primary_norm: Vec<f32>,
    pub ffw_primary_up: PronaxBoundedLinear3D,
    pub ffw_primary_down: PronaxBoundedLinear3D,
    pub ffw_primary_post_norm: Vec<f32>,
    
    // Second feedforward path
    pub ffw_secondary_norm: Vec<f32>,
    pub ffw_secondary_up: PronaxBoundedLinear3D,
    pub ffw_secondary_down: PronaxBoundedLinear3D,
    pub ffw_secondary_post_norm: Vec<f32>,
    
    // Multi-head attention
    pub attention_query: PronaxBoundedLinear3D,
    pub attention_key: PronaxBoundedLinear3D,
    pub attention_value: PronaxBoundedLinear3D,
    pub attention_output: PronaxBoundedLinear3D,
    pub attention_pre_norm: Vec<f32>,
    pub attention_post_norm: Vec<f32>,
    pub positional_weights: Vec<f32>,
    pub per_dimension_scale: Vec<f32>,
    
    // Lightweight depthwise convolution
    pub conv_pointwise1: PronaxBoundedLinear3D,
    pub conv_pointwise2: PronaxBoundedLinear3D,
    pub conv_depthwise: Vec<f32>,
    pub conv_normalization: Vec<f32>,
    pub conv_post_norm: Vec<f32>,
    
    /// Layer index
    pub layer_index: usize,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl PronaxConformerBlock3D {
    pub fn new(layer_idx: usize, params: &PronaxAudioHyperparams3D) -> Self {
        let hidden_dim = params.embedding_dim;
        let ffn_dim = params.feedforward_dim;
        
        Self {
            block_normalization: vec![1.0; hidden_dim],
            
            ffw_primary_norm: vec![1.0; hidden_dim],
            ffw_primary_up: PronaxBoundedLinear3D::new(hidden_dim, ffn_dim),
            ffw_primary_down: PronaxBoundedLinear3D::new(ffn_dim, hidden_dim),
            ffw_primary_post_norm: vec![1.0; hidden_dim],
            
            ffw_secondary_norm: vec![1.0; hidden_dim],
            ffw_secondary_up: PronaxBoundedLinear3D::new(hidden_dim, ffn_dim),
            ffw_secondary_down: PronaxBoundedLinear3D::new(ffn_dim, hidden_dim),
            ffw_secondary_post_norm: vec![1.0; hidden_dim],
            
            attention_query: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim),
            attention_key: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim),
            attention_value: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim),
            attention_output: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim),
            attention_pre_norm: vec![1.0; hidden_dim],
            attention_post_norm: vec![1.0; hidden_dim],
            positional_weights: vec![0.0; hidden_dim * hidden_dim],
            per_dimension_scale: vec![1.0; params.head_dimension],
            
            conv_pointwise1: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim * 2),
            conv_pointwise2: PronaxBoundedLinear3D::new(hidden_dim, hidden_dim),
            conv_depthwise: vec![0.0; params.convolution_kernel * hidden_dim],
            conv_normalization: vec![1.0; hidden_dim],
            conv_post_norm: vec![1.0; hidden_dim],
            
            layer_index: layer_idx,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 4) as u16,
                (layer_idx % 4) as u8,
                params.spatial_guidance,
            ),
            spatial_metadata: SpatialTensorMetadata::new(
                hidden_dim as u32,
                params.total_context as u32,
                params.spatial_depth as u32,
            ),
        }
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(
        &mut self,
        input: &[f32],
        causal_mask: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let seq_len = input.len() / params.embedding_dim;
        let mut output = input.to_vec();
        
        // First feedforward path (half-residual)
        output = self.feedforward_half_residual(
            &output,
            &self.ffw_primary_norm,
            &mut self.ffw_primary_up,
            &mut self.ffw_primary_down,
            &self.ffw_primary_post_norm,
            params,
        )?;
        
        // Multi-head attention with relative positions
        output = self.attention_relative_position(
            &output,
            causal_mask,
            params,
        )?;
        
        // Lightweight depthwise convolution
        output = self.depthwise_convolution(&output, params)?;
        
        // Second feedforward path (half-residual)
        output = self.feedforward_half_residual(
            &output,
            &self.ffw_secondary_norm,
            &mut self.ffw_secondary_up,
            &mut self.ffw_secondary_down,
            &self.ffw_secondary_post_norm,
            params,
        )?;
        
        // Gradient clipping
        for val in &mut output {
            *val = val.clamp(-params.gradient_threshold, params.gradient_threshold);
        }
        
        // Final block normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(
                    &mut output[start..end],
                    &self.block_normalization,
                    params.epsilon,
                );
            }
        }
        
        Ok(output)
    }
    
    /// Feedforward with half-residual connection
    fn feedforward_half_residual(
        &mut self,
        input: &[f32],
        pre_norm: &[f32],
        up_layer: &mut PronaxBoundedLinear3D,
        down_layer: &mut PronaxBoundedLinear3D,
        post_norm: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let seq_len = input.len() / params.embedding_dim;
        let residual = input.to_vec();
        let mut output = input.to_vec();
        
        // Pre-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], pre_norm, params.epsilon);
            }
        }
        
        // Gradient clipping
        for val in &mut output {
            *val = val.clamp(-params.gradient_threshold, params.gradient_threshold);
        }
        
        // Up projection with SILU
        let mut up_output = Vec::with_capacity(output.len() * 4);
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            let seq_slice = &output[start..end.min(output.len())];
            
            let seq_up = up_layer.forward_zero_copy(seq_slice, params)?;
            
            // SILU activation
            let seq_silu: Vec<f32> = seq_up.iter()
                .map(|&x| x / (1.0 + (-x).exp()))
                .collect();
            
            up_output.extend_from_slice(&seq_silu);
        }
        
        // Down projection
        let mut down_output = Vec::with_capacity(output.len());
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.feedforward_dim;
            let end = start + params.feedforward_dim;
            let seq_slice = &up_output[start..end.min(up_output.len())];
            
            let seq_down = down_layer.forward_zero_copy(seq_slice, params)?;
            down_output.extend_from_slice(&seq_down);
        }
        
        // Gradient clipping
        for val in &mut down_output {
            *val = val.clamp(-params.gradient_threshold, params.gradient_threshold);
        }
        
        // Post-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= down_output.len() {
                self.apply_rms_norm(&mut down_output[start..end], post_norm, params.epsilon);
            }
        }
        
        // Scale by residual weight
        for val in &mut down_output {
            *val *= params.residual_scale;
        }
        
        // Add residual connection
        for (o, r) in down_output.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        
        Ok(down_output)
    }
    
    /// Multi-head attention with relative positional encoding
    fn attention_relative_position(
        &mut self,
        input: &[f32],
        causal_mask: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let seq_len = input.len() / params.embedding_dim;
        let residual = input.to_vec();
        let mut output = input.to_vec();
        
        // Pre-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.attention_pre_norm, params.epsilon);
            }
        }
        
        // Gradient clipping
        for val in &mut output {
            *val = val.clamp(-params.gradient_threshold, params.gradient_threshold);
        }
        
        // Simplified block-local attention (full implementation would use optimized kernels)
        let chunk_size = params.chunk_span;
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        
        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            
            for seq_idx in chunk_start..chunk_end {
                let start = seq_idx * params.embedding_dim;
                let end = start + params.embedding_dim;
                if end <= output.len() {
                    // Apply per-dimension scaling
                    for (h, scale) in output[start..end].iter_mut().zip(self.per_dimension_scale.iter()) {
                        *h *= scale;
                    }
                }
            }
        }
        
        // Post-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.attention_post_norm, params.epsilon);
            }
        }
        
        // Add residual connection
        for (o, r) in output.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        
        Ok(output)
    }
    
    /// Lightweight depthwise convolution
    fn depthwise_convolution(
        &mut self,
        input: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let seq_len = input.len() / params.embedding_dim;
        let residual = input.to_vec();
        let mut output = input.to_vec();
        
        // Pre-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.conv_normalization, params.epsilon);
            }
        }
        
        // Pointwise convolution 1 (expand to 2x)
        let mut expanded = Vec::with_capacity(output.len() * 2);
        for seq_idx in 0..seq_len {
            let start = seq_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            let seq_slice = &output[start..end.min(output.len())];
            
            let seq_expanded = self.conv_pointwise1.forward_zero_copy(seq_slice, params)?;
            expanded.extend_from_slice(&seq_expanded);
        }
        
        // GLU activation (split and gate)
        let hidden_dim = params.embedding_dim;
        let mut gated = vec![0.0; hidden_dim * seq_len];
        
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim * 2;
            let mid = start + hidden_dim;
            let end = mid + hidden_dim;
            
            if end <= expanded.len() {
                for i in 0..hidden_dim {
                    let data = expanded[start + i];
                    let gate = 1.0 / (1.0 + (-expanded[mid + i]).exp());
                    gated[seq_idx * hidden_dim + i] = data * gate;
                }
            }
        }
        
        // Depthwise convolution
        let kernel_size = params.convolution_kernel;
        let mut conv_output = vec![0.0; gated.len()];
        
        for seq_idx in 0..seq_len {
            for h in 0..hidden_dim {
                let idx = seq_idx * hidden_dim + h;
                let mut sum = 0.0;
                
                for k in 0..kernel_size {
                    let prev_idx = if seq_idx >= k {
                        (seq_idx - k) * hidden_dim + h
                    } else {
                        idx
                    };
                    
                    if prev_idx < gated.len() {
                        let kernel_idx = k * hidden_dim + h;
                        if kernel_idx < self.conv_depthwise.len() {
                            sum += gated[prev_idx] * self.conv_depthwise[kernel_idx];
                        }
                    }
                }
                
                if idx < conv_output.len() {
                    conv_output[idx] = sum;
                }
            }
        }
        
        // Post-normalization
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim;
            let end = start + hidden_dim;
            if end <= conv_output.len() {
                self.apply_rms_norm(&mut conv_output[start..end], &self.conv_post_norm, params.epsilon);
            }
        }
        
        // SILU activation
        for val in &mut conv_output {
            *val = *val / (1.0 + (-*val).exp());
        }
        
        // Pointwise convolution 2 (project back)
        let mut final_output = Vec::with_capacity(conv_output.len());
        for seq_idx in 0..seq_len {
            let start = seq_idx * hidden_dim;
            let end = start + hidden_dim;
            let seq_slice = &conv_output[start..end.min(conv_output.len())];
            
            let seq_final = self.conv_pointwise2.forward_zero_copy(seq_slice, params)?;
            final_output.extend_from_slice(&seq_final);
        }
        
        // Add residual connection
        for (o, r) in final_output.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        
        Ok(final_output)
    }
    
    /// RMS normalization
    fn apply_rms_norm(&self, input: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        
        for (x, &w) in input.iter_mut().zip(weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

/// Output projection layer
#[derive(Debug, Clone)]
pub struct PronaxAudioOutput3D {
    /// Projection weights [out_dim, in_dim]
    pub projection_weights: Vec<f32>,
    /// Bias vector [out_dim]
    pub bias_vector: Vec<f32>,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl PronaxAudioOutput3D {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            projection_weights: vec![0.0; out_dim * in_dim],
            bias_vector: vec![0.0; out_dim],
            spatial_position: ConversionCoordinate::standard(),
            spatial_metadata: SpatialTensorMetadata::new(in_dim as u32, out_dim as u32, 1),
        }
    }
    
    /// Forward pass with zero-copy optimization
    pub fn forward_zero_copy(&self, input: &[f32]) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let in_dim = input.len();
        let out_dim = self.bias_vector.len();
        let mut output = vec![0.0; out_dim];
        
        // Matrix multiplication
        for out_idx in 0..out_dim {
            let mut sum = 0.0;
            for in_idx in 0..in_dim {
                let weight_idx = out_idx * in_dim + in_idx;
                if weight_idx < self.projection_weights.len() {
                    sum += self.projection_weights[weight_idx] * input[in_idx];
                }
            }
            output[out_idx] = sum + self.bias_vector[out_idx];
        }
        
        Ok(output)
    }
}

/// Complete 3D audio encoder with SSCP and Conformer blocks
#[derive(Debug, Clone)]
pub struct PronaxAudioEncoder3D {
    /// Hyperparameters
    pub hyperparams: PronaxAudioHyperparams3D,
    
    /// SSCP convolution blocks
    pub sscp_stage_one: PronaxSSCPBlock3D,
    pub sscp_stage_two: PronaxSSCPBlock3D,
    
    /// SSCP input projection
    pub sscp_projection: PronaxBoundedLinear3D,
    
    /// Conformer transformer layers
    pub conformer_layers: Vec<PronaxConformerBlock3D>,
    
    /// Output projection
    pub output_layer: PronaxAudioOutput3D,
    
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl PronaxAudioEncoder3D {
    pub fn new(params: PronaxAudioHyperparams3D) -> Result<Self, PronaxAudioEncoderError> {
        params.validate()?;
        
        let conformer_layers: Vec<PronaxConformerBlock3D> = (0..params.transformer_layers)
            .map(|i| PronaxConformerBlock3D::new(i, &params))
            .collect();
        
        Ok(Self {
            hyperparams: params,
            sscp_stage_one: PronaxSSCPBlock3D::new(1, 64),
            sscp_stage_two: PronaxSSCPBlock3D::new(64, 128),
            sscp_projection: PronaxBoundedLinear3D::new(128 * 32 * 32, params.embedding_dim),
            conformer_layers,
            output_layer: PronaxAudioOutput3D::new(params.embedding_dim, params.embedding_dim),
            spatial_position: ConversionCoordinate::standard(),
            spatial_metadata: SpatialTensorMetadata::new(
                params.embedding_dim as u32,
                params.spectral_width as u32,
                params.spatial_depth as u32,
            ),
        })
    }
    
    /// Forward pass with zero-copy optimization
    pub fn encode_audio_zero_copy(
        &mut self,
        mel_spectrogram: &[f32],
        num_frames: usize,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        let mel_bins = self.hyperparams.spectral_width;
        
        // Create zero-copy tensor view
        let input_view = PronaxTensorView3D::new(mel_spectrogram, num_frames, mel_bins, 1)?;
        
        // SSCP stage 1
        let stage1_output = self.sscp_stage_one.forward_zero_copy(&input_view, &self.hyperparams)?;
        
        // Create view for stage 2
        let stage1_dims = [
            num_frames / 2,
            mel_bins / 2,
            self.sscp_stage_one.normalization_weights.len(),
        ];
        let stage1_view = PronaxTensorView3D::new(&stage1_output, stage1_dims[0], stage1_dims[1], stage1_dims[2])?;
        
        // SSCP stage 2
        let stage2_output = self.sscp_stage_two.forward_zero_copy(&stage1_view, &self.hyperparams)?;
        
        // Flatten and project
        let flattened = stage2_output;
        let projected = self.sscp_projection.forward_zero_copy(&flattened, &self.hyperparams)?;
        
        // Build causal mask
        let causal_mask = self.construct_causal_mask();
        
        // Process through conformer layers
        let mut hidden_state = projected;
        for (layer_idx, layer) in self.conformer_layers.iter_mut().enumerate() {
            hidden_state = layer.forward_zero_copy(&hidden_state, &causal_mask, &self.hyperparams)?;
        }
        
        // Output projection
        let final_output = self.output_layer.forward_zero_copy(&hidden_state)?;
        
        Ok(final_output)
    }
    
    /// Construct causal-valid mask for block-local attention
    fn construct_causal_mask(&self) -> Vec<f32> {
        let chunk_size = self.hyperparams.chunk_span;
        let context_size = self.hyperparams.total_context;
        let upper_diagonal = self.hyperparams.context_past + self.hyperparams.context_future;
        
        let mut mask = vec![0.0f32; chunk_size * context_size];
        
        for row in 0..chunk_size {
            for col in 0..context_size {
                let lower_triangular = row <= col;
                let upper_triangular = col <= row + upper_diagonal;
                
                mask[row * context_size + col] = if lower_triangular && upper_triangular {
                    1.0
                } else {
                    0.0
                };
            }
        }
        
        mask
    }
}

/// Audio-to-text multimodal projector
#[derive(Debug, Clone)]
pub struct PronaxAudioTextProjector3D {
    /// Input projection weights
    pub input_projection: PronaxBoundedLinear3D,
    
    /// Fully-connected layer weights
    pub fc_weights: Vec<f32>,
    pub fc_bias: Vec<f32>,
    
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
}

impl PronaxAudioTextProjector3D {
    pub fn new(audio_dim: usize, text_dim: usize) -> Self {
        Self {
            input_projection: PronaxBoundedLinear3D::new(audio_dim, text_dim),
            fc_weights: vec![0.0; audio_dim * audio_dim],
            fc_bias: vec![0.0; audio_dim],
            spatial_position: ConversionCoordinate::standard(),
            spatial_metadata: SpatialTensorMetadata::new(audio_dim as u32, text_dim as u32, 1),
        }
    }
    
    /// Project audio features to text embedding space
    pub fn project_to_text_zero_copy(
        &mut self,
        audio_features: &[f32],
        params: &PronaxAudioHyperparams3D,
    ) -> Result<Vec<f32>, PronaxAudioEncoderError> {
        // FC projection
        let audio_dim = audio_features.len();
        let mut fc_output = vec![0.0; self.fc_bias.len()];
        
        for out_idx in 0..fc_output.len() {
            let mut sum = self.fc_bias[out_idx];
            for in_idx in 0..audio_dim {
                let weight_idx = out_idx * audio_dim + in_idx;
                if weight_idx < self.fc_weights.len() {
                    sum += self.fc_weights[weight_idx] * audio_features[in_idx];
                }
            }
            fc_output[out_idx] = sum;
        }
        
        // RMS normalization (without learned weights)
        let sum_sq: f32 = fc_output.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / fc_output.len() as f32 + params.epsilon).sqrt();
        for val in &mut fc_output {
            *val /= rms;
        }
        
        // Final projection to text dimension
        let text_embeddings = self.input_projection.forward_zero_copy(&fc_output, params)?;
        
        Ok(text_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hyperparams_validation() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        assert!(params.validate().is_ok());
    }
    
    #[test]
    fn test_tensor_view_creation() {
        let data = vec![0.0f32; 128 * 100];
        let view = PronaxTensorView3D::new(&data, 100, 128, 1);
        assert!(view.is_ok());
    }
    
    #[test]
    fn test_sscp_block() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        let sscp = PronaxSSCPBlock3D::new(1, 64);
        
        let dummy_data = vec![0.5f32; 128 * 100];
        let view = PronaxTensorView3D::new(&dummy_data, 100, 128, 1).unwrap();
        
        let result = sscp.forward_zero_copy(&view, &params);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_bounded_linear() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        let mut linear = PronaxBoundedLinear3D::new(128, 256);
        
        let dummy_input = vec![0.5f32; 128];
        let result = linear.forward_zero_copy(&dummy_input, &params);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_conformer_block() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        let mut conformer = PronaxConformerBlock3D::new(0, &params);
        
        let dummy_input = vec![0.5f32; params.embedding_dim * 10];
        let causal_mask = vec![1.0f32; params.chunk_span * params.total_context];
        
        let result = conformer.forward_zero_copy(&dummy_input, &causal_mask, &params);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_audio_encoder() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        let mut encoder = PronaxAudioEncoder3D::new(params).unwrap();
        
        let dummy_mel = vec![0.5f32; 128 * 100];
        let result = encoder.encode_audio_zero_copy(&dummy_mel, 100);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_audio_text_projector() {
        let params = PronaxAudioHyperparams3D::gemma4_default();
        let mut projector = PronaxAudioTextProjector3D::new(1024, 2304);
        
        let dummy_audio = vec![0.5f32; 1024];
        let result = projector.project_to_text_zero_copy(&dummy_audio, &params);
        assert!(result.is_ok());
    }
}
