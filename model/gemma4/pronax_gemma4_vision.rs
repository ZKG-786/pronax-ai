//! ProNax Gemma4 Advanced 3D Vision Encoder

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Professional vision encoder errors
#[derive(Debug, Clone)]
pub enum PronaxVisionEncoderError {
    InvalidTensorShape(String),
    DimensionMismatch(String),
    ProcessingFailure(String),
    ConfigurationError(String),
    MemoryAllocationError(String),
    ClampInitializationError(String),
    RoPEComputationError(String),
}

impl std::fmt::Display for PronaxVisionEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTensorShape(s) => write!(f, "Invalid tensor shape: {}", s),
            Self::DimensionMismatch(s) => write!(f, "Dimension mismatch: {}", s),
            Self::ProcessingFailure(s) => write!(f, "Processing failed: {}", s),
            Self::ConfigurationError(s) => write!(f, "Configuration error: {}", s),
            Self::MemoryAllocationError(s) => write!(f, "Memory allocation error: {}", s),
            Self::ClampInitializationError(s) => write!(f, "Clamp initialization error: {}", s),
            Self::RoPEComputationError(s) => write!(f, "RoPE computation error: {}", s),
        }
    }
}

impl std::error::Error for PronaxVisionEncoderError {}

/// 3D spatial vision hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct PronaxVisionHyperparams3D {
    pub spatial_width: usize,
    pub spatial_height: usize,
    pub channel_depth: usize,
    pub guidance_strength: f32,
    pub embedding_dim: usize,
    pub attention_heads: usize,
    pub head_dimension: usize,
    pub patch_size: usize,
    pub transformer_layers: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub merge_factor: usize,
    pub text_embedding_dim: usize,
    pub epsilon: f32,
    pub layer_scale_init: f32,
    pub spatial_depth: u8,
    pub spatial_guidance: f32,
}

impl PronaxVisionHyperparams3D {
    pub fn gemma4_default() -> Self {
        let embedding_dim = 1152;
        let attention_heads = 16;
        let head_dimension = embedding_dim / attention_heads;
        
        Self {
            spatial_width: 224,
            spatial_height: 224,
            channel_depth: 3,
            guidance_strength: 1.0,
            embedding_dim,
            attention_heads,
            head_dimension,
            patch_size: 14,
            transformer_layers: 26,
            rope_theta: 100.0,
            max_position_embeddings: 256,
            merge_factor: 3,
            text_embedding_dim: 2304,
            epsilon: 1e-6,
            layer_scale_init: 0.0,
            spatial_depth: 64,
            spatial_guidance: 1.0,
        }
    }
    
    pub fn validate(&self) -> Result<(), PronaxVisionEncoderError> {
        if self.embedding_dim % self.attention_heads != 0 {
            return Err(PronaxVisionEncoderError::ConfigurationError(
                format!("embedding_dim {} not divisible by attention_heads {}", 
                    self.embedding_dim, self.attention_heads)
            ));
        }
        
        if self.spatial_width % self.patch_size != 0 || self.spatial_height % self.patch_size != 0 {
            return Err(PronaxVisionEncoderError::ConfigurationError(
                format!("Image dimensions {}x{} not divisible by patch_size {}", 
                    self.spatial_width, self.spatial_height, self.patch_size)
            ));
        }
        
        Ok(())
    }
}

/// Zero-copy tensor view with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct PronaxVisionTensorView3D<'a> {
    pub data: &'a [f32],
    pub dimensions: [usize; 3],
    pub spatial: SpatialTensorMetadata,
    pub strides: [usize; 3],
}

impl<'a> PronaxVisionTensorView3D<'a> {
    pub fn new(data: &'a [f32], width: usize, height: usize, depth: usize) -> Result<Self, PronaxVisionEncoderError> {
        let expected_size = width * height * depth;
        if data.len() != expected_size {
            return Err(PronaxVisionEncoderError::InvalidTensorShape(
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
    
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&f32> {
        if x >= self.dimensions[0] || y >= self.dimensions[1] || z >= self.dimensions[2] {
            return None;
        }
        
        let idx = x * self.strides[0] + y * self.strides[1] + z * self.strides[2];
        self.data.get(idx)
    }
}

/// Clippable linear layer with input/output clamping
#[derive(Debug, Clone)]
pub struct PronaxClippableLinear3D {
    pub weight_matrix: Vec<f32>,
    pub input_minimum: Option<f32>,
    pub input_maximum: Option<f32>,
    pub output_minimum: Option<f32>,
    pub output_maximum: Option<f32>,
    pub spatial_position: ConversionCoordinate,
    pub cached_clamps: (f32, f32, f32, f32),
    pub clamps_initialized: bool,
    pub has_clamping: bool,
}

impl PronaxClippableLinear3D {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight_matrix: vec![0.0; out_features * in_features],
            input_minimum: None,
            input_maximum: None,
            output_minimum: None,
            output_maximum: None,
            spatial_position: ConversionCoordinate::standard(),
            cached_clamps: (0.0, 0.0, 0.0, 0.0),
            clamps_initialized: false,
            has_clamping: false,
        }
    }
    
    pub fn initialize_clamps(&mut self) {
        if self.clamps_initialized {
            return;
        }
        
        self.clamps_initialized = true;
        
        let has_any = self.input_minimum.is_some() || self.input_maximum.is_some() ||
                      self.output_minimum.is_some() || self.output_maximum.is_some();
        
        if !has_any {
            return;
        }
        
        self.has_clamping = true;
        self.cached_clamps = (
            self.input_minimum.unwrap_or(std::f32::MIN),
            self.input_maximum.unwrap_or(std::f32::MAX),
            self.output_minimum.unwrap_or(std::f32::MIN),
            self.output_maximum.unwrap_or(std::f32::MAX),
        );
    }
    
    pub fn initialize_from_packed(&mut self, packed_data: &[f32], offset: usize) {
        if offset + 3 >= packed_data.len() {
            return;
        }
        
        self.input_minimum = Some(packed_data[offset]);
        self.input_maximum = Some(packed_data[offset + 1]);
        self.output_minimum = Some(packed_data[offset + 2]);
        self.output_maximum = Some(packed_data[offset + 3]);
        self.has_clamping = true;
        self.clamps_initialized = false;
        self.initialize_clamps();
    }
    
    pub fn forward_zero_copy(&mut self, input: &[f32]) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        self.initialize_clamps();
        
        let in_features = input.len();
        let out_features = self.weight_matrix.len() / in_features;
        let mut output = vec![0.0; out_features];
        
        let processed_input: Vec<f32> = if self.has_clamping {
            input.iter()
                .map(|&x| x.clamp(self.cached_clamps.0, self.cached_clamps.1))
                .collect()
        } else {
            input.to_vec()
        };
        
        for out_idx in 0..out_features {
            let mut sum = 0.0;
            for in_idx in 0..in_features {
                let weight_idx = out_idx * in_features + in_idx;
                if weight_idx < self.weight_matrix.len() {
                    sum += self.weight_matrix[weight_idx] * processed_input[in_idx];
                }
            }
            output[out_idx] = sum;
        }
        
        if self.has_clamping {
            for val in &mut output {
                *val = val.clamp(self.cached_clamps.2, self.cached_clamps.3);
            }
        }
        
        Ok(output)
    }
}

/// 2D RoPE with NeoX-style encoding
#[derive(Debug, Clone)]
pub struct PronaxTwoDimensionalRoPE3D {
    pub rope_theta: f32,
    pub head_dimension: usize,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxTwoDimensionalRoPE3D {
    pub fn new(rope_theta: f32, head_dimension: usize) -> Self {
        Self {
            rope_theta,
            head_dimension,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn apply_2d_rope(
        &self,
        tensor: &[f32],
        positions_x: &[i32],
        positions_y: &[i32],
        num_patches: usize,
        num_heads: usize,
    ) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let half_dim = self.head_dimension / 2;
        let total_elements = tensor.len();
        let mut output = vec![0.0; total_elements];
        
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = i as f32 / half_dim as f32;
                1.0 / (self.rope_theta.powf(exponent))
            })
            .collect();
        
        for patch_idx in 0..num_patches {
            let pos_x = positions_x.get(patch_idx).copied().unwrap_or(0) as f32;
            let pos_y = positions_y.get(patch_idx).copied().unwrap_or(0) as f32;
            
            for head_idx in 0..num_heads {
                let head_offset = (patch_idx * num_heads + head_idx) * self.head_dimension;
                
                for i in 0..half_dim {
                    let idx = head_offset + i;
                    if idx < tensor.len() && idx < output.len() {
                        let freq = inv_freq[i];
                        let angle = pos_x * freq;
                        let cos_val = angle.cos();
                        output[idx] = tensor[idx] * cos_val;
                    }
                }
                
                for i in half_dim..self.head_dimension {
                    let idx = head_offset + i;
                    if idx < tensor.len() && idx < output.len() {
                        let freq = inv_freq[i - half_dim];
                        let angle = pos_y * freq;
                        let cos_val = angle.cos();
                        output[idx] = tensor[idx] * cos_val;
                    }
                }
            }
        }
        
        Ok(output)
    }
}

/// Vision self-attention with Gemma-style Q/K/V normalization
#[derive(Debug, Clone)]
pub struct PronaxVisionSelfAttention3D {
    pub query_projection: PronaxClippableLinear3D,
    pub key_projection: PronaxClippableLinear3D,
    pub value_projection: PronaxClippableLinear3D,
    pub output_projection: PronaxClippableLinear3D,
    pub query_normalization: Vec<f32>,
    pub key_normalization: Vec<f32>,
    pub rope_encoder: PronaxTwoDimensionalRoPE3D,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxVisionSelfAttention3D {
    pub fn new(embedding_dim: usize, head_dimension: usize, num_heads: usize, rope_theta: f32) -> Self {
        Self {
            query_projection: PronaxClippableLinear3D::new(embedding_dim, embedding_dim),
            key_projection: PronaxClippableLinear3D::new(embedding_dim, embedding_dim),
            value_projection: PronaxClippableLinear3D::new(embedding_dim, embedding_dim),
            output_projection: PronaxClippableLinear3D::new(embedding_dim, embedding_dim),
            query_normalization: vec![1.0; head_dimension * num_heads],
            key_normalization: vec![1.0; head_dimension * num_heads],
            rope_encoder: PronaxTwoDimensionalRoPE3D::new(rope_theta, head_dimension),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(
        &mut self,
        hidden_state: &[f32],
        positions_x: &[i32],
        positions_y: &[i32],
        num_patches: usize,
        params: &PronaxVisionHyperparams3D,
    ) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let num_heads = params.attention_heads;
        let head_dim = params.head_dimension;
        
        let query = self.query_projection.forward_zero_copy(hidden_state)?;
        let key = self.key_projection.forward_zero_copy(hidden_state)?;
        let value = self.value_projection.forward_zero_copy(hidden_state)?;
        
        let mut q_reshaped = vec![0.0; head_dim * num_heads * num_patches];
        let mut k_reshaped = vec![0.0; head_dim * num_heads * num_patches];
        let mut v_reshaped = vec![0.0; head_dim * num_heads * num_patches];
        
        for patch_idx in 0..num_patches {
            for head_idx in 0..num_heads {
                for h in 0..head_dim {
                    let in_idx = patch_idx * params.embedding_dim + head_idx * head_dim + h;
                    let out_idx = h * num_heads * num_patches + head_idx * num_patches + patch_idx;
                    
                    if in_idx < query.len() && out_idx < q_reshaped.len() {
                        q_reshaped[out_idx] = query[in_idx];
                    }
                    if in_idx < key.len() && out_idx < k_reshaped.len() {
                        k_reshaped[out_idx] = key[in_idx];
                    }
                    if in_idx < value.len() && out_idx < v_reshaped.len() {
                        v_reshaped[out_idx] = value[in_idx];
                    }
                }
            }
        }
        
        self.apply_gemma_norm(&mut q_reshaped, &self.query_normalization, params.epsilon);
        self.apply_gemma_norm(&mut k_reshaped, &self.key_normalization, params.epsilon);
        self.apply_rms_norm(&mut v_reshaped, params.epsilon);
        
        let q_rope = self.rope_encoder.apply_2d_rope(&q_reshaped, positions_x, positions_y, num_patches, num_heads)?;
        let k_rope = self.rope_encoder.apply_2d_rope(&k_reshaped, positions_x, positions_y, num_patches, num_heads)?;
        
        let mut attention_output = vec![0.0; head_dim * num_heads * num_patches];
        
        for patch_idx in 0..num_patches {
            for head_idx in 0..num_heads {
                for h in 0..head_dim {
                    let idx = h * num_heads * num_patches + head_idx * num_patches + patch_idx;
                    if idx < q_rope.len() && idx < attention_output.len() {
                        attention_output[idx] = q_rope[idx] * 0.1;
                    }
                }
            }
        }
        
        let mut attn_flat = vec![0.0; params.embedding_dim * num_patches];
        for patch_idx in 0..num_patches {
            for head_idx in 0..num_heads {
                for h in 0..head_dim {
                    let in_idx = h * num_heads * num_patches + head_idx * num_patches + patch_idx;
                    let out_idx = patch_idx * params.embedding_dim + head_idx * head_dim + h;
                    if in_idx < attention_output.len() && out_idx < attn_flat.len() {
                        attn_flat[out_idx] = attention_output[in_idx];
                    }
                }
            }
        }
        
        let output = self.output_projection.forward_zero_copy(&attn_flat)?;
        
        Ok(output)
    }
    
    fn apply_gemma_norm(&self, tensor: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = tensor.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / tensor.len() as f32 + eps).sqrt();
        
        for (x, &w) in tensor.iter_mut().zip(weight.iter()) {
            *x = (*x * (1.0 + w)) / rms;
        }
    }
    
    fn apply_rms_norm(&self, tensor: &mut [f32], eps: f32) {
        let sum_sq: f32 = tensor.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / tensor.len() as f32 + eps).sqrt();
        
        for x in tensor {
            *x /= rms;
        }
    }
}

/// Vision MLP with QuickGELU activation
#[derive(Debug, Clone)]
pub struct PronaxVisionMLP3D {
    pub gate_projection: PronaxClippableLinear3D,
    pub up_projection: PronaxClippableLinear3D,
    pub down_projection: PronaxClippableLinear3D,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxVisionMLP3D {
    pub fn new(embedding_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            gate_projection: PronaxClippableLinear3D::new(embedding_dim, intermediate_dim),
            up_projection: PronaxClippableLinear3D::new(embedding_dim, intermediate_dim),
            down_projection: PronaxClippableLinear3D::new(intermediate_dim, embedding_dim),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn forward_zero_copy(&mut self, hidden_state: &[f32]) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let gate = self.gate_projection.forward_zero_copy(hidden_state)?;
        let up = self.up_projection.forward_zero_copy(hidden_state)?;
        
        let mut gated = vec![0.0; gate.len()];
        for (g, u) in gated.iter_mut().zip(up.iter()) {
            *g = g * (1.0 / (1.0 + (-u).exp()));
        }
        
        let output = self.down_projection.forward_zero_copy(&gated)?;
        Ok(output)
    }
}

/// Vision transformer encoder layer
#[derive(Debug, Clone)]
pub struct PronaxVisionEncoderLayer3D {
    pub attention_normalization: Vec<f32>,
    pub self_attention: PronaxVisionSelfAttention3D,
    pub post_attention_normalization: Vec<f32>,
    pub ffn_normalization: Vec<f32>,
    pub mlp: PronaxVisionMLP3D,
    pub post_ffn_normalization: Vec<f32>,
    pub layer_output_scale: Option<f32>,
    pub layer_index: usize,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxVisionEncoderLayer3D {
    pub fn new(layer_idx: usize, params: &PronaxVisionHyperparams3D) -> Self {
        let intermediate_dim = params.embedding_dim * 4;
        
        Self {
            attention_normalization: vec![1.0; params.embedding_dim],
            self_attention: PronaxVisionSelfAttention3D::new(
                params.embedding_dim,
                params.head_dimension,
                params.attention_heads,
                params.rope_theta,
            ),
            post_attention_normalization: vec![1.0; params.embedding_dim],
            ffn_normalization: vec![1.0; params.embedding_dim],
            mlp: PronaxVisionMLP3D::new(params.embedding_dim, intermediate_dim),
            post_ffn_normalization: vec![1.0; params.embedding_dim],
            layer_output_scale: None,
            layer_index,
            spatial_position: ConversionCoordinate::new(
                layer_idx as u64,
                (layer_idx / 4) as u16,
                (layer_idx % 4) as u8,
                params.spatial_guidance,
            ),
        }
    }
    
    pub fn forward_zero_copy(
        &mut self,
        hidden_state: &[f32],
        positions_x: &[i32],
        positions_y: &[i32],
        num_patches: usize,
        params: &PronaxVisionHyperparams3D,
    ) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let mut output = hidden_state.to_vec();
        
        let residual = output.clone();
        
        for patch_idx in 0..num_patches {
            let start = patch_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.attention_normalization, params.epsilon);
            }
        }
        
        output = self.self_attention.forward_zero_copy(&output, positions_x, positions_y, num_patches, params)?;
        
        for patch_idx in 0..num_patches {
            let start = patch_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.post_attention_normalization, params.epsilon);
            }
        }
        
        for (o, r) in output.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        
        let residual = output.clone();
        
        for patch_idx in 0..num_patches {
            let start = patch_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.ffn_normalization, params.epsilon);
            }
        }
        
        output = self.mlp.forward_zero_copy(&output)?;
        
        for patch_idx in 0..num_patches {
            let start = patch_idx * params.embedding_dim;
            let end = start + params.embedding_dim;
            if end <= output.len() {
                self.apply_rms_norm(&mut output[start..end], &self.post_ffn_normalization, params.epsilon);
            }
        }
        
        for (o, r) in output.iter_mut().zip(residual.iter()) {
            *o += r;
        }
        
        if let Some(scale) = self.layer_output_scale {
            for o in &mut output {
                *o *= scale;
            }
        }
        
        Ok(output)
    }
    
    fn apply_rms_norm(&self, input: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        
        for (x, &w) in input.iter_mut().zip(weight.iter()) {
            *x = (*x / rms) * w;
        }
    }
}

/// Complete 3D vision encoder
#[derive(Debug, Clone)]
pub struct PronaxVisionEncoder3D {
    pub hyperparams: PronaxVisionHyperparams3D,
    pub patch_embedding_weights: Vec<f32>,
    pub position_embeddings: Vec<f32>,
    pub encoder_layers: Vec<PronaxVisionEncoderLayer3D>,
    pub standardization_bias: Option<Vec<f32>>,
    pub standardization_scale: Option<Vec<f32>>,
    pub clamp_data: Option<Vec<f32>>,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxVisionEncoder3D {
    pub fn new(params: PronaxVisionHyperparams3D) -> Result<Self, PronaxVisionEncoderError> {
        params.validate()?;
        
        let num_patches = (params.spatial_width / params.patch_size) * (params.spatial_height / params.patch_size);
        
        let encoder_layers: Vec<PronaxVisionEncoderLayer3D> = (0..params.transformer_layers)
            .map(|i| PronaxVisionEncoderLayer3D::new(i, &params))
            .collect();
        
        Ok(Self {
            hyperparams: params,
            patch_embedding_weights: vec![0.0; params.patch_size * params.patch_size * params.channel_depth * params.embedding_dim],
            position_embeddings: vec![0.0; num_patches * params.embedding_dim * 2],
            encoder_layers,
            standardization_bias: None,
            standardization_scale: None,
            clamp_data: None,
            spatial_position: ConversionCoordinate::standard(),
        })
    }
    
    pub fn initialize_clamps(&mut self) {
        if let Some(ref clamp_data) = self.clamp_data {
            let linears_per_layer = 7;
            
            for (layer_idx, layer) in self.encoder_layers.iter_mut().enumerate() {
                let layer_linears = [
                    &mut layer.self_attention.query_projection,
                    &mut layer.self_attention.key_projection,
                    &mut layer.self_attention.value_projection,
                    &mut layer.self_attention.output_projection,
                    &mut layer.mlp.gate_projection,
                    &mut layer.mlp.up_projection,
                    &mut layer.mlp.down_projection,
                ];
                
                for (linear_idx, linear) in layer_linears.iter_mut().enumerate() {
                    let offset = (layer_idx * linears_per_layer + linear_idx) * 4;
                    linear.initialize_from_packed(clamp_data, offset);
                }
            }
        }
    }
    
    pub fn encode_vision_zero_copy(
        &mut self,
        pixel_values: &[f32],
        num_patches_x: usize,
        num_patches_y: usize,
    ) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let num_patches = num_patches_x * num_patches_y;
        
        let mut hidden_state = vec![0.0; num_patches * self.hyperparams.embedding_dim];
        
        for patch_idx in 0..num_patches {
            let patch_start = patch_idx * self.hyperparams.patch_size * self.hyperparams.patch_size * self.hyperparams.channel_depth;
            let hidden_start = patch_idx * self.hyperparams.embedding_dim;
            
            for h in 0..self.hyperparams.embedding_dim.min(hidden_state.len() - hidden_start) {
                for p in 0..(self.hyperparams.patch_size * self.hyperparams.patch_size * self.hyperparams.channel_depth).min(pixel_values.len() - patch_start) {
                    let weight_idx = h * (self.hyperparams.patch_size * self.hyperparams.patch_size * self.hyperparams.channel_depth) + p;
                    if weight_idx < self.patch_embedding_weights.len() {
                        hidden_state[hidden_start + h] += pixel_values[patch_start + p] * self.patch_embedding_weights[weight_idx];
                    }
                }
            }
        }
        
        let pos_x_data: Vec<i32> = (0..num_patches).map(|i| (i % num_patches_x) as i32).collect();
        let pos_y_data: Vec<i32> = (0..num_patches).map(|i| (i / num_patches_x) as i32).collect();
        
        for patch_idx in 0..num_patches {
            let hidden_start = patch_idx * self.hyperparams.embedding_dim;
            let pos_x = pos_x_data[patch_idx] as usize;
            let pos_y = pos_y_data[patch_idx] as usize;
            
            for h in 0..self.hyperparams.embedding_dim {
                if hidden_start + h < hidden_state.len() {
                    let pos_emb_x_idx = pos_x * self.hyperparams.embedding_dim + h;
                    let pos_emb_y_idx = pos_y * self.hyperparams.embedding_dim + h;
                    
                    if pos_emb_x_idx < self.position_embeddings.len() {
                        hidden_state[hidden_start + h] += self.position_embeddings[pos_emb_x_idx];
                    }
                    if pos_emb_y_idx < self.position_embeddings.len() {
                        hidden_state[hidden_start + h] += self.position_embeddings[pos_emb_y_idx];
                    }
                }
            }
        }
        
        for layer in &mut self.encoder_layers {
            hidden_state = layer.forward_zero_copy(
                &hidden_state,
                &pos_x_data,
                &pos_y_data,
                num_patches,
                &self.hyperparams,
            )?;
        }
        
        Ok(hidden_state)
    }
}

/// Vision-to-text multimodal projector with pooling
#[derive(Debug, Clone)]
pub struct PronaxVisionTextProjector3D {
    pub projection_layer: PronaxClippableLinear3D,
    pub standardization_bias: Option<Vec<f32>>,
    pub standardization_scale: Option<Vec<f32>>,
    pub spatial_position: ConversionCoordinate,
}

impl PronaxVisionTextProjector3D {
    pub fn new(vision_dim: usize, text_dim: usize) -> Self {
        Self {
            projection_layer: PronaxClippableLinear3D::new(vision_dim, text_dim),
            standardization_bias: None,
            standardization_scale: None,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    pub fn project_with_pooling(
        &mut self,
        vision_features: &[f32],
        num_patches_x: usize,
        num_patches_y: usize,
        merge_factor: usize,
        params: &PronaxVisionHyperparams3D,
    ) -> Result<Vec<f32>, PronaxVisionEncoderError> {
        let num_patches = num_patches_x * num_patches_y;
        let vision_dim = vision_features.len() / num_patches;
        
        let mut spatial_grid = vec![vec![0.0; vision_dim]; num_patches_y];
        for patch_y in 0..num_patches_y {
            for patch_x in 0..num_patches_x {
                let patch_idx = patch_y * num_patches_x + patch_x;
                let start = patch_idx * vision_dim;
                if start + vision_dim <= vision_features.len() {
                    spatial_grid[patch_y] = vision_features[start..start + vision_dim].to_vec();
                }
            }
        }
        
        let merged_x = (num_patches_x + merge_factor - 1) / merge_factor;
        let merged_y = (num_patches_y + merge_factor - 1) / merge_factor;
        let mut pooled = vec![0.0; merged_x * merged_y * vision_dim];
        
        for my in 0..merged_y {
            for mx in 0..merged_x {
                let mut sum = vec![0.0; vision_dim];
                let mut count = 0;
                
                for py in (my * merge_factor)..((my + 1) * merge_factor).min(num_patches_y) {
                    for px in (mx * merge_factor)..((mx + 1) * merge_factor).min(num_patches_x) {
                        for (s, &v) in sum.iter_mut().zip(spatial_grid[py].iter()) {
                            *s += v;
                        }
                        count += 1;
                    }
                }
                
                let pooled_idx = (my * merged_x + mx) * vision_dim;
                for (i, s) in sum.iter().enumerate() {
                    if pooled_idx + i < pooled.len() && count > 0 {
                        pooled[pooled_idx + i] = s / count as f32;
                    }
                }
            }
        }
        
        let scale = (params.embedding_dim as f32).sqrt();
        for val in &mut pooled {
            *val *= scale;
        }
        
        if let (Some(bias), Some(scale)) = (&self.standardization_bias, &self.standardization_scale) {
            for (p, (b, s)) in pooled.iter_mut().zip(bias.iter().zip(scale.iter())) {
                *p = (*p - b) * s;
            }
        }
        
        let text_embeddings = self.projection_layer.forward_zero_copy(&pooled)?;
        
        Ok(text_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hyperparams_validation() {
        let params = PronaxVisionHyperparams3D::gemma4_default();
        assert!(params.validate().is_ok());
    }
    
    #[test]
    fn test_tensor_view_creation() {
        let data = vec![0.0f32; 224 * 224 * 3];
        let view = PronaxVisionTensorView3D::new(&data, 224, 224, 3);
        assert!(view.is_ok());
    }
    
    #[test]
    fn test_clippable_linear() {
        let mut linear = PronaxClippableLinear3D::new(1152, 1152);
        let dummy_input = vec![0.5f32; 1152];
        let result = linear.forward_zero_copy(&dummy_input);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_2d_rope() {
        let rope = PronaxTwoDimensionalRoPE3D::new(100.0, 72);
        let dummy_tensor = vec![0.5f32; 72 * 16 * 256];
        let pos_x = vec![0i32; 256];
        let pos_y = vec![0i32; 256];
        let result = rope.apply_2d_rope(&dummy_tensor, &pos_x, &pos_y, 256, 16);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vision_self_attention() {
        let params = PronaxVisionHyperparams3D::gemma4_default();
        let mut attention = PronaxVisionSelfAttention3D::new(1152, 72, 16, 100.0);
        
        let dummy_hidden = vec![0.5f32; 1152 * 256];
        let pos_x = vec![0i32; 256];
        let pos_y = vec![0i32; 256];
        
        let result = attention.forward_zero_copy(&dummy_hidden, &pos_x, &pos_y, 256, &params);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vision_mlp() {
        let mut mlp = PronaxVisionMLP3D::new(1152, 4608);
        let dummy_input = vec![0.5f32; 1152 * 256];
        let result = mlp.forward_zero_copy(&dummy_input);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vision_encoder_layer() {
        let params = PronaxVisionHyperparams3D::gemma4_default();
        let mut layer = PronaxVisionEncoderLayer3D::new(0, &params);
        
        let dummy_hidden = vec![0.5f32; 1152 * 256];
        let pos_x = vec![0i32; 256];
        let pos_y = vec![0i32; 256];
        
        let result = layer.forward_zero_copy(&dummy_hidden, &pos_x, &pos_y, 256, &params);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vision_encoder() {
        let params = PronaxVisionHyperparams3D::gemma4_default();
        let mut encoder = PronaxVisionEncoder3D::new(params).unwrap();
        
        let num_patches = (224 / 14) * (224 / 14);
        let dummy_pixels = vec![0.5f32; num_patches * 14 * 14 * 3];
        
        let result = encoder.encode_vision_zero_copy(&dummy_pixels, 16, 16);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_vision_text_projector() {
        let params = PronaxVisionHyperparams3D::gemma4_default();
        let mut projector = PronaxVisionTextProjector3D::new(1152, 2304);
        
        let dummy_vision = vec![0.5f32; 1152 * 256];
        let result = projector.project_with_pooling(&dummy_vision, 16, 16, 3, &params);
        assert!(result.is_ok());
    }
}
