use std::sync::Arc;

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;
use crate::kvcache::pronax_kvcache_causal::CausalKVCache;
use crate::model::pronax_model_input::{NeuralBatch, NeuralInput, NeuralMultimodalEmbedding};
use crate::tokenizer::pronax_bpe_tokenizer::NeuralBpeTokenizer;

use super::pronax_deepseekocr_image::{DeepSeekOcrConfig, DeepSeekOcrImageProcessor, ProcessedImage3D, TileAspectRatio};

/// DeepSeekOCR model errors
#[derive(Debug, Clone)]
pub enum DeepSeekOcrModelError {
    EncodingError(String),
    ForwardError(String),
    ConfigurationError(String),
    ImageProcessingError(String),
    CacheError(String),
}

impl std::fmt::Display for DeepSeekOcrModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EncodingError(s) => write!(f, "Encoding error: {}", s),
            Self::ForwardError(s) => write!(f, "Forward error: {}", s),
            Self::ConfigurationError(s) => write!(f, "Config error: {}", s),
            Self::ImageProcessingError(s) => write!(f, "Image processing: {}", s),
            Self::CacheError(s) => write!(f, "Cache error: {}", s),
        }
    }
}

impl std::error::Error for DeepSeekOcrModelError {}

/// 3D-aware DeepSeekOCR model configuration
#[derive(Debug, Clone, Copy)]
pub struct DeepSeekOcrModelConfig {
    /// Text model hidden size
    pub text_hidden_size: usize,
    /// Vision model hidden size
    pub vision_hidden_size: usize,
    /// SAM (Segment Anything) hidden size
    pub sam_hidden_size: usize,
    /// Number of text layers
    pub num_text_layers: usize,
    /// Number of vision layers
    pub num_vision_layers: usize,
    /// Number of SAM layers
    pub num_sam_layers: usize,
    /// Number of attention heads (text)
    pub text_num_heads: usize,
    /// Number of attention heads (vision)
    pub vision_num_heads: usize,
    /// Number of attention heads (SAM)
    pub sam_num_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Image size for vision model
    pub image_size: usize,
    /// Patch size for vision model
    pub patch_size: usize,
    /// RMS norm epsilon
    pub eps: f32,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RoPE scaling factor
    pub rope_scale: f32,
    /// Number of experts (for MoE layers)
    pub num_experts: usize,
    /// Number of experts used per token
    pub num_experts_used: usize,
    /// Dense layers before MoE
    pub num_dense_layers: usize,
    /// Image newline token embedding dimension
    pub image_newline_dim: usize,
    /// View separator token
    pub view_separator_token: i32,
    /// OCR special token
    pub ocr_token: i32,
    /// 3D spatial dimension
    pub spatial_dim: u16,
}

impl DeepSeekOcrModelConfig {
    /// Default DeepSeekOCR configuration
    pub fn default_ocr() -> Self {
        Self {
            text_hidden_size: 4096,
            vision_hidden_size: 1024,
            sam_hidden_size: 768,
            num_text_layers: 26,
            num_vision_layers: 24,
            num_sam_layers: 12,
            text_num_heads: 32,
            vision_num_heads: 16,
            sam_num_heads: 12,
            vocab_size: 129280,
            image_size: 640,
            patch_size: 14,
            eps: 1e-6,
            rope_base: 10000.0,
            rope_scale: 1.0,
            num_experts: 64,
            num_experts_used: 6,
            num_dense_layers: 1,
            image_newline_dim: 1024,
            view_separator_token: 128816,
            ocr_token: 128815,
            spatial_dim: 256,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), DeepSeekOcrModelError> {
        if self.text_hidden_size % self.text_num_heads != 0 {
            return Err(DeepSeekOcrModelError::ConfigurationError(
                "text_hidden_size not divisible by text_num_heads".to_string()
            ));
        }
        
        if self.image_size % self.patch_size != 0 {
            return Err(DeepSeekOcrModelError::ConfigurationError(
                "image_size not divisible by patch_size".to_string()
            ));
        }
        
        Ok(())
    }
}

impl Default for DeepSeekOcrModelConfig {
    fn default() -> Self {
        Self::default_ocr()
    }
}

/// 3D-aware multimodal embedding for OCR
#[derive(Debug, Clone)]
pub struct OcrMultimodalEmbedding3D {
    /// Local features from tiled image (high-res)
    pub local_features: Vec<f32>,
    /// Global features from thumbnail (low-res)
    pub global_features: Vec<f32>,
    /// View separator embedding
    pub view_separator: Vec<f32>,
    /// Combined embedding for model input
    pub combined_embedding: Vec<f32>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
    /// Feature positions in 3D space
    pub feature_positions: Vec<ConversionCoordinate>,
    /// Crop ratio used for tiling
    pub crop_ratio: TileAspectRatio,
}

impl OcrMultimodalEmbedding3D {
    /// Total feature dimensions
    pub fn total_features(&self) -> usize {
        self.local_features.len() + self.global_features.len() + self.view_separator.len()
    }
    
    /// Number of local patches
    pub fn num_local_patches(&self) -> usize {
        self.local_features.len() / self.spatial.depth as usize
    }
}

/// SAM (Segment Anything Model) encoder for OCR
#[derive(Debug, Clone)]
pub struct SamEncoder3D {
    /// Encoder blocks
    pub blocks: Vec<SamBlock3D>,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Global attention layer indices
    pub global_attention_layers: Vec<usize>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

/// SAM encoder block
#[derive(Debug, Clone)]
pub struct SamBlock3D {
    /// Attention layer
    pub attention: SamAttention3D,
    /// MLP layer
    pub mlp: SamMlp3D,
    /// Layer norm
    pub norm: SamNorm3D,
    /// Is global attention
    pub is_global: bool,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// SAM attention mechanism
#[derive(Debug, Clone)]
pub struct SamAttention3D {
    pub qkv_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
}

/// SAM MLP
#[derive(Debug, Clone)]
pub struct SamMlp3D {
    pub up_weights: Vec<f32>,
    pub down_weights: Vec<f32>,
    pub intermediate_size: usize,
}

/// SAM normalization
#[derive(Debug, Clone)]
pub struct SamNorm3D {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl SamEncoder3D {
    /// Create new SAM encoder
    pub fn new(config: &DeepSeekOcrModelConfig, global_attention_layers: Vec<usize>) -> Self {
        let blocks: Vec<SamBlock3D> = (0..config.num_sam_layers)
            .map(|i| SamBlock3D {
                attention: SamAttention3D {
                    qkv_weights: vec![0.0; config.sam_hidden_size * config.sam_hidden_size * 3],
                    output_weights: vec![0.0; config.sam_hidden_size * config.sam_hidden_size],
                    num_heads: config.sam_num_heads,
                    head_dim: config.sam_hidden_size / config.sam_num_heads,
                },
                mlp: SamMlp3D {
                    up_weights: vec![0.0; config.sam_hidden_size * 4 * config.sam_hidden_size],
                    down_weights: vec![0.0; 4 * config.sam_hidden_size * config.sam_hidden_size],
                    intermediate_size: 4 * config.sam_hidden_size,
                },
                norm: SamNorm3D {
                    weight: vec![1.0; config.sam_hidden_size],
                    eps: config.eps,
                },
                is_global: global_attention_layers.contains(&i),
                spatial_position: ConversionCoordinate::new(
                    i as u64,
                    (i / 4) as u16,
                    (i % 4) as u8,
                    1.0,
                ),
            })
            .collect();
        
        Self {
            blocks,
            hidden_size: config.sam_hidden_size,
            num_heads: config.sam_num_heads,
            global_attention_layers,
            spatial: SpatialTensorMetadata::new(
                config.image_size as u32,
                config.image_size as u32,
                config.sam_hidden_size as u32,
            ),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, DeepSeekOcrModelError> {
        // Simplified forward
        Ok(vec![0.0; input.len()])
    }
}

/// Vision encoder for OCR (ViT-style)
#[derive(Debug, Clone)]
pub struct VisionEncoder3D {
    /// Vision blocks
    pub blocks: Vec<VisionBlock3D>,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Patch size
    pub patch_size: usize,
    /// Image size
    pub image_size: usize,
    /// Number of patches per side
    pub num_patches: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

/// Vision encoder block
#[derive(Debug, Clone)]
pub struct VisionBlock3D {
    /// Attention layer
    pub attention: VisionAttention3D,
    /// MLP layer
    pub mlp: VisionMlp3D,
    /// Layer norms
    pub norm1: VisionNorm3D,
    pub norm2: VisionNorm3D,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// Vision attention
#[derive(Debug, Clone)]
pub struct VisionAttention3D {
    pub qkv_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub num_heads: usize,
}

/// Vision MLP
#[derive(Debug, Clone)]
pub struct VisionMlp3D {
    pub up_weights: Vec<f32>,
    pub down_weights: Vec<f32>,
}

/// Vision normalization
#[derive(Debug, Clone)]
pub struct VisionNorm3D {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl VisionEncoder3D {
    /// Create new vision encoder
    pub fn new(config: &DeepSeekOcrModelConfig) -> Self {
        let num_patches = config.image_size / config.patch_size;
        
        let blocks: Vec<VisionBlock3D> = (0..config.num_vision_layers)
            .map(|i| VisionBlock3D {
                attention: VisionAttention3D {
                    qkv_weights: vec![0.0; config.vision_hidden_size * config.vision_hidden_size * 3],
                    output_weights: vec![0.0; config.vision_hidden_size * config.vision_hidden_size],
                    num_heads: config.vision_num_heads,
                },
                mlp: VisionMlp3D {
                    up_weights: vec![0.0; config.vision_hidden_size * 4 * config.vision_hidden_size],
                    down_weights: vec![0.0; 4 * config.vision_hidden_size * config.vision_hidden_size],
                },
                norm1: VisionNorm3D {
                    weight: vec![1.0; config.vision_hidden_size],
                    eps: config.eps,
                },
                norm2: VisionNorm3D {
                    weight: vec![1.0; config.vision_hidden_size],
                    eps: config.eps,
                },
                spatial_position: ConversionCoordinate::new(
                    i as u64,
                    (i / 6) as u16,
                    (i % 6) as u8,
                    1.0,
                ),
            })
            .collect();
        
        Self {
            blocks,
            hidden_size: config.vision_hidden_size,
            num_heads: config.vision_num_heads,
            patch_size: config.patch_size,
            image_size: config.image_size,
            num_patches,
            spatial: SpatialTensorMetadata::new(
                num_patches as u32,
                num_patches as u32,
                config.vision_hidden_size as u32,
            ),
        }
    }
    
    /// Forward pass
    pub fn forward(&self, patches: &[f32], sam_features: &[f32]) -> Result<Vec<f32>, DeepSeekOcrModelError> {
        // Combine patch embeddings with SAM features
        Ok(vec![0.0; patches.len()])
    }
}

/// Projector to align vision and text embeddings
#[derive(Debug, Clone)]
pub struct MultimodalProjector3D {
    /// Projection weights
    pub weight: Vec<f32>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension (text hidden size)
    pub out_dim: usize,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl MultimodalProjector3D {
    /// Create new projector
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight: vec![0.0; in_dim * out_dim],
            in_dim,
            out_dim,
            spatial: SpatialTensorMetadata::new(in_dim as u32, out_dim as u32, 1),
        }
    }
    
    /// Project features
    pub fn project(&self, features: &[f32]) -> Vec<f32> {
        vec![0.0; features.len() * self.out_dim / self.in_dim]
    }
}

/// Text model (LLM) for OCR
#[derive(Debug, Clone)]
pub struct TextModel3D {
    /// Token embedding
    pub token_embedding: Vec<f32>,
    /// Transformer blocks
    pub blocks: Vec<TextBlock3D>,
    /// Output normalization
    pub output_norm: TextNorm3D,
    /// Output projection (LM head)
    pub output: Vec<f32>,
    /// Configuration
    pub config: TextConfig3D,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

/// Text configuration
#[derive(Debug, Clone, Copy)]
pub struct TextConfig3D {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_experts: usize,
    pub num_experts_used: usize,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub eps: f32,
}

/// Text transformer block
#[derive(Debug, Clone)]
pub struct TextBlock3D {
    /// Attention layer
    pub attention: TextAttention3D,
    /// Feed-forward (dense or MoE)
    pub feed_forward: TextFeedForward3D,
    /// Layer norms
    pub attn_norm: TextNorm3D,
    pub ffn_norm: TextNorm3D,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// Text attention
#[derive(Debug, Clone)]
pub struct TextAttention3D {
    pub q_weights: Vec<f32>,
    pub k_weights: Vec<f32>,
    pub v_weights: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Text feed-forward (polymorphic: dense or MoE)
#[derive(Debug, Clone)]
pub enum TextFeedForward3D {
    Dense(TextDenseFfn3D),
    MoE(TextMoeFfn3D),
}

/// Dense FFN
#[derive(Debug, Clone)]
pub struct TextDenseFfn3D {
    pub up_weights: Vec<f32>,
    pub down_weights: Vec<f32>,
}

/// MoE FFN
#[derive(Debug, Clone)]
pub struct TextMoeFfn3D {
    pub router: Vec<f32>,
    pub expert_up: Vec<f32>,
    pub expert_down: Vec<f32>,
    pub num_experts: usize,
    pub num_experts_used: usize,
}

/// Text normalization
#[derive(Debug, Clone)]
pub struct TextNorm3D {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl TextModel3D {
    /// Create new text model
    pub fn new(config: &DeepSeekOcrModelConfig) -> Self {
        let text_config = TextConfig3D {
            hidden_size: config.text_hidden_size,
            num_heads: config.text_num_heads,
            num_kv_heads: config.text_num_heads, // GQA
            num_experts: config.num_experts,
            num_experts_used: config.num_experts_used,
            rope_base: config.rope_base,
            rope_scale: config.rope_scale,
            eps: config.eps,
        };
        
        let blocks: Vec<TextBlock3D> = (0..config.num_text_layers)
            .map(|i| {
                let is_dense = i < config.num_dense_layers;
                
                TextBlock3D {
                    attention: TextAttention3D {
                        q_weights: vec![0.0; config.text_hidden_size * config.text_hidden_size],
                        k_weights: vec![0.0; config.text_hidden_size * config.text_hidden_size],
                        v_weights: vec![0.0; config.text_hidden_size * config.text_hidden_size],
                        output_weights: vec![0.0; config.text_hidden_size * config.text_hidden_size],
                        num_heads: config.text_num_heads,
                        num_kv_heads: config.text_num_heads,
                        head_dim: config.text_hidden_size / config.text_num_heads,
                    },
                    feed_forward: if is_dense {
                        TextFeedForward3D::Dense(TextDenseFfn3D {
                            up_weights: vec![0.0; config.text_hidden_size * 4 * config.text_hidden_size],
                            down_weights: vec![0.0; 4 * config.text_hidden_size * config.text_hidden_size],
                        })
                    } else {
                        TextFeedForward3D::MoE(TextMoeFfn3D {
                            router: vec![0.0; config.num_experts * config.text_hidden_size],
                            expert_up: vec![0.0; config.num_experts * config.text_hidden_size * 4 * config.text_hidden_size],
                            expert_down: vec![0.0; config.num_experts * 4 * config.text_hidden_size * config.text_hidden_size],
                            num_experts: config.num_experts,
                            num_experts_used: config.num_experts_used,
                        })
                    },
                    attn_norm: TextNorm3D {
                        weight: vec![1.0; config.text_hidden_size],
                        eps: config.eps,
                    },
                    ffn_norm: TextNorm3D {
                        weight: vec![1.0; config.text_hidden_size],
                        eps: config.eps,
                    },
                    spatial_position: ConversionCoordinate::new(
                        i as u64,
                        (i / 8) as u16,
                        (i % 8) as u8,
                        1.0,
                    ),
                }
            })
            .collect();
        
        Self {
            token_embedding: vec![0.0; config.vocab_size * config.text_hidden_size],
            blocks,
            output_norm: TextNorm3D {
                weight: vec![1.0; config.text_hidden_size],
                eps: config.eps,
            },
            output: vec![0.0; config.text_hidden_size * config.vocab_size],
            config: text_config,
            spatial: SpatialTensorMetadata::new(
                config.vocab_size as u32,
                config.text_hidden_size as u32,
                config.spatial_dim as u32,
            ),
        }
    }
    
    /// Apply RoPE shift
    pub fn shift(&self, key: &mut [f32], shift: &[f32]) {
        // Simplified shift
    }
}

/// Complete DeepSeekOCR model with 3D spatial awareness
pub struct DeepSeekOcrModel3D {
    /// SAM encoder for local/global features
    pub sam: SamEncoder3D,
    /// Vision encoder
    pub vision: VisionEncoder3D,
    /// Text model (LLM)
    pub text: TextModel3D,
    /// Multimodal projector
    pub projector: MultimodalProjector3D,
    /// Image newline embedding
    pub image_newline: Vec<f32>,
    /// View separator embedding
    pub view_separator: Vec<f32>,
    /// Configuration
    pub config: DeepSeekOcrModelConfig,
    /// Image processor
    pub image_processor: DeepSeekOcrImageProcessor,
    /// KV cache
    pub cache: CausalKVCache,
    /// Tokenizer
    pub tokenizer: Option<NeuralBpeTokenizer>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
}

impl DeepSeekOcrModel3D {
    /// Create new DeepSeekOCR model
    pub fn new(
        config: DeepSeekOcrModelConfig,
        global_attention_layers: Vec<usize>,
    ) -> Result<Self, DeepSeekOcrModelError> {
        config.validate()?;
        
        let sam = SamEncoder3D::new(&config, global_attention_layers);
        let vision = VisionEncoder3D::new(&config);
        let text = TextModel3D::new(&config);
        
        // Projector: vision_hidden_size + sam_hidden_size -> text_hidden_size
        let projector = MultimodalProjector3D::new(
            config.vision_hidden_size + config.sam_hidden_size,
            config.text_hidden_size,
        );
        
        let image_processor = DeepSeekOcrImageProcessor::default();
        
        let cache = CausalKVCache::new(
            config.num_text_layers,
            4096, // max context
            config.text_num_heads,
            config.text_hidden_size / config.text_num_heads,
        );
        
        Ok(Self {
            sam,
            vision,
            text,
            projector,
            image_newline: vec![0.0; config.text_hidden_size],
            view_separator: vec![0.0; config.text_hidden_size],
            config,
            image_processor,
            cache,
            tokenizer: None,
            spatial: SpatialTensorMetadata::new(
                config.image_size as u32,
                config.image_size as u32,
                config.spatial_dim as u32,
            ),
        })
    }
    
    /// Encode image to multimodal embeddings
    pub fn encode_multimodal(
        &self,
        image_data: &[u8],
    ) -> Result<Vec<NeuralMultimodalEmbedding>, DeepSeekOcrModelError> {
        // Process image to tiles and thumbnail
        let processed = self.image_processor.process_image(image_data)
            .map_err(|e| DeepSeekOcrModelError::ImageProcessingError(e.to_string()))?;
        
        let mut embeddings = Vec::new();
        let mut feature_positions = Vec::new();
        
        // Encode local features (tiled)
        let local_features = self.encode_local_features(&processed)?;
        
        // Encode global features (thumbnail)
        let global_features = self.encode_global_features(&processed)?;
        
        // Combine: local + newline + global + separator
        let mut combined = Vec::new();
        
        // Add local features with 3D positions
        for (i, tile) in processed.tiles.iter().enumerate() {
            combined.extend(&tile.normalized_data);
            feature_positions.push(tile.spatial_coord);
        }
        
        // Add image newline tokens between tiles
        for _ in 0..processed.tiles.len() {
            combined.extend(&self.image_newline);
        }
        
        // Add global features
        combined.extend(&global_features);
        feature_positions.push(processed.thumbnail.spatial_position);
        
        // Add view separator
        combined.extend(&self.view_separator);
        
        // Create multimodal embedding
        let spatial = SpatialTensorMetadata::new(
            combined.len() as u32 / self.config.text_hidden_size as u32,
            self.config.text_hidden_size as u32,
            1,
        );
        
        let embedding = NeuralMultimodalEmbedding::new(
            spatial,
            crate::model::pronax_model_input::MultimodalType::Image,
            0, // hash
        );
        
        embeddings.push(embedding);
        
        Ok(embeddings)
    }
    
    /// Encode local features from tiles
    fn encode_local_features(&self, processed: &ProcessedImage3D) -> Result<Vec<f32>, DeepSeekOcrModelError> {
        let tiles_data = &processed.combined_tiles_data;
        
        // SAM forward on tiles
        let sam_output = self.sam.forward(tiles_data)?;
        
        // Vision forward on tiles
        let vision_output = self.vision.forward(tiles_data, &sam_output)?;
        
        // Reshape and project
        let reshaped = self.reshape_features(&vision_output, &sam_output, processed.tile_ratio)?;
        
        // Project to text space
        let projected = self.projector.project(&reshaped);
        
        Ok(projected)
    }
    
    /// Encode global features from thumbnail
    fn encode_global_features(&self, processed: &ProcessedImage3D) -> Result<Vec<f32>, DeepSeekOcrModelError> {
        let thumb_data = &processed.thumbnail.normalized_data;
        
        // SAM forward on thumbnail
        let sam_output = self.sam.forward(thumb_data)?;
        
        // Vision forward on thumbnail
        let vision_output = self.vision.forward(thumb_data, &sam_output)?;
        
        // Project to text space
        let projected = self.projector.project(&vision_output);
        
        Ok(projected)
    }
    
    /// Reshape features with 3D spatial layout
    fn reshape_features(
        &self,
        vision: &[f32],
        sam: &[f32],
        crop_ratio: TileAspectRatio,
    ) -> Result<Vec<f32>, DeepSeekOcrModelError> {
        // Concatenate vision and SAM features
        let mut combined: Vec<f32> = vision.to_vec();
        combined.extend(sam);
        
        // Reshape to [h, w, channels] based on crop ratio
        let h = crop_ratio.y as usize;
        let w = crop_ratio.x as usize;
        
        // Simplified reshape
        Ok(combined)
    }
    
    /// Post-tokenization for multimodal inputs
    pub fn post_tokenize(&self, inputs: &[NeuralInput]) -> Result<Vec<NeuralInput>, DeepSeekOcrModelError> {
        let mut outputs = Vec::with_capacity(inputs.len() * 2);
        
        for input in inputs {
            if input.multimodal.is_empty() {
                outputs.push(input.clone());
                continue;
            }
            
            // Replace with OCR token and expand
            let num_embeddings = input.embedding_count();
            
            // First token with multimodal
            outputs.push(NeuralInput::mixed(
                self.config.ocr_token,
                input.multimodal.clone(),
                input.position_in_sequence,
                input.sequence_id,
            ).with_batch_constraint(num_embeddings));
            
            // Remaining OCR tokens (placeholders)
            for _ in 1..num_embeddings {
                outputs.push(NeuralInput::token(
                    self.config.ocr_token,
                    input.position_in_sequence,
                    input.sequence_id,
                ));
            }
        }
        
        Ok(outputs)
    }
    
    /// Forward pass
    pub fn forward(&mut self, batch: &NeuralBatch) -> Result<Vec<Vec<f32>>, DeepSeekOcrModelError> {
        // Get token embeddings
        let mut hidden_states: Vec<Vec<f32>> = batch.inputs.iter()
            .map(|input| {
                let token_id = input.token_id.max(0) as usize;
                let start = token_id * self.config.text_hidden_size;
                let end = start + self.config.text_hidden_size;
                
                if end <= self.text.token_embedding.len() {
                    self.text.token_embedding[start..end].to_vec()
                } else {
                    vec![0.0; self.config.text_hidden_size]
                }
            })
            .collect();
        
        // Inject multimodal embeddings
        for (i, input) in batch.inputs.iter().enumerate() {
            if !input.multimodal.is_empty() {
                // Replace with multimodal embedding
                // Simplified: just use zeros for structure
                hidden_states[i] = vec![0.0; self.config.text_hidden_size];
            }
        }
        
        // Process through text blocks
        for (layer_idx, block) in self.text.blocks.iter().enumerate() {
            self.cache.set_layer(layer_idx);
            
            for h in hidden_states.iter_mut() {
                // Apply attention and FFN
                self.apply_text_block(h, block)?;
            }
        }
        
        // Output normalization and projection
        let logits: Vec<Vec<f32>> = hidden_states.iter_mut()
            .map(|h| {
                // Apply norm
                for (x, &w) in h.iter_mut().zip(self.text.output_norm.weight.iter()) {
                    *x = *x * w;
                }
                
                // Project to vocab (simplified)
                vec![0.0; self.config.vocab_size]
            })
            .collect();
        
        Ok(logits)
    }
    
    fn apply_text_block(
        &self,
        hidden: &mut [f32],
        block: &TextBlock3D,
    ) -> Result<(), DeepSeekOcrModelError> {
        // Simplified attention + FFN
        // Full implementation would include RoPE, GQA, etc.
        Ok(())
    }
    
    /// Get model info
    pub fn model_info(&self) -> OcrModelInfo {
        OcrModelInfo {
            name: "DeepSeekOCR-3D".to_string(),
            text_params: self.estimate_text_params(),
            vision_params: self.estimate_vision_params(),
            sam_params: self.estimate_sam_params(),
            total_params: self.estimate_total_params(),
            vocab_size: self.config.vocab_size,
            image_size: self.config.image_size,
            use_moe: self.config.num_experts > 1,
            num_experts: self.config.num_experts,
        }
    }
    
    fn estimate_text_params(&self) -> usize {
        let c = &self.config;
        let embedding = c.vocab_size * c.text_hidden_size;
        let per_layer = c.text_hidden_size * c.text_hidden_size * 4; // attn + ffn
        embedding + c.num_text_layers * per_layer + embedding
    }
    
    fn estimate_vision_params(&self) -> usize {
        let c = &self.config;
        let per_layer = c.vision_hidden_size * c.vision_hidden_size * 4;
        c.num_vision_layers * per_layer
    }
    
    fn estimate_sam_params(&self) -> usize {
        let c = &self.config;
        let per_layer = c.sam_hidden_size * c.sam_hidden_size * 4;
        c.num_sam_layers * per_layer
    }
    
    fn estimate_total_params(&self) -> usize {
        self.estimate_text_params() + self.estimate_vision_params() + self.estimate_sam_params()
    }
}

/// OCR model information
#[derive(Debug, Clone)]
pub struct OcrModelInfo {
    pub name: String,
    pub text_params: usize,
    pub vision_params: usize,
    pub sam_params: usize,
    pub total_params: usize,
    pub vocab_size: usize,
    pub image_size: usize,
    pub use_moe: bool,
    pub num_experts: usize,
}

/// Utility functions for OCR model
pub mod ocr_model_utils {
    use super::*;
    
    /// Compute memory requirement for model
    pub fn estimate_memory(config: &DeepSeekOcrModelConfig) -> u64 {
        let text_mem = (config.vocab_size * config.text_hidden_size * 4) as u64;
        let vision_mem = (config.num_vision_layers * config.vision_hidden_size * config.vision_hidden_size * 4) as u64;
        let sam_mem = (config.num_sam_layers * config.sam_hidden_size * config.sam_hidden_size * 4) as u64;
        
        text_mem + vision_mem + sam_mem
    }
    
    /// Get tokenizer regex patterns
    pub fn get_tokenizer_patterns() -> Vec<&'static str> {
        vec![
            r"\p{N}{1,3}",
            r"[一-龥぀-ゟ゠-ヿ]+",
            r"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config() {
        let config = DeepSeekOcrModelConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.text_hidden_size, 4096);
    }
    
    #[test]
    fn test_sam_encoder() {
        let config = DeepSeekOcrModelConfig::default();
        let sam = SamEncoder3D::new(&config, vec![2, 5, 8, 11]);
        
        assert_eq!(sam.blocks.len(), config.num_sam_layers);
        assert!(sam.blocks[2].is_global);
        assert!(!sam.blocks[1].is_global);
    }
    
    #[test]
    fn test_vision_encoder() {
        let config = DeepSeekOcrModelConfig::default();
        let vision = VisionEncoder3D::new(&config);
        
        assert_eq!(vision.blocks.len(), config.num_vision_layers);
        assert_eq!(vision.patch_size, config.patch_size);
    }
    
    #[test]
    fn test_text_model() {
        let config = DeepSeekOcrModelConfig::default();
        let text = TextModel3D::new(&config);
        
        assert_eq!(text.blocks.len(), config.num_text_layers);
        
        // First layer should be dense
        assert!(matches!(text.blocks[0].feed_forward, TextFeedForward3D::Dense(_)));
    }
    
    #[test]
    fn test_model_creation() {
        let config = DeepSeekOcrModelConfig::default();
        let model = DeepSeekOcrModel3D::new(config, vec![2, 5, 8, 11]);
        
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert!(info.total_params > 1_000_000_000); // > 1B params
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = DeepSeekOcrModelConfig::default();
        let mem = ocr_model_utils::estimate_memory(&config);
        
        assert!(mem > 0);
        assert!(mem > 1_000_000_000); // > 1GB
    }
}
