use std::collections::HashMap;
use serde::{Deserialize, Deserializer, Serialize};
use crate::convert::pronax_converter_core::{ConversionCoordinate, NeuralMetadataKV, NeuralModelConverter, NeuralSourceTensor};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::{NeuralConversionTokenizer, SpecialTokenType};

/// 3D Spatial coordinate for Nemotron-H
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NemotronHCoordinate { pub layer_seq: u64, pub hybrid_tier: u16, pub ssm_depth: u8, pub evolution: f32 }
impl NemotronHCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, evo: f32) -> Self { Self { layer_seq: seq, hybrid_tier: tier, ssm_depth: depth, evolution: evo } }
    pub const fn ssm() -> Self { Self::new(0, 1000, 40, 0.999) }
    pub const fn attn() -> Self { Self::new(0, 980, 20, 0.995) }
    pub const fn moe() -> Self { Self::new(0, 960, 18, 0.992) }
    pub const fn dense() -> Self { Self::new(0, 940, 16, 0.990) }
    pub fn evolution_score(&self) -> u64 { (self.hybrid_tier as u64 * 85) + (self.ssm_depth as u64 * 15) + (self.evolution * 2500.0) as u64 }
}

/// Hybrid layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayer { Mamba, Attention, MoE, Dense }
impl HybridLayer { 
    pub fn from_char(c: char) -> Option<Self> { match c { 'M' => Some(Self::Mamba), '*'|'A' => Some(Self::Attention), 'E' => Some(Self::MoE), '-' => Some(Self::Dense), _ => None } }
    pub fn as_char(&self) -> char { match self { Self::Mamba => 'M', Self::Attention => 'A', Self::MoE => 'E', Self::Dense => '-' } }
    pub fn coord(&self) -> NemotronHCoordinate { match self { Self::Mamba => NemotronHCoordinate::ssm(), Self::Attention => NemotronHCoordinate::attn(), Self::MoE => NemotronHCoordinate::moe(), Self::Dense => NemotronHCoordinate::dense() } }
}

/// Hybrid pattern (string or array)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPattern(pub String);
impl<'de> Deserialize<'de> for HybridPattern {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        if let Ok(s) = String::deserialize(d) { return Ok(Self(s.trim().to_string())); }
        let parts: Vec<String> = Vec::deserialize(d).map_err(|e| D::Error::custom(e))?;
        Ok(Self(parts.join("")))
    }
}
impl Serialize for HybridPattern { fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { s.serialize_str(&self.0) } }
impl HybridPattern {
    pub fn parse(&self) -> Vec<HybridLayer> { self.0.chars().filter_map(HybridLayer::from_char).collect() }
    pub fn validate(&self, expected: u32) -> Result<(), String> { if self.0.len() != expected as usize { Err(format!("pattern length {} != {}", self.0.len(), expected)) } else { Ok(()) } }
}

/// SSM config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmConfig { pub conv_kernel: u32, pub state_size: u32, pub num_heads: u32, pub head_dim: u32, pub group_count: u32 }
impl SsmConfig { pub fn new() -> Self { Self { conv_kernel: 4, state_size: 16, num_heads: 64, head_dim: 16, group_count: 1 } } pub fn inner_size(&self) -> u32 { self.head_dim * self.num_heads } }
impl Default for SsmConfig { fn default() -> Self { Self::new() } }

/// MoE config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig { pub n_routed_experts: Option<u32>, pub num_experts: Option<u32>, pub n_shared_experts: Option<u32>, pub num_shared_experts: Option<u32>, pub experts_per_tok: u32, pub intermediate_size: Option<u32>, pub norm_topk_prob: bool, pub routed_scaling_factor: f32, pub expert_group_count: u32, pub expert_group_used_count: u32 }
impl MoeConfig { pub fn new() -> Self { Self { n_routed_experts: None, num_experts: None, n_shared_experts: None, num_shared_experts: None, experts_per_tok: 0, intermediate_size: None, norm_topk_prob: false, routed_scaling_factor: 1.0, expert_group_count: 0, expert_group_used_count: 0 } } pub fn is_enabled(&self) -> bool { self.routed_count() > 0 || self.experts_per_tok > 0 } pub fn routed_count(&self) -> u32 { self.n_routed_experts.or(self.num_experts).unwrap_or(0) } pub fn shared_count(&self) -> u32 { self.n_shared_experts.or(self.num_shared_experts).unwrap_or(0) } }
impl Default for MoeConfig { fn default() -> Self { Self::new() } }

/// Nemotron-H config
#[derive(Debug, Clone, Deserialize)]
pub struct NemotronHConfig { pub max_position_embeddings: u32, pub hidden_size: u32, pub num_hidden_layers: u32, pub num_attention_heads: u32, pub num_key_value_heads: Option<u32>, pub head_dim: Option<u32>, pub layer_norm_epsilon: Option<f32>, pub norm_eps: Option<f32>, pub rope_theta: f32, pub partial_rotary_factor: f32, pub intermediate_size: u32, pub hybrid_pattern: HybridPattern, #[serde(flatten)] pub ssm: SsmConfig, #[serde(flatten)] pub moe: MoeConfig }
impl NemotronHConfig {
    pub fn base() -> Self { Self { max_position_embeddings: 4096, hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: None, head_dim: None, layer_norm_epsilon: None, norm_eps: None, rope_theta: 10000.0, partial_rotary_factor: 1.0, intermediate_size: 14336, hybrid_pattern: HybridPattern("MMAA".to_string() + &"-".repeat(28)), ssm: SsmConfig::new(), moe: MoeConfig::new() } }
    pub fn eff_head_dim(&self) -> u32 { self.head_dim.unwrap_or_else(|| self.hidden_size / self.num_attention_heads) }
    pub fn eff_kv_heads(&self) -> u32 { self.num_key_value_heads.unwrap_or(self.num_attention_heads) }
    pub fn epsilon(&self) -> f32 { self.norm_eps.or(self.layer_norm_epsilon).unwrap_or(1e-5) }
    pub fn validate(&self) -> Result<(), String> {
        if self.num_hidden_layers == 0 || self.hidden_size == 0 || self.num_attention_heads == 0 { return Err("missing required fields".to_string()); }
        if self.head_dim.is_none() && self.hidden_size % self.num_attention_heads != 0 { return Err("hidden_size not divisible by num_attention_heads".to_string()); }
        if self.hybrid_pattern.0.is_empty() { return Err("hybrid_pattern required".to_string()); }
        self.hybrid_pattern.validate(self.num_hidden_layers)?;
        Ok(())
    }
    pub fn layer_arrays(&self) -> (Vec<u32>, Vec<u32>) {
        let layers = self.hybrid_pattern.parse();
        let (mut kv, mut ffn) = (Vec::new(), Vec::new());
        let attn_kv = self.eff_kv_heads();
        let moe_ffn = self.moe.intermediate_size.unwrap_or(self.intermediate_size);
        for lt in layers { match lt { HybridLayer::Mamba => { kv.push(0); ffn.push(0); } HybridLayer::Attention => { kv.push(attn_kv); ffn.push(0); } HybridLayer::MoE => { kv.push(attn_kv); ffn.push(moe_ffn); } HybridLayer::Dense => { kv.push(attn_kv); ffn.push(self.intermediate_size); } } }
        (kv, ffn)
    }
}

/// Nemotron-H converter
#[derive(Debug, Clone)]
pub struct NeuralNemotronHConverter { pub config: NemotronHConfig, pub coord: NemotronHCoordinate, pub replacements: HashMap<String, String> }
impl NeuralNemotronHConverter {
    pub fn new(config: NemotronHConfig) -> Self {
        let mut c = Self { config, coord: NemotronHCoordinate::ssm(), replacements: HashMap::new() };
        c.init_replacements(); c
    }
    fn init_replacements(&mut self) {
        let r = &mut self.replacements;
        r.insert("lm_head".to_string(), "output".to_string()); r.insert("backbone.embeddings".to_string(), "token_embd".to_string()); r.insert("backbone.norm_f".to_string(), "output_norm".to_string()); r.insert("backbone.layers".to_string(), "blk".to_string());
        r.insert("mixer.in_proj".to_string(), "ssm_in".to_string()); r.insert("mixer.out_proj".to_string(), "ssm_out".to_string()); r.insert("mixer.dt_bias".to_string(), "ssm_dt.bias".to_string()); r.insert("mixer.A_log".to_string(), "ssm_a".to_string()); r.insert("mixer.D".to_string(), "ssm_d".to_string()); r.insert("mixer.conv1d".to_string(), "ssm_conv1d".to_string()); r.insert("mixer.norm.weight".to_string(), "ssm_norm.weight".to_string());
        r.insert("mixer.q_proj".to_string(), "attn_q".to_string()); r.insert("mixer.k_proj".to_string(), "attn_k".to_string()); r.insert("mixer.v_proj".to_string(), "attn_v".to_string()); r.insert("mixer.o_proj".to_string(), "attn_output".to_string());
        r.insert("mixer.gate.e_score_correction_bias".to_string(), "exp_probs_b.bias".to_string()); r.insert("mixer.gate".to_string(), "ffn_gate_inp".to_string()); r.insert("mixer.fc1_latent_proj".to_string(), "ffn_latent_in".to_string()); r.insert("mixer.fc2_latent_proj".to_string(), "ffn_latent_out".to_string()); r.insert("mixer.shared_experts.up_proj".to_string(), "ffn_up_shexp".to_string()); r.insert("mixer.shared_experts.down_proj".to_string(), "ffn_down_shexp".to_string()); r.insert("mixer.up_proj".to_string(), "ffn_up".to_string()); r.insert("mixer.down_proj".to_string(), "ffn_down".to_string());
        r.insert(".norm.weight".to_string(), ".attn_norm.weight".to_string());
    }
    pub fn replace_name(&self, name: &str) -> String { let mut r = name.to_string(); for (f, t) in &self.replacements { r = r.replace(f, t); } r }
    pub fn extract_layer(&self, name: &str) -> Option<usize> { if !name.starts_with("blk.") { return None; } name.split('.').nth(1)?.parse().ok() }
    pub fn coord_for(&self, name: &str, idx: Option<usize>) -> NemotronHCoordinate {
        if let Some(i) = idx { if let Some(lt) = self.config.hybrid_pattern.0.chars().nth(i).and_then(HybridLayer::from_char) { let mut c = lt.coord(); c.layer_seq = i as u64; return c; } }
        if name.contains("ssm") { NemotronHCoordinate::ssm() } else if name.contains("attn") { NemotronHCoordinate::attn() } else if name.contains("exps") || name.contains("gate_inp") { NemotronHCoordinate::moe() } else { self.coord }
    }
    pub fn repack_ssm_a(&self, d: &[f32]) -> Vec<f32> { d.iter().map(|&v| -v.exp()).collect() }
    pub fn norm_shape(&self, s: &[u64]) -> Vec<u64> { match s.len() { 1 => vec![s[0], 1], 2 => if s[0] == 1 && s[1] > 1 { vec![s[1], 1] } else if s[1] == 1 && s[0] > 1 { vec![s[0], 1] } else { s.to_vec() }, _ => s.to_vec() } }
    pub fn reshape_norm(&self, s: &[u64], g: u64) -> Vec<u64> { match s.len() { 1 => if g > 0 && s[0] % g == 0 { vec![g, s[0] / g] } else { s.to_vec() }, 2 => if s[0] == 1 && g > 0 && s[1] % g == 0 { vec![g, s[1] / g] } else { s.to_vec() }, _ => s.to_vec() } }
    pub fn reshape_conv(&self, s: &[u64]) -> Vec<u64> { if s.len() == 3 { if s[0] == 1 { vec![s[1], s[2]] } else if s[1] == 1 { vec![s[0], s[2]] } else { s.to_vec() } } else { s.to_vec() } }
    pub fn info(&self) -> (u32, u32, u32, u32) {
        let layers = self.config.hybrid_pattern.parse(); let mut s = (0, 0, 0, 0);
        for l in layers { match l { HybridLayer::Mamba => s.0 += 1, HybridLayer::Attention => s.1 += 1, HybridLayer::MoE => s.2 += 1, HybridLayer::Dense => s.3 += 1, } }
        s
    }
}

impl NeuralModelConverter for NeuralNemotronHConverter {
    fn to_metadata_kv(&self, t: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new(); let c = &self.config;
        kv.insert("general.architecture", if c.moe.is_enabled() { "nemotron_h_moe" } else { "nemotron_h" });
        kv.insert("block_count", c.num_hidden_layers); kv.insert("context_length", c.max_position_embeddings); kv.insert("embedding_length", c.hidden_size); kv.insert("attention.head_count", c.num_attention_heads); kv.insert("attention.key_length", c.eff_head_dim()); kv.insert("attention.value_length", c.eff_head_dim());
        kv.insert("attention.layer_norm_epsilon", c.epsilon()); kv.insert("attention.layer_norm_rms_epsilon", c.epsilon()); kv.insert("rope.freq_base", c.rope_theta);
        if c.partial_rotary_factor > 0.0 && c.partial_rotary_factor <= 1.0 { kv.insert("rope.dimension_count", (c.eff_head_dim() as f32 * c.partial_rotary_factor) as u32); }
        let (kv_heads, ffn_lens) = c.layer_arrays(); kv.insert("attention.head_count_kv", kv_heads); kv.insert("feed_forward_length", ffn_lens);
        kv.insert("ssm.conv_kernel", c.ssm.conv_kernel); kv.insert("ssm.inner_size", c.ssm.inner_size()); kv.insert("ssm.state_size", c.ssm.state_size); kv.insert("ssm.group_count", c.ssm.group_count); kv.insert("ssm.time_step_rank", c.ssm.num_heads);
        if c.moe.is_enabled() { kv.insert("expert_count", c.moe.routed_count()); kv.insert("expert_used_count", c.moe.experts_per_tok); kv.insert("expert_feed_forward_length", c.moe.intermediate_size.unwrap_or(c.intermediate_size)); if c.moe.shared_count() > 0 { kv.insert("expert_shared_count", c.moe.shared_count()); } kv.insert("expert_weights_norm", c.moe.norm_topk_prob); kv.insert("expert_weights_scale", c.moe.routed_scaling_factor); if c.moe.expert_group_count > 0 { kv.insert("expert_group_count", c.moe.expert_group_count); } if c.moe.expert_group_used_count > 0 { kv.insert("expert_group_used_count", c.moe.expert_group_used_count); } }
        kv.insert("pronax.coordinate.tier", self.coord.hybrid_tier); kv.insert("pronax.coordinate.depth", self.coord.ssm_depth); kv.insert("pronax.coordinate.evolution", self.coord.evolution);
        let tok_kv = t.to_kv(); kv.merge(tok_kv); kv.set_architecture(self.architecture()); kv
    }
    fn convert_tensors(&self, ts: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut out = Vec::new(); let g = self.config.ssm.group_count as u64;
        for t in ts {
            let name = self.replace_name(&t.name); let idx = self.extract_layer(&name);
            let mut shape = t.shape.clone(); let mut data = t.data.clone();
            if name.ends_with(".ssm_a") { shape = self.norm_shape(&shape); data = self.repack_ssm_a(&data); }
            else if name.ends_with(".ssm_d") { shape = self.norm_shape(&shape); }
            else if name.ends_with(".ssm_norm.weight") { shape = self.reshape_norm(&shape, g); }
            else if name.ends_with(".ssm_conv1d.weight") { shape = self.reshape_conv(&shape); }
            let co = self.coord_for(&name, idx);
            out.push(NeuralGgmlTensor::new(name, t.data_type, shape, data).with_coordinate(ConversionCoordinate::new(idx.unwrap_or(0) as u64, co.hybrid_tier, co.ssm_depth, co.evolution)));
        }
        out
    }
    fn name_replacements(&self) -> Vec<(String, String)> { self.replacements.iter().map(|(k, v)| (k.clone(), v.clone())).collect() }
    fn special_token_types(&self) -> Vec<SpecialTokenType> { vec![SpecialTokenType::Bos, SpecialTokenType::Eos, SpecialTokenType::Pad, SpecialTokenType::Unknown] }
    fn architecture(&self) -> &str { if self.config.moe.is_enabled() { "nemotron_h_moe" } else { "nemotron_h" } }
    fn coordinate(&self) -> ConversionCoordinate { ConversionCoordinate::new(self.coord.layer_seq, self.coord.hybrid_tier, self.coord.ssm_depth, self.coord.evolution) }
}

pub fn create_nemotron_h(config: NemotronHConfig) -> NeuralNemotronHConverter { NeuralNemotronHConverter::new(config) }
pub fn create_nemotron_h_base() -> NeuralNemotronHConverter { NeuralNemotronHConverter::new(NemotronHConfig::base()) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_coord() { let c = NemotronHCoordinate::ssm(); assert_eq!(c.hybrid_tier, 1000); assert!(c.evolution_score() > 0); }
    #[test] fn test_hybrid_layer() { assert_eq!(HybridLayer::from_char('M'), Some(HybridLayer::Mamba)); assert_eq!(HybridLayer::from_char('A'), Some(HybridLayer::Attention)); assert_eq!(HybridLayer::from_char('E'), Some(HybridLayer::MoE)); assert_eq!(HybridLayer::from_char('-'), Some(HybridLayer::Dense)); }
    #[test] fn test_hybrid_pattern() { let p = HybridPattern("MMAA--".to_string()); assert_eq!(p.parse().len(), 6); assert!(p.validate(6).is_ok()); assert!(p.validate(5).is_err()); }
    #[test] fn test_ssm_config() { let s = SsmConfig::new(); assert_eq!(s.inner_size(), s.head_dim * s.num_heads); }
    #[test] fn test_moe_config() { let m = MoeConfig::new(); assert!(!m.is_enabled()); let mut m2 = MoeConfig::new(); m2.n_routed_experts = Some(8); m2.experts_per_tok = 2; assert!(m2.is_enabled()); assert_eq!(m2.routed_count(), 8); }
    #[test] fn test_config() { let c = NemotronHConfig::base(); assert_eq!(c.eff_head_dim(), c.hidden_size / c.num_attention_heads); let (kv, ffn) = c.layer_arrays(); assert_eq!(kv.len(), c.num_hidden_layers as usize); assert_eq!(ffn.len(), c.num_hidden_layers as usize); }
    #[test] fn test_converter() { let conv = create_nemotron_h_base(); assert_eq!(conv.architecture(), "nemotron_h"); let (m, a, e, d) = conv.info(); assert_eq!(m + a + e + d, 32); }
    #[test] fn test_ssm_repack() { let conv = create_nemotron_h_base(); let data = vec![0.0f32, 1.0f32, 2.0f32]; let repacked = conv.repack_ssm_a(&data); assert!(repacked[0] < -0.99); assert!(repacked[1] < -2.0); }
    #[test] fn test_shape_norm() { let conv = create_nemotron_h_base(); assert_eq!(conv.norm_shape(&[128]), vec![128, 1]); assert_eq!(conv.norm_shape(&[1, 128]), vec![128, 1]); }
    #[test] fn test_name_replace() { let conv = create_nemotron_h_base(); let r = conv.replace_name("backbone.layers.0.mixer.q_proj.weight"); assert!(r.contains("blk")); assert!(r.contains("attn_q")); }
}
