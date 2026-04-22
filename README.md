## Hi there 👋

<!--
**Pronax-Ai/pronax-ai** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
# <div align="center">⚡ Pronax AI</div>

<div align="center">
  <strong>High-Performance AI/ML Infrastructure in Rust</strong>
  <br><br>
  
  ![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
  ![AI](https://img.shields.io/badge/AI%2FML-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![LLM](https://img.shields.io/badge/LLM-412991?style=for-the-badge&logo=openai&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
</div>

---

## 🚀 Overview

**Pronax AI** is a cutting-edge AI/ML infrastructure platform built from the ground up in **Rust**, designed for high-performance inference and model deployment. Engineered for speed, safety, and scalability.

> *"Building the future of AI infrastructure, one line of Rust at a time."* — **ZKG**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦀 **Rust-Powered Core** | Memory-safe, zero-cost abstractions with blazing performance |
| 🧠 **Multi-Model Support** | Gemma, DeepSeek, BERT, and custom architectures |
| ⚡ **GGUF/GGML Format** | Native support for efficient quantized models |
| 🔄 **KV Cache Management** | Intelligent caching for accelerated inference |
| 🖼️ **Multimodal** | Text + Vision capabilities with image processing |
| 🔌 **OpenAI Compatible** | Drop-in API replacement with middleware layer |
| 🌐 **Cross-Platform** | Linux, macOS, Windows support with GPU acceleration |
| 📦 **Model Registry** | Built-in model discovery and management |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                            │
│         (OpenAI Compatible / Anthropic)                 │
├─────────────────────────────────────────────────────────┤
│                  Middleware Stack                       │
│              (Auth · Rate Limiting · Cache)           │
├─────────────────────────────────────────────────────────┤
│                   Model Registry                        │
│      (Gemma · DeepSeek · BERT · Custom Models)         │
├─────────────────────────────────────────────────────────┤
│                 ML Backend (Rust)                       │
│    (GGML · CUDA · Metal · Vulkan Compute)              │
├─────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer                 │
│         (GPU Detection · CPU Optimization)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Language**: Rust 🦀
- **ML Backends**: GGML, CUDA, Metal, Vulkan
- **Model Formats**: GGUF, GGML
- **Protocols**: HTTP/REST, WebSocket
- **Platforms**: Linux, macOS, Windows

---

## 📦 Quick Start

```bash
# Clone the repository
git clone https://github.com/ProNax-Ai/pronax-ai.git
cd pronax-ai

# Build the project
cargo build --release

# Run the server
cargo run --release -- server
```

---

## 🔧 Supported Models

| Model | Status | Quantization |
|-------|--------|--------------|
| Gemma 2 | ✅ Ready | Q4_K_M, Q5_K_M, Q8_0 |
| DeepSeek v2 | ✅ Ready | Q4_K_M, Q5_K_M |
| DeepSeek OCR | ✅ Ready | Vision + Text |
| BERT Embeddings | ✅ Ready | F32, Q8_0 |
| Mistral | 🚧 Beta | Q4_K_M |
| LLaMA | 🚧 Beta | Q4_K_M |

---

## 🌟 Key Highlights

```rust
// High-performance inference in Rust
use pronax_ai::model::Gemma2;

let model = Gemma2::load("gemma-2-9b-it-q4_k_m.gguf")
    .with_gpu_acceleration()
    .with_kv_cache(KVCacheConfig::optimized());

let response = model.generate("Explain quantum computing...").await?;
```

---

# <div align="center">⚡ Pronax AI</div>

<div align="center">
  <strong>High-Performance AI/ML Infrastructure in Rust</strong>
  <br><br>
  
  ![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
  ![AI](https://img.shields.io/badge/AI%2FML-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![LLM](https://img.shields.io/badge/LLM-412991?style=for-the-badge&logo=openai&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
</div>

---

## 🚀 Overview

**Pronax AI** is a cutting-edge AI/ML infrastructure platform built from the ground up in **Rust**, designed for high-performance inference and model deployment. Engineered for speed, safety, and scalability.

> *"Building the future of AI infrastructure, one line of Rust at a time."* — **ZKG**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦀 **Rust-Powered Core** | Memory-safe, zero-cost abstractions with blazing performance |
| 🧠 **Multi-Model Support** | Gemma, DeepSeek, BERT, and custom architectures |
| ⚡ **GGUF/GGML Format** | Native support for efficient quantized models |
| 🔄 **KV Cache Management** | Intelligent caching for accelerated inference |
| 🖼️ **Multimodal** | Text + Vision capabilities with image processing |
| 🔌 **OpenAI Compatible** | Drop-in API replacement with middleware layer |
| 🌐 **Cross-Platform** | Linux, macOS, Windows support with GPU acceleration |
| 📦 **Model Registry** | Built-in model discovery and management |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                            │
│         (OpenAI Compatible / Anthropic)                 │
├─────────────────────────────────────────────────────────┤
│                  Middleware Stack                       │
│              (Auth · Rate Limiting · Cache)           │
├─────────────────────────────────────────────────────────┤
│                   Model Registry                        │
│      (Gemma · DeepSeek · BERT · Custom Models)         │
├─────────────────────────────────────────────────────────┤
│                 ML Backend (Rust)                       │
│    (GGML · CUDA · Metal · Vulkan Compute)              │
├─────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer                 │
│         (GPU Detection · CPU Optimization)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Language**: Rust 🦀
- **ML Backends**: GGML, CUDA, Metal, Vulkan
- **Model Formats**: GGUF, GGML
- **Protocols**: HTTP/REST, WebSocket
- **Platforms**: Linux, macOS, Windows

---

## 📦 Quick Start

```bash
# Clone the repository
git clone https://github.com/ProNax-Ai/pronax-ai.git
cd pronax-ai

# Build the project
cargo build --release

# Run the server
cargo run --release -- server
```

---

## 🔧 Supported Models

| Model | Status | Quantization |
|-------|--------|--------------|
| Gemma 2 | ✅ Ready | Q4_K_M, Q5_K_M, Q8_0 |
| DeepSeek v2 | ✅ Ready | Q4_K_M, Q5_K_M |
| DeepSeek OCR | ✅ Ready | Vision + Text |
| BERT Embeddings | ✅ Ready | F32, Q8_0 |
| Mistral | 🚧 Beta | Q4_K_M |
| LLaMA | 🚧 Beta | Q4_K_M |

---

## 🌟 Key Highlights

```rust
// High-performance inference in Rust
use pronax_ai::model::Gemma2;

let model = Gemma2::load("gemma-2-9b-it-q4_k_m.gguf")
    .with_gpu_acceleration()
    .with_kv_cache(KVCacheConfig::optimized());

let response = model.generate("Explain quantum computing...").await?;
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting PRs.

---

## 📬 Connect with ZKG

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ZKG)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](#)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by **ZKG** and the Pronax AI Team

</div>
