# DRP: Difference-aware Reasoning Personalization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization**
> 
> Suyu Chen¹, Yimeng Bai¹†, Xiaoyan Zhao², Yang Zhang³†
> 
> ¹University of Science and Technology of China, Hefei, China  
> ²The Chinese University of Hong Kong, Hong Kong, China  
> ³National University of Singapore, Singapore


## 📖 Overview

This repository contains the official implementation of **DRP (Difference-aware Reasoning Personalization)**, a novel framework that enhances LLM personalization through inference scaling. DRP addresses the limitations of existing personalization methods by:

- 🎯 **Automatic Dimension Generation**: Autonomously identifies relevant difference feature dimensions beyond fixed, predefined categories
- 🧠 **Deliberate Reasoning**: Leverages inference scaling to enable deeper, System-2 thinking over user differences
- 📊 **Enhanced Coverage & Granularity**: Captures broader and more fine-grained user preference patterns
- ✨ **Training-Free**: Achieves personalization without additional model training

## 🔬 Abstract

Large Language Models (LLMs) are increasingly integrated into personalized applications, but existing methods face limitations in **coverage** and **granularity** when capturing user differences. **DRP** introduces a reasoning-enhanced framework that:

1. Reconstructs the difference extraction mechanism using **inference scaling**
2. Autonomously identifies relevant difference features from user histories
3. Generates structured definitions and descriptions for each dimension
4. Produces personalized outputs through deliberate reasoning (System-2 thinking)

### Key Results

- 🏆 **Up to 23.0% improvement** in BLEU score on personalized review generation tasks
- 📈 Consistent outperformance across multiple baseline methods
- 🎓 Superior performance on both **Books** and **CDs & Vinyl** datasets
- 🔍 Demonstrated benefits of broader coverage and deeper semantic analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Chen-Suyu/DRP.git
cd DRP

# Install dependencies
pip install -r requirements.txt
```

## 📊 Repository Structure

```
DRP/
├── 0_prepare_diff_inputs.py          # Prepare difference extraction inputs
├── 1_get_diff_outputs_deepseek.py    # Extract differences using DeepSeek
├── 1_get_diff_outputs_qwen.py        # Extract differences using Qwen
├── 2_get_final_outputs.py            # Generate final personalized outputs
├── 3_eval_basic.py                   # Basic evaluation metrics
├── analyze_llm_improved.py           # Analyze LLM improvements
├── data/
│   └── create_data.py                # Data preparation utilities
├── utils/
│   ├── get_local_model.py            # Local model loading utilities
│   ├── metrics.py                    # Evaluation metrics
│   ├── preprocess.py                 # Data preprocessing
│   ├── templates.py                  # Prompt templates
│   └── utils.py                      # General utilities
├── README.md
├── requirements.txt
└── LICENSE
```

## 🔧 Usage

### Step 1: Prepare Difference Extraction Inputs

```bash
python 0_prepare_diff_inputs.py
```

### Step 2: Extract User Differences

Using DeepSeek models:
```bash
python 1_get_diff_outputs_deepseek.py
```

Or using Qwen models:
```bash
python 1_get_diff_outputs_qwen.py
```

### Step 3: Generate Personalized Outputs

```bash
python 2_get_final_outputs.py
```

### Step 4: Evaluate Results

```bash
python 3_eval_basic.py
```

## 📈 Experimental Results

### Performance Comparison


### Key Findings

1. **Reasoning-Enhanced Models Win**: DeepSeek models consistently outperform Qwen models due to superior reasoning capabilities
2. **Scaling Effects**: Larger models (14B, 32B) show improved performance, benefiting from enhanced reasoning capacity
3. **Coverage Matters**: More unique valid features correlate strongly with better personalization quality (r² shown in Figure 1)

## 🎯 Methodology

### DRP Pipeline

```
User History → Representative User Selection → Difference Extraction 
    ↓                                              ↓
  Target Item ← Personalized Generation ← Summarization & Context
```

#### Key Components:

1. **Representative User Selection**: Cluster users based on historical texts and select representatives from different clusters

2. **Difference Extraction**: 
   - Automatic dimension generation (Λ_auto)
   - Deliberate reasoning with LLM_E (System-2 thinking)
   - Reflective validation to filter invalid differences

3. **Summarization and Generation**:
   - Compress personalized context using LLM_S
   - Generate final output with LLM_G

## 📚 Datasets

We evaluate DRP on the **Amazon Reviews dataset** preprocessed by DPL, which includes:
- **Books** category
- **CDs & Vinyl** category

Each dataset contains user review histories for personalized review generation tasks.

## 🔬 Implementation Details

- **Models**: DeepSeek-R1-Distill-Qwen and Qwen-2.5-Instruct
- **Parameter Scales**: 1.5B, 7B, 14B, 32B
- **Temperature**: Experiments conducted at temperatures 0 and 0.8, results averaged
- **Evaluation Metrics**: BLEU, METEOR, ROUGE-1, ROUGE-L


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of DPL for preprocessing the Amazon Reviews dataset and providing baseline implementations.

## 🔗 Related Work

- [DPL: Difference-Aware Personalization](https://github.com/XXX/DPL)
- [DeepSeek Models](https://github.com/deepseek-ai)
- [Qwen Models](https://github.com/QwenLM/Qwen)

---

**Keywords**: LLM Personalization, Inference Scaling, LLM Reasoning, User Modeling, Personalized Text Generation
