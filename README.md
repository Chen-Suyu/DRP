<div align=center>

<h1>DRP: Deep Reasoning Project</h1>

[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-DPL--main-yellow?style=plastic)](https://huggingface.co/datasets/SnowCharmQ/DPL-main)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-DPL--meta-yellow?style=plastic)](https://huggingface.co/datasets/SnowCharmQ/DPL-meta)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-DPL--Yelp-yellow?style=plastic)](https://huggingface.co/datasets/SnowCharmQ/DPL-Yelp)
[![License](https://img.shields.io/badge/License-MIT-green?style=plastic)](LICENSE)

</div>

<br/>

This repository contains the implementation of the Deep Reasoning Project (DRP), focusing on evaluating and comparing different large language models' performance on reasoning tasks across Books and CDs & Vinyl datasets.

<p id="Catalogue"></p>  

## 📋 Catalogue 

- [📋 Catalogue](#-catalogue)
- [⚙️ Environment Setup](#️-environment-setup)
- [📊 Dataset](#-dataset)
- [⌛️ Quick Start](#️-quick-start)
- [📈 Experimental Results](#-experimental-results)
- [📁 Project Structure](#-project-structure)
- [🔧 Requirements](#-requirements)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## ⚙️ Environment Setup

```bash
# Create and activate conda environment
conda create -n DRP python=3.8
conda activate DRP

# Install required packages
pip install -r requirements.txt
```

## 📊 Dataset

This project uses datasets adapted from the [DPL-main dataset](https://huggingface.co/datasets/SnowCharmQ/DPL-main) on Hugging Face:

- **Books**: Book reviews and ratings dataset
- **CDs & Vinyl**: Music album reviews and ratings dataset

### Download Dataset

You can download the dataset from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download SnowCharmQ/DPL-main --repo-type dataset --local-dir data/

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SnowCharmQ/DPL-main", repo_type="dataset", local_dir="data/")
```

### Data Preprocessing

```bash
# Prepare differential inputs
python 0_prepare_diff_inputs.py

# Generate model outputs (DeepSeek)
python 1_get_diff_outputs_deepseek.py

# Generate model outputs (Qwen)
python 1_get_diff_outputs_qwen.py

# Get final outputs
python 2_get_final_outputs.py
```

## ⌛️ Quick Start

### Step 1: Download Dataset
```bash
# Download from Hugging Face
huggingface-cli download SnowCharmQ/DPL-main --repo-type dataset --local-dir data/
```

### Step 2: Run Preprocessing
```bash
python 0_prepare_diff_inputs.py
```

### Step 3: Generate Model Outputs
```bash
# For DeepSeek models
python 1_get_diff_outputs_deepseek.py

# For Qwen models
python 1_get_diff_outputs_qwen.py
```

### Step 4: Get Final Results
```bash
python 2_get_final_outputs.py
```

### Step 5: Evaluate
```bash
python 3_eval_basic.py
```

## 📈 Experimental Results

### Performance Comparison

![Performance Comparison](images/performance_comparison.png)

**Table 1**: Results on both datasets. **QwenX** and **DpSkX** refer to the Qwen-Instruct and DeepSeek-R1-Distill-Qwen models, respectively, each with X parameters. The best and second-best results are highlighted in **bold** and <u>underlined</u> font, respectively.

### Key Findings

- Our DRP method achieves competitive performance across different model sizes
- DeepSeek models show strong performance on the Books dataset
- Qwen models demonstrate excellent results on CDs & Vinyl dataset

## 📁 Project Structure

```
DRP/
├── data/                           # Dataset directory
├── utils/                          # Utility functions
├── images/                         # Result images
├── 0_prepare_diff_inputs.py       # Data preprocessing
├── 1_get_diff_outputs_deepseek.py # DeepSeek inference
├── 1_get_diff_outputs_qwen.py     # Qwen inference
├── 2_get_final_outputs.py         # Final output generation
├── 3_eval_basic.py                # Evaluation script
├── analyze_llm_improved.py        # Analysis script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🔧 Requirements

```
torch>=2.0.0
transformers>=4.30.0
huggingface-hub>=0.20.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the developers of the baseline methods and datasets used in this project. Special thanks to the [DPL project](https://huggingface.co/datasets/SnowCharmQ/DPL-main) for providing the dataset.

---

<div align=center>

**Note**: This is a research project. For questions or issues, please open an issue in this repository.

*Last updated: 2025-11-04*

</div>
