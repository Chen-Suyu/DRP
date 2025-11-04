<div align=center>

<h1>DRP: Deep Reasoning Project</h1>
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-DPL--main-yellow?style=for-the-badge)](https://huggingface.co/datasets/SnowCharmQ/DPL-main)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

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

## ⚙️ Environment Setup

```bash
# Create and activate conda environment
conda create -n DRP python=3.8
conda activate DRP

# Install required packages
pip install -r requirements.txt
```

## 📊 Dataset

This project uses two datasets:
- **Books**: Book reviews and ratings dataset
- **CDs & Vinyl**: Music album reviews and ratings dataset

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

### Step 1: Prepare Data
Ensure your input data is placed in the `data/` directory.

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
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the developers of the baseline methods and datasets used in this project.

---

**Note**: This is a research project. For questions or issues, please open an issue in this repository.
