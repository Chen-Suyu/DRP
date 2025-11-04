<div align="center">

# 🧠 Difference-aware Reasoning Personalization (DRP)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is the implementation of the **Difference-aware Reasoning Personalization (DRP)** framework,  
which extends the *Difference-aware Personalization Learning (DPL)* paradigm by introducing  
a **reasoning-enhanced difference extractor** to capture richer stylistic and semantic differences.

</div>

---

## 🌐 Overview

![DRP Framework](fig/drp_framework.png)

DRP builds upon the idea of DPL to compare each target user's review history  
with representative peer users, but removes DPL's restriction of *fixed stylistic dimensions*.  
Instead, DRP employs a reasoning-enhanced extractor (`LLM_E`) to freely generate multi-dimensional  
difference representations (δ), allowing the model to learn more nuanced user distinctions.

---

## ⚙️ Environment Setup

```bash
conda create -n drp python=3.11
conda activate drp
pip install -r requirements.txt
