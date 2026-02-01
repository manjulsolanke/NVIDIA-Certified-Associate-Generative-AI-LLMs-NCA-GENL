# NVIDIA Certified Associate Generative AI LLMs (NCA-GENL)
---
Hi, I’m **Manjul Solanke**. I have successfully completed the **NVIDIA NCA-GENL [Link](https://www.linkedin.com/feed/update/urn:li:activity:7414669243301908481/)**.

---
### 📖 About this Guide  
Collection of study materials, hands-on guides, and reference notes for preparing NCA-GENL. This repository is my attempt to make preparation easier for future aspirants.  

---

## 📌 Table of Contents
- [1. Trustworthy & Secure AI](#1-trustworthy--secure-ai)
- [2. Model Evaluation & Validation](#2-model-evaluation--validation)
- [3. Deep Learning Fundamentals](#3-deep-learning-fundamentals)
- [4. NLP & Language Models](#4-nlp--language-models)
- [5. Training & Data Pipelines](#5-training--data-pipelines)
- [6. Distributed Training & Scaling](#6-distributed-training--scaling)
- [7. NVIDIA GPU & AI Ecosystem](#7-nvidia-gpu--ai-ecosystem)
- [8. End-to-End AI Lifecycle](#8-end-to-end-ai-lifecycle)
- [9. References](#9-references)

---

## 1. Trustworthy & Secure AI

- **Confidential Computing in AI Systems**
  - Protecting data *in use* using Trusted Execution Environments (TEE)

- **Certification in Trustworthy AI Systems**
  - Compliance with standards for:
    - Security
    - Privacy
    - Fairness
    - Explainability
    - Robustness

- **Non-Discrimination & Bias**
  - Bias sources
  - Bias detection
  - Bias mitigation techniques

- **Hallucination in AI / LLMs**
  - What it is
  - Why it happens
  - Mitigation strategies

---

## 2. Model Evaluation & Validation

- **Model Evaluation**
  - Offline evaluation
  - Online evaluation

- **Evaluation Metrics**
  - Classification
  - Regression
  - Generative models

- **Machine Translation Evaluation**
  - BLEU Score

- **A/B Model Testing**
  - Shadow testing
  - Canary deployment
  - Statistical significance

- **Model Evaluation in Production**
  - Data drift
  - Concept drift
  - Performance decay

---

## 3. Deep Learning Fundamentals

- **Backpropagation**
- **Activation Functions**
  - ReLU
  - GELU
  - Sigmoid
  - Tanh

- **Loss Functions**
  - Cross-Entropy Loss
  - Mean Squared Error (MSE)
  - KL Divergence

- **Layer Fusion**
  - Performance optimization during inference

---

## 4. NLP & Language Models

### Tokenization
- Word tokenization
- Sub-word tokenization (BPE, WordPiece)
- Character tokenization

### Token Representation
- One-Hot Encoding
- Dense Vector Embeddings
- Cosine Similarity (dot product of vectors)

### Word Embeddings
- Word2Vec
  - CBOW (Continuous Bag of Words)
  - Skip-Gram

### Language Model Concepts
- Positional Encoding
- Context length vs computation trade-off
- Foundation Models
  - Large-scale pretraining
  - Multi-task adaptability

---

## 5. Training & Data Pipelines

- **EDA (Exploratory Data Analysis)**
  - Data distribution
  - Missing values
  - Outliers

- **Data Preparation Pipeline**

- **Transfer Learning**
- Pretraining
- Fine-tuning

---

## 6. Distributed Training & Scaling

- **ALL-Reduce**
- Gradient synchronization across GPUs

- **Token Limit Trade-offs**
- Increasing input tokens:
  - Improves context awareness
  - Increases memory usage
  - Increases compute cost and latency

---

## 7. NVIDIA GPU & AI Ecosystem

- **NVIDIA NeMo Toolkit**
- Training and serving LLMs
- NLP and speech models

- **TensorRT**
- Inference optimization
- Kernel fusion
- Precision reduction (FP16 / INT8)

- **RAPIDS**
- GPU-accelerated data processing
- Mandatory for heavy data-parallel computations

- **cuDF**
- GPU DataFrame library (Pandas-like)

- **cuML**
- GPU-accelerated ML algorithms

---

## 8. End-to-End AI Lifecycle

- Data ingestion
- Data preprocessing
- Model training
- Model evaluation
- Optimization (quantization, TensorRT, layer fusion)
- Deployment
- Monitoring & continuous improvement

---

## 9. References
- [Offical Exam Guide](https://nvdam.widen.net/s/rpdddpdgtc/nvt-certification-exam-study-guide-gen-ai-llm-3262644-r7-web)
- https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy
- [Transformer & LLM research papers](https://arxiv.org/abs/1706.03762)
- [NCA-GENL Cheat-Sheet](https://drive.google.com/file/d/1t3QBwbH6MwT6l49nHuqrkhpFHX6Lc6Ty/view?usp=sharing)

---
### ⚠️ Disclaimer  

This repository and its contents have been created to the best of my knowledge and personal experience while preparing for the NVIDIA NCA-GENL certification.  

- This is not an official NVIDIA study guide.  
- The official exam study guide provided by NVIDIA is the best and primary source of truth for exam preparation.  

If you find errors or have better references, feel free to suggest improvements. PRs are always welcome 🙂
