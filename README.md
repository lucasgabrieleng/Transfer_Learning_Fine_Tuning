# Transfer_Learning_Fine_Tuning
Generative AI and Large Language Models (LLMs) for specialized Text Sentiment Analysis

# Efficient Fine-Tuning of LLMs for Sentiment Analysis

This repository demonstrates the implementation of an **Efficient Fine-Tuning** pipeline to adapt a Large Language Model (LLM) for specialized **Text Sentiment Analysis**. The project focuses on maximizing model performance while significantly reducing computational and memory overhead.

## 🚀 Project Overview

The core of this project is the adaptation of the **Llama-2-7b-chat-hf** model. By leveraging state-of-the-art optimization techniques, the pipeline allows for high-quality sentiment classification and text generation based on initial evaluations, even on accessible GPU hardware.

## 🛠️ Technical Expertise & Features

### 1. Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA (Low-Rank Adaptation):** Implemented to update only a fraction of the model's parameters. By approximating weight matrices with low-rank decompositions (Rank=32, Alpha=16), we maintain model expressiveness while drastically reducing storage and memory requirements.

### 2. Advanced Quantization (QLoRA)
- **4-bit NormalFloat (NF4):** Utilized `bitsandbytes` to load the 7-billion parameter model in 4-bit precision.
- **Mixed Precision (FP16):** Calculations are performed in 16-bit floating-point to balance numerical stability and processing speed.

### 3. Training Orchestration
- **Supervised Fine-Tuning (SFT):** Used the `trl` library's `SFTTrainer` for efficient instruction tuning.
- **Memory Management:** Employed **Gradient Checkpointing** and **Paged AdamW (32-bit)** optimizers to prevent VRAM overflow during training.
- **Sequence Grouping:** Optimized training efficiency by grouping sequences of similar lengths to minimize padding waste.

### 4. End-to-End Pipeline
- **Inference:** Integrated a specialized text-generation pipeline that processes raw text prompts into structured sentiment insights.
- **Model Merging:** Demonstrated the process of merging LoRA weights back into the base model for a unified, deployment-ready final architecture.

## 📦 Stack
- **Languages:** Python
- **Libraries:** Transformers, PEFT, TRL, BitsAndBytes, Datasets, PyTorch, Numba
- **Hardware Target:** NVIDIA GPUs (Optimized for A100/H100 via Colab/Cloud environments)

## 📖 How to Use
The primary logic is contained within the Jupyter Notebook `Transfer_Learning_Fine_Tuning.ipynb`. It covers:
1. Environment setup and package installation.
2. Dataset loading and preprocessing.
3. Quantization and LoRA configuration.
4. Model training and performance logging.
5. Inference testing with custom prompts.
6. Saving and merging the final fine-tuned model.

---
*Project developed as part of advanced studies in Generative AI and Industrial AI Applications.*

