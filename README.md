# AmarZero 🧠🎯

A Vision-Based Reinforcement Learning Agent for Real-Time Object Tracking & Planning

This repository contains the full implementation and research paper of **AmarZero**, a hybrid model-based reinforcement learning system for object tracking tasks in both single and multi-object scenarios.

---

## 📄 Research Paper

🔗 [Read the paper on Zenodo](https://doi.org/10.5281/zenodo.15208166)  
📄 `AmarZero-paper.pdf` is also included in this repository.

---

## 🚀 Key Features

- ✅ No need for external object detectors  
- 🔍 Uses **Vision Transformers (ViT)** for perception  
- 🔄 **Kalman Filtering** for robust tracking under occlusion  
- 🌲 **Monte Carlo Tree Search (MCTS)** for planning  
- 🧠 **World Models** for environment prediction  
- 🧪 Evaluated on **MOT17** dataset  

---

## 📁 Project Structure

AmarZero/ ├── single_task/ │ └── amarzero_single.py # Single-object tracking implementation ├── multi_task/ │ └── amarzero_multi.py # Multi-object tracking implementation ├── AmarZero-paper.pdf # Published research paper ├── README.md # Project overview

---

## 📊 Evaluation

The scripts automatically generate CSV reports containing predicted and ground-truth positions with tracking error.  
You will find:

- `evaluation.csv` → for single-object tracking  
- `evaluation_multi.csv` → for multi-object tracking

---

## 📫 Contact

- **Author:** Amar Ahmed Hamed  
- **Email:** aammaarrah10@gmail.com  
- **ORCID:** [https://orcid.org/0009-0005-5278-5123](https://orcid.org/0009-0005-5278-5123)  
- **Kaggle:** [@amarahmedhamed](https://www.kaggle.com/amarahmedhamed)
