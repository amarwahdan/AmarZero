# AmarZero 🧠🎯  
**A Vision-Based Reinforcement Learning Agent for Real-Time Object Tracking & Planning**  

Welcome to the **AmarZero** repository! This project implements a hybrid model-based reinforcement learning system designed for real-time object tracking tasks, in both single-object and multi-object scenarios.

## 📄 Research Paper  
🔗 [Read the research paper on Zenodo](https://zenodo.org/records/15208167)  
The paper is also included as `AmarZero-paper.pdf` in this repository for reference.

---

## 🚀 Key Features  
- ✅ **No need for external object detectors**  
- 🔍 Utilizes **Vision Transformers (ViT)** for advanced perception  
- 🔄 **Kalman Filtering** for robust tracking, even under occlusion  
- 🌲 **Monte Carlo Tree Search (MCTS)** for enhanced planning  
- 🧠 **World Models** for more accurate environment prediction  
- 🧪 Evaluated on the **MOT17 dataset**

---

## 📁 Project Structure  
This repository is structured as follows:

AmarZero/ │ ├── single_task/ # Directory for single-object tracking │ └── amarzero_single.py # Implementation for single-object tracking │ ├── multi_task/ # Directory for multi-object tracking │ └── amarzero_multi.py # Implementation for multi-object tracking │ ├── AmarZero-paper.pdf # Published research paper ├── README.md # Project overview and details


---

## 📊 Evaluation  
The tracking scripts automatically generate CSV reports that contain predicted and ground-truth positions along with tracking error. The reports generated are:

- `evaluation.csv` → Single-object tracking results  
- `evaluation_multi.csv` → Multi-object tracking results

---

## 📫 Contact  
**Author:** Amar Ahmed Hamed  
📧 Email: [aammaarrah10@gmail.com](mailto:aammaarrah10@gmail.com)  
🌐 ORCID: [https://orcid.org/0009-0005-5278-5123](https://orcid.org/0009-0005-5278-5123)  
📊 Kaggle: [@amarahmedhamed](https://www.kaggle.com/amarahmedhamed)

---

Feel free to open an issue or contribute to the project! 🚀

