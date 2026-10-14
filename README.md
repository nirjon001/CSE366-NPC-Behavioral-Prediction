<div align="center">

# 🤖 Optimizing NPC Behavioral Prediction

### A Comparative Study of Genetic-Algorithm-based Neural Architecture Search & Ensemble Learning

**Research Paper · CSE 366**

![Field](https://img.shields.io/badge/Field-Machine%20Learning-2563eb?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Game%20AI-8b5cf6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Models](https://img.shields.io/badge/Models%20Compared-9-0e8a86?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-80%2C000%20records-brightgreen?style=for-the-badge)

*Predicting Non-Player Character actions from behavioral state features — benchmarking probabilistic, classical, ensemble, and evolved (GA-NAS) neural models.*

[Overview](#-overview) · [Dataset](#-dataset) · [Models](#-models-compared) · [Findings](#-key-findings) · [Contents](#-repository-contents)

</div>

---

## 📖 Overview

**Non-Player Characters (NPCs)** are central to player engagement and immersion.
Static, rule-based NPC systems fail to adapt to changing player strategies, which
makes gameplay predictable and repetitive.

This work treats NPC decision-making as a **multi-class classification problem** and
asks a simple question: *which model family predicts NPC actions best?*

> We compare **nine models** — probabilistic baselines, classical learners, tree
> ensembles, a hand-designed deep network, and a **Genetic-Algorithm-optimized Neural
> Architecture Search (GA-NAS)** network — on a purpose-built dataset of **80,000
> annotated behavioral records**.

---

## 📊 Dataset

| Feature | Type | Description |
|:--|:--|:--|
| `Environment_State` | categorical | Calm · Combat · Stealth · Chaotic |
| `Opponent_Strategy` | categorical | Aggressive · Balanced · Defensive · Random |
| `Sensory_Input_Level` | continuous `[0,1]` | NPC perceptual acuity (normalized) |
| `Decision_Time` | continuous | Seconds before action selection |
| `Policy_Confidence` | continuous `[0,1]` | Internal policy-certainty score |
| `Reward_Score` | integer | Environmental reward signal |
| `Human_Likeness_Score` | continuous `[0,1]` | Behavioral believability rating |
| `Behavioral_Diversity` | continuous `[0,1]` | Variability in behavior patterns |
| **`NPC_Action_Type`** | **target** | **Attack · Defend · Hide · Explore · Communicate** |

A **20,000-sample stratified subset** was used to train and evaluate all models in a
controlled experimental setting.

---

## 🧠 Models Compared

<table>
<tr><td valign="top" width="33%">

**🔹 Probabilistic**
1. Gaussian Naive Bayes
2. Multinomial Naive Bayes

</td><td valign="top" width="33%">

**🔹 Classical**
3. Support Vector Machine (RBF)
4. K-Nearest Neighbors
5. Decision Tree

</td><td valign="top" width="34%">

**🔹 Ensemble & Neural**
6. Random Forest
7. Gradient Boosting
8. Architecture 5 _(128-64-24)_
9. **GA-NAS** _(evolved)_

</td></tr>
</table>

The **genetic algorithm** evolves network topologies as chromosomes encoding
**7–25 hidden layers** of **4–32 neurons**, using **elitist selection** (top-2),
**single-point crossover**, and **mutation (p = 0.3)**, with fitness `f = 1 − loss`
over a population of **6 across 4 generations**.

---

## 🏆 Key Findings

| Rank tier | Model(s) | Why |
|:--:|:--|:--|
| 🥇 Best | **GA-NAS** | Highest accuracy & F1 — automated search beats manual design |
| 🥈 Near-best | **Architecture 5** (128-64-24) | A well-designed fixed net stays competitive with a limited search budget |
| 🥉 Strong classics | **Random Forest**, **Gradient Boosting** | Best accuracy/latency trade-off among non-neural models |
| 💡 Surprise | **K-Nearest Neighbors** | Very competitive — NPC state spaces show strong *local* geometric structure |
| ⚡ Fastest | **Naive Bayes**, **Decision Tree** | Near-instant inference, suitable for high-frequency game loops |

**Takeaways**

- 🎮 For **latency-critical** game loops → tree-based / Naive Bayes models.
- 🖥️ For **offline / asynchronous** NPC pipelines → **GA-NAS** or **Architecture 5**.
- 🧭 Neural networks dominate on accuracy; classical models dominate on speed.

Full ROC curves, Precision–Recall curves, and aggregate comparisons are in the paper.

---

## 📂 Repository Contents

| File | Description |
|:--|:--|
| 📄 [`NPC_Research_Paper.pdf`](./NPC_Research_Paper.pdf) | The complete research paper |
| 🗃️ [`npc_behavior_dataset_80k.csv`](./npc_behavior_dataset_80k.csv) | Main dataset — **80,000** records |
| 🗃️ [`npc_behavior_dataset.csv`](./npc_behavior_dataset.csv) | **1,000**-record sample of the dataset |

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib`

---

## 📚 References

The paper's bibliography includes foundational works by **Cortes & Vapnik (1995)**,
**Cover & Hart (1967)**, **Quinlan (1986)**, **Holland (1992)**, **Zoph & Le (2016)**,
**Breiman (2001)**, **Friedman (2001)**, and **Yannakakis & Togelius (2018)** —
see the *References* section of the PDF for the full list.

---

<div align="center">

### ⭐ If you find this research interesting, a star means a lot!

*Teaching NPCs to think — one architecture at a time.* 🤖

</div>
