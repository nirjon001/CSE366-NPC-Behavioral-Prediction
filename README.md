# NPC Behavioral Prediction — GA-based NAS vs. Ensemble Learning

**Research paper (CSE 366):**
*Optimizing NPC Behavioral Prediction: A Comparative Study of Genetic Algorithm-based
Neural Architecture Search and Ensemble Learning.*

A comparative machine-learning study that predicts the **action type of a
Non-Player Character (NPC)** in a game from behavioral state features, benchmarking
**nine models** — from probabilistic baselines to a **Genetic-Algorithm-optimized
Neural Architecture Search (GA-NAS)** network.

---

## Motivation

Non-Player Characters (NPCs) are central to player engagement and immersion.
Static, rule-based NPC systems fail to adapt to changing player strategies, which
makes gameplay predictable and repetitive. This work treats NPC decision-making as a
**multi-class classification problem** and asks which model family predicts NPC
actions best — hand-designed neural networks, evolved (NAS) networks, ensembles, or
classical learners.

---

## Dataset

A purpose-built dataset of **80,000 annotated behavioral records**
(`npc_behavior_dataset_80k.csv`). A smaller 1,000-row file
(`npc_behavior_dataset.csv`) is included as a quick sample.

**Features (9):**

| Feature | Type | Description |
|---|---|---|
| `Environment_State` | categorical | Calm, Combat, Stealth, Chaotic |
| `Opponent_Strategy` | categorical | Aggressive, Balanced, Defensive, Random |
| `Sensory_Input_Level` | continuous [0,1] | NPC perceptual acuity (normalized) |
| `Decision_Time` | continuous | Time (s) before action selection |
| `Policy_Confidence` | continuous [0,1] | NPC internal policy-certainty score |
| `Reward_Score` | integer | Environmental reward signal |
| `Human_Likeness_Score` | continuous [0,1] | Behavioral believability rating |
| `Behavioral_Diversity` | continuous [0,1] | Variability in NPC behavior patterns |
| `NPC_Action_Type` | categorical (target) | Attack, Defend, Hide, Explore, Communicate |

A **20,000-sample stratified subset** was used to train and evaluate all models in a
controlled setting.

---

## Models compared (9)

1. **Gaussian Naive Bayes**
2. **Multinomial Naive Bayes**
3. **Support Vector Machine** (RBF kernel)
4. **K-Nearest Neighbors**
5. **Decision Tree**
6. **Random Forest**
7. **Gradient Boosting**
8. **Architecture 5** — fixed fully-connected network (128-64-24 hidden layers)
9. **GA-NAS** — neural architecture evolved by a genetic algorithm
   (population 6, 4 generations; chromosomes encode 7–25 hidden layers of 4–32 neurons)

The genetic algorithm uses elitist selection (top-2), single-point crossover, and
mutation (p = 0.3); fitness is `1 − loss`.

---

## Key findings

- **GA-NAS** achieved the highest accuracy and F1-score, showing that automated
  architecture optimization is practical for game-AI classification.
- **Architecture 5 (128-64-24)** closely matched GA-NAS, so well-designed fixed
  architectures remain competitive when the search budget is limited.
- **Random Forest** and **Gradient Boosting** were the strongest non-neural models,
  offering the best accuracy/latency compromise.
- **KNN** was surprisingly competitive, which the authors attribute to strong local
  geometric structure (similar game states cluster in feature space).
- **Naive Bayes / Decision Tree** are the fastest (suitable for high-frequency game
  loops) but trade off accuracy.
- For offline / asynchronous pipelines, GA-NAS or Architecture 5 are recommended;
  for latency-critical loops, the tree-based models are preferred.

---

## Repository contents

| File | Description |
|---|---|
| `NPC_Research_Paper.pdf` | The full research paper. |
| `npc_behavior_dataset_80k.csv` | Main dataset — 80,000 records. |
| `npc_behavior_dataset.csv` | 1,000-record sample of the dataset. |

---

## Tech stack

`Python`, `TensorFlow / Keras`, `scikit-learn`, `pandas`, `NumPy`, `Matplotlib`

---

## Citation / reference

> *Optimizing NPC Behavioral Prediction: A Comparative Study of Genetic
> Algorithm-based Neural Architecture Search and Ensemble Learning.*
> CSE 366 research paper, East West University.

See the **References** section of the PDF for the full bibliography
(Cortes & Vapnik 1995; Cover & Hart 1967; Quinlan 1986; Holland 1992;
Zoph & Le 2016; Yannakakis & Togelius 2018; and others).
