# 🧠 Multilingual Toxicity Classification System

## 📌 Problem Statement
This project is developed as part of **NeuroLogic '26: Global NLP Datathon**.  
The objective is to classify text into:
- **0 → Non-toxic**
- **1 → Toxic**

The dataset contains multilingual text (English + Hindi), making the problem more challenging.

---

## 📊 Dataset
- **Training Dataset:** `toxic_labeled.xlsx`
- **Evaluation Dataset:** `toxic_no_label_evaluation.xlsx`

Each record contains:
- `text` → input sentence
- `label` → (only in training data)

---

## 🧠 Approach

### 1. Data Preprocessing
- Converted text to lowercase  
- Removed URLs and unwanted characters  
- Preserved Hindi + English text  

---

### 2. Feature Engineering
- Used **TF-IDF Vectorization**
- Parameters:
  - `max_features = 5000`
  - `ngram_range = (1,2)`

---

### 3. Model
- **Logistic Regression**
- Efficient and suitable for text classification

---

### 4. Training Strategy
- Split training data into:
  - 80% Training
  - 20% Validation
- Evaluated using ROC-AUC metric

---

## 📈 Model Performance

- **ROC-AUC Score:** **0.94**

This indicates excellent performance in distinguishing toxic and non-toxic text.

### 📊 ROC Curve

![ROC Curve]

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
