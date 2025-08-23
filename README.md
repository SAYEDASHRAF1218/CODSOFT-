# 🎬 Movie Genre Prediction

This project is an **Internship Task** that predicts **movie genres** from **plot summaries** using multiple machine learning classifiers. It applies **Natural Language Processing (NLP)** for text preprocessing and feature extraction, and evaluates models based on classification performance.

## 📌 Features
- Preprocesses text using NLTK (stopwords removal, lemmatization)
- Converts text to numerical features using **TF-IDF Vectorization**
- Handles **multi-label classification** for genres
- Implements multiple classifiers:
  - Naive Bayes
  - Logistic Regression
  - Linear SVC (One-vs-Rest strategy)
- Evaluates models with **Accuracy**, **Classification Report**, and **ROC-AUC**

## 📂 Files
- `moviefinal.ipynb` – Main Jupyter notebook with complete implementation

## 🛠 Requirements
Make sure you have the following Python libraries installed:
```bash
pandas
numpy
nltk
matplotlib
scikit-learn
```
To install all dependencies:
```bash
pip install pandas numpy nltk matplotlib scikit-learn
```

## ▶️ How to Run
### **Option 1: Run in Google Colab**
1. Open the notebook in Google Colab: [Upload `moviefinal.ipynb` to Colab]
2. Install dependencies if prompted
3. Run all cells sequentially

### **Option 2: Run Locally**
1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd <repo-name>
   ```
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook moviefinal.ipynb
   ```
3. Run all cells

## 📊 Example Outputs
- Preprocessing results (cleaned text)
- TF-IDF feature shape
- Classification performance metrics (Accuracy, ROC-AUC)
- Visualization plots for genre distribution and evaluation
