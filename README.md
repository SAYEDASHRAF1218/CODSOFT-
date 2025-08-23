# 🎬 Movie Genre Prediction with Multiple Classifiers + Visualization

## 📌 Overview
This project is part of my **Internship Task** on **Movie Genre Classification from Plot Summaries**.  
The model predicts one or more genres for a movie based on its **plot summary** using **Natural Language Processing (NLP)** and **Machine Learning classifiers**.  

Both **Logistic Regression** and **Support Vector Machine (SVM)** classifiers are trained and evaluated, with performance compared visually.  

---

## 📂 Dataset
- **train_data.txt** → Contains `ID`, `Title`, `Genres`, and `Plot`.  
- **test_data.txt** → Contains `ID`, `Title`, and `Plot` (without genres).  

Example training data:

| ID | Title                          | Genres    | Plot (short) |
|----|--------------------------------|-----------|--------------|
| 1  | Oscar et la dame rose (2009)   | drama     | Listening in to a conversation ... |
| 2  | Cupid (1997)                   | thriller  | A brother and sister with a past... |
| 3  | Young, Wild and Wonderful ...  | adult     | As the bus empties the students... |

---

## 🛠️ Steps in the Project
1. **Text Preprocessing**  
   - Removed non-alphabetic characters  
   - Converted to lowercase  
   - Stopword removal and Lemmatization (NLTK)  

2. **Multi-Label Encoding**  
   - Genres converted into multi-hot encoded format using `MultiLabelBinarizer`.  

3. **Feature Extraction**  
   - `TF-IDF Vectorizer` with 5000 features.  

4. **Model Training**  
   - Logistic Regression (One-vs-Rest)  
   - Linear SVM (One-vs-Rest)  

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision, Recall, F1-score (per genre)  
   - ROC AUC (for Logistic Regression)  

6. **Visualization**  
   - Bar chart comparing validation accuracies of both models.  

---

## 📊 Results

### Logistic Regression
- **Accuracy:** ~0.35  
- **ROC AUC:** ~0.89  
- Performs well on genres like **documentary, drama, horror, western**, but poorly on rare genres like **fantasy, musical, war**.  

### SVM
- **Accuracy:** ~0.41  
- Stronger performance than Logistic Regression on overall classification.  

---

## 🔮 Sample Predictions
**Input:**  
- `"A young boy discovers he has magical powers and attends a wizarding school."`  
- `"A detective investigates a mysterious murder in a small town."`  

**Predictions:**  

- Logistic Regression  
  - Wizarding school → `()` (no confident prediction)  
  - Detective mystery → `mystery`  

- SVM  
  - Wizarding school → `drama`  
  - Detective mystery → `mystery, thriller`  

---

## 📈 Visualization
Bar chart comparing model accuracies:  

- Logistic Regression → **0.35**  
- SVM → **0.41**

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/SAYEDASHRAF1218/CODSOFT.1.git
