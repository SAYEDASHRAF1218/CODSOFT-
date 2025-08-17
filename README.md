# CODSOFT-

🎬 Movie Genre Prediction - Internship Task

📌 Objective

This project predicts the genre of a movie based on its plot summary using Machine Learning.
It was developed as part of an internship assignment using Python and popular ML classifiers.


---

📂 Dataset

We use the IMDb Genre Classification Dataset from Kaggle:
👉 https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Each row contains:

text → Movie plot summary (input feature)

genre → Movie genre (target label)



---

⚙️ Tech Stack

Python

Pandas

Scikit-learn

KaggleHub (for dataset download)



---

🛠️ Steps to Run

1. Clone the GitHub repository.


2. Create a virtual environment (optional but recommended).


3. Install dependencies:

pip install -r requirements.txt


4. Run the script:

python main.py


5. The program will train models and display evaluation results.


6. You can then enter your own movie plot to test predictions.




---

📊 Models Used

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)


The script trains all three models and automatically selects the best performing model.


---

📈 Results (Example)

Logistic Regression → 85%

Naive Bayes → 80%

SVM → 88%


🏆 Best Model: SVM (88%)


---

🚀 Features

Automatic download of IMDb dataset via KaggleHub

Text feature extraction using TF-IDF

Training and evaluation of multiple classifiers

Accuracy and classification report for each model

User input: enter your own movie plot and get predicted genre



---

📝 Code (main.py)

"""
Movie Genre Prediction using TF-IDF + ML Classifiers
Dataset: hijest/genre-classification-dataset-imdb
Author: Your Name | Internship Task Project
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import kagglehub

# 1. Load dataset
print("📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
df = pd.read_csv(f"{path}/IMDb.csv")[['text', 'genre']].dropna()
print("✅ Dataset loaded:", df.shape)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['genre'], test_size=0.2, random_state=42
)

# 3. Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (model, acc)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

# 5. Pick best model
best_model_name = max(results, key=lambda x: results[x][1])
best_model = results[best_model_name][0]
print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name][1]:.2f} accuracy)")

# 6. Custom user prediction
while True:
    plot = input("\nEnter a movie plot (or 'exit' to quit): ")
    if plot.lower() == "exit":
        print("👋 Exiting...")
        break
    plot_tfidf = vectorizer.transform([plot])
    print("Predicted Genre:", best_model.predict(plot_tfidf)[0])


---

📋 Requirements (requirements.txt)

pandas
scikit-learn
kagglehub


---

📜 Internship Task Deliverables

main.py → Training + prediction script

requirements.txt → Dependencies

README.md → Documentation

report.pdf → Internship report with results


✅ Upload this repository to GitHub as your internship submission.


---

👉 Bro, this text now has documentation + code + requirements in one place.
Do you also want me to write a short README.md template (GitHub style) so you can directly copy it into your repo?

