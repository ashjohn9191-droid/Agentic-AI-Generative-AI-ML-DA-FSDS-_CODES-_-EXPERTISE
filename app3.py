import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ===============================
# NLTK Download
# ===============================
nltk.download("stopwords")

# ===============================
# Load Dataset
# ===============================
dataset = pd.read_csv(
    r"C:\Users\ANITHA\Downloads\Restaurant_Reviews.tsv",
    delimiter="\t",
    quoting=3
)

# ===============================
# Text Preprocessing
# ===============================
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

corpus = []
for review in dataset["Review"]:
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(" ".join(review))

# ===============================
# Vectorization
# ===============================
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Multiple Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC()
}

accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_results[name] = accuracy_score(y_test, y_pred)

# ===============================
# Best Model
# ===============================
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]
best_accuracy = accuracy_results[best_model_name]

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="centered"
)

# ===============================
# Custom CSS (Safe for Light UI)
# ===============================
st.markdown("""
<style>
body {
    background-color: #f9fafb;
}

.main {
    background: transparent;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.good {
    background-color: #16a34a;
    padding: 16px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.bad {
    background-color: #dc2626;
    padding: 16px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.best {
    color: #15803d;
    font-weight: 600;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Title (VISIBLE FIXED)
# ===============================
st.markdown('<div class="title">🍽️ Restaurant Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered sentiment analysis using NLP & Machine Learning</div>', unsafe_allow_html=True)

# ===============================
# Accuracy Section
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    "Model": accuracy_results.keys(),
    "Accuracy": [f"{v:.2f}" for v in accuracy_results.values()]
})

st.dataframe(accuracy_df, use_container_width=True)

st.markdown(
    f'<p class="best">🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.2f})</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# User Input
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("✍️ Enter a Restaurant Review")

review_input = st.text_area(
    "",
    placeholder="Type your restaurant experience here...",
    height=120
)

analyze = st.button("🔍 Analyze Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prediction
# ===============================
if analyze:
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        review = re.sub("[^a-zA-Z]", " ", review_input)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = " ".join(review)

        review_vector = cv.transform([review]).toarray()
        prediction = best_model.predict(review_vector)

        if prediction[0] == 1:
            st.markdown('<div class="good">✅ POSITIVE REVIEW 😊</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bad">❌ NEGATIVE REVIEW 😞</div>', unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🚀 Built with NLP • Scikit-learn • Streamlit")
