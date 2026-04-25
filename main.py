import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_excel("toxic_labeled.xlsx")
test = pd.read_excel("toxic_no_label_evaluation.xlsx")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\u0900-\u097F]", "", text)
    return text

train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train['text'])
y = train['label']

X_test = vectorizer.transform(test['text'])

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Predict
test['label'] = model.predict(X_test)

# Save output
test.to_csv("no_label.csv", index=False, encoding='utf-8-sig')

print("no_label.csv generated successfully!")