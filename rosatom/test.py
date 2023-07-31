from django.shortcuts import render
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
BASE_DIR = Path(__file__).resolve().parent.parent
rosatom = joblib.load(BASE_DIR / "rosatom/rosatom.joblib")
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
text = ['very bad movie']
print(text)
text = vectorizer.transform(text)
text = text.toarray()
prediction = rosatom.predict(text)
print(text)
print(prediction)