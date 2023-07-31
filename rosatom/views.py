from django.shortcuts import render
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


def index(request):
    return render(request, "base.html", {"prediction": "None"})


def prediction(request):
    BASE_DIR = Path(__file__).resolve().parent.parent
    rosatom = joblib.load(BASE_DIR / "rosatom/rosatom.joblib")
    vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
    text = request.POST["review_text"]
    text_array = [f'{text}']
    text_array = vectorizer.transform(text_array)
    text_array = text_array.toarray()
    prediction = rosatom.predict(text_array)
    rating = round(((rosatom.predict_proba(text_array))[0][1])*10)
    return render(request, "base.html", {"prediction": prediction[0], "review": text, "rating": rating})
