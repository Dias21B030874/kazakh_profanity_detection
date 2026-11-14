import joblib
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('kazakh_profanity_model.pkl')

text = "Өте нашар!"
score = model.predict_proba(vectorizer.transform([text]))[0][1]
print(f"Profanity Probability: {score:.2%}")

text2 = "Пиздец блять еба!"
score = model.predict_proba(vectorizer.transform([text2]))[0][1]
print(f"Profanity Probability: {score:.2%}")