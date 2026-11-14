import streamlit as st
import joblib

@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('kazakh_profanity_model.pkl')
    return vectorizer, model

vectorizer, model = load_models()

st.title("Profanity Detection")
st.write("Бонусный проект KBTU | DSSC5113 | Диас Момбеков (25MD0202)")

# Поле для ввода текста
user_input = st.text_area("Enter your text:", height=100)

if st.button("Check Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec_text = vectorizer.transform([user_input])
        prediction = model.predict(vec_text)[0]
        probability = model.predict_proba(vec_text)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"**Verdict: PROFANITY FOUND (Profanity/Offensive)**")
            st.write(f"Model Confidence: {probability:.2%}")
        else:
            st.success(f"**Verdict: TEXT IS CLEAN (Neutral)**")
            st.write(f"Profanity Probability: {probability:.2%}")

st.caption("Model trained on Logistic Regression + TF-IDF (char_wb n-grams). Detects Kazakh profanity, Russian borrowings, and masked words (with asterisks).")