# Kazakh Profanity Detection
**KBTU | DSSC5113 Algorithms of Computer Vision | Bonus Project**
**Author:** Dias Mombekov (Student ID: 25MD0202)

## 1. Task Definition
Binary classification of Kazakh product reviews and comments. 
Output: `0` (Clean/Neutral) or `1` (Profanity/Offensive).

## 2. Dataset Preparation & Labeling Rules
The dataset was built by combining generic reviews from the **KazSAnDRA** dataset and filtering them using roots from the **Rustem-Kaimolla/kazakh-swear-words** dictionary.
- **Cleaning:** The open-source dictionary was manually audited to remove false-positive polite words (e.g., *жақсы, рахмет, адам*). Only entries with `label=1` were kept.
- **Morphology & Regex:** Regex patterns `\w*` were applied to catch all Kazakh agglutinative suffixes (e.g., *қотақ* -> *қотағым*, *қотақбас*). Short roots (like *ам*) were isolated to exact matches `\b` to prevent false positives on normal words like *амалы* or *аман*.
- **Masked Words:** Custom regex patterns `[\*\.\-]+` were introduced to catch censored Kazakh profanity (e.g., *қ*тақ*, *ш*шең*) and common Russian code-switching profanities (e.g., *б***ть*, *п*здец*).
- **Balance:** The final dataset contains ~5800 rows with a 1:2 ratio (Profane : Clean). Negative reviews (1-2 stars) without profanity were intentionally included so the model learns to differentiate between harsh criticism and actual swear words.

## 3. Data Splits
- **Train:** 70%
- **Validation:** 15% (used for tuning the decision threshold)
- **Test:** 15% (used strictly once for final evaluation)
- **Random Seed:** 42 (with stratification to preserve class balance).

## 4. Model Choice & Preprocessing
**TF-IDF + Logistic Regression** was chosen over deep learning models to ensure maximum interpretability and lightweight reproducible deployment. 
- **TF-IDF:** Uses `analyzer='char_wb'` with `ngram_range=(2, 5)`. Character-level n-grams effectively capture subword patterns, typos, and masked profanities (like *б***ь*) without relying on full-word vocabulary.
- **Logistic Regression:** Trained with `class_weight='balanced'`.

## 5. Results (Test Set)
- **Precision (Class 1):** 0.98
- **Recall (Class 1):** 0.84
- **F1-Score (Class 1):** 0.91
- **Accuracy:** 0.94

**Confusion Matrix:**
- True Negatives: 581
- False Positives: 4
- False Negatives: 46
- True Positives: 246

## 6. Error Analysis (Limitations)
1. **False Positive (Model mistakenly flagged as profanity):** 
   * *Text:* "...ақша жасау амалы ғана..."
   * *Reason:* The `char_wb` n-gram analyzer captured the substring "ам" inside the normal word "амалы" (method/way), giving it a high profanity weight despite regex boundaries in the initial labeling.
   * *Fix:* Implement an absolute exclusion list for exact safe words prior to TF-IDF vectorization, or increase the minimum n-gram range.
2. **False Negative (Model missed the profanity):** 
   * *Text:* (Creative obfuscation using unrelated symbols or emojis, e.g., "ш е ш е ң")
   * *Reason:* The user bypassed the regex patterns by inserting whitespace between every single character. The TF-IDF model evaluates these as separate 1-character tokens.
   * *Fix:* Add a preprocessing step to strip all whitespaces and punctuation before running predictions.
3. **False Negative (Contextual Insult):** 
   * *Text:* "Мал екенсің ғой" (You are an animal/cattle)
   * *Reason:* The word "мал" (cattle) is neutral in an agricultural context but highly offensive when addressed to a person. TF-IDF lacks semantic context to differentiate the two.
   * *Fix:* Fine-tune a Transformer model (like XLM-RoBERTa) instead of a bag-of-words approach to capture bidirectional context.

## 7. How to Reproduce & Run

There are three ways to test the model on custom text without retraining. 
*(Note: You may see an `InconsistentVersionWarning` if your local `scikit-learn` version is different from the Google Colab environment (1.6.1), but predictions will still execute correctly).*

### Option A: Web App (Streamlit)
A working interactive UI has been deployed using Streamlit Community Cloud.
**Live Demo:** **

### Option B: Run the test script
Simply execute the provided test script in your terminal:
```bash
python test_example.py
```

### Option C: Inline Code Execution
```bash
import joblib
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('kazakh_profanity_model.pkl')

text = "Өте нашар!"
score = model.predict_proba(vectorizer.transform([text]))[0][1]
print(f"Profanity Probability: {score:.2%}")

text2 = "Пиздец блять еба!"
score = model.predict_proba(vectorizer.transform([text2]))[0][1]
print(f"Profanity Probability: {score:.2%}")
```