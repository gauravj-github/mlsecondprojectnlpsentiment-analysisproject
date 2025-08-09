import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords_tokenize(txt):
    if not isinstance(txt, str):
        return txt
    tokens = word_tokenize(txt)
    filtered = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered)

# ---------------------------
#  Load model and vectorizer
# ---------------------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ---------------------------
#  Label mapping
# ---------------------------
label_list = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
label_map = {label: idx for idx, label in enumerate(label_list)}


st.title("Sentiment / Emotion Prediction App")
st.write("Enter text and get emotion + numeric label")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    # Preprocess
    cleaned_text = remove_stopwords_tokenize(user_input)
    
    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict
    pred = model.predict(vectorized_text)[0]  # could be number or label depending on training
    
    # If prediction is label, convert to number
    if isinstance(pred, str):
        pred_label = pred
        pred_number = label_map[pred_label]
    else:
        pred_number = int(pred)
        pred_label = label_list[pred_number]
    
    st.success(f"Predicted: **{pred_label}**  (Label Number: {pred_number})")
