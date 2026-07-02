import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.card {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    background-color: var(--background-color);
    border: 1px solid #ddd;
}

.result-real {
    border-left: 6px solid green;
    padding: 15px;
}

.result-fake {
    border-left: 6px solid red;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

@st.cache_resource
def load_model():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["class"] = 0
    true["class"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1)
    data = data[['text', 'class']]

    data['text'] = data['text'].apply(clean_text)

    x = data['text']
    y = data['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    model = LogisticRegression()
    model.fit(xv_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(xv_test))

    return model, vectorizer, accuracy, data

model, vectorizer, accuracy, data = load_model()


st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Statistics", "🧾 History"])


if "history" not in st.session_state:
    st.session_state.history = []

if page == "🏠 Home":

    st.title("🧠 Fake News Detection System")

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Enter News Text")
        news_input = st.text_area("Paste your news here...")
        predict_btn = st.button("🔍 Predict")
        st.markdown('</div>', unsafe_allow_html=True)


    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Model Info")
        st.write(f"Accuracy: **{round(accuracy*100,2)}%**")
        st.write("Model: Logistic Regression")
        st.write("Method: TF-IDF")
        st.markdown('</div>', unsafe_allow_html=True)


    if predict_btn:
        if news_input.strip() == "":
            st.warning("⚠️ Please enter text")
        else:
            with st.spinner("Analyzing..."):
                cleaned = clean_text(news_input)
                vector = vectorizer.transform([cleaned])
                prediction = model.predict(vector)[0]
                prob = model.predict_proba(vector)
                confidence = np.max(prob) * 100

            result_text = "REAL" if prediction == 1 else "FAKE"

            
            st.session_state.history.append({
                "text": news_input[:100],
                "result": result_text,
                "confidence": round(confidence,2)
            })

            if prediction == 1:
                st.success(f"✅ REAL News (Confidence: {round(confidence,2)}%)")
            else:
                st.error(f"❌ FAKE News (Confidence: {round(confidence,2)}%)")


elif page == "📊 Statistics":

    st.title("📊 Dataset Statistics")

    real_count = data[data["class"] == 1].shape[0]
    fake_count = data[data["class"] == 0].shape[0]

    col1, col2 = st.columns(2)

    col1.metric("Real News", real_count)
    col2.metric("Fake News", fake_count)

    st.bar_chart({
        "Real": [real_count],
        "Fake": [fake_count]
    })


elif page == "🧾 History":

    st.title("🧾 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")
    else:
        for item in st.session_state.history[::-1]:
            st.markdown(f"""
            <div class="card">
            <b>Text:</b> {item['text']}...<br>
            <b>Result:</b> {item['result']}<br>
            <b>Confidence:</b> {item['confidence']}%
            </div>
            """, unsafe_allow_html=True)