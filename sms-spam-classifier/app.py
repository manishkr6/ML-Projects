import streamlit as st
import pickle

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# -------------------------------
# Custom CSS (Modern UI)
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #AAAAAA;
        }
        .stTextArea textarea {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton button {
            width: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="title">📩 SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# Input Box
# -------------------------------
input_sms = st.text_area("✍️ Enter your message:", height=150)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Transform input
        transformed_sms = vectorizer.transform([input_sms])

        # Prediction
        result = model.predict(transformed_sms)[0]

        # Output UI
        st.write("")

        if result == 1:
            st.error("🚫 Spam Message Detected!")
        else:
            st.success("✅ Not Spam (Safe Message)")

# -------------------------------
# Footer
# -------------------------------
st.write("")
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")