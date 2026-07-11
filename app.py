import streamlit as st
import requests

st.set_page_config(
    page_title="Transformer Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("English → French Translator")
st.write("Powered by your Transformer model")

text = st.text_area(
    "Enter English Text",
    height=150
)

if st.button("Translate"):
    if text.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:8000/translate",
                json={"text": text}
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Translation")
                st.write(result["translation"])

            else:
                st.error("FastAPI returned an error.")

        except Exception:
            st.error("Cannot connect to FastAPI server.")
    else:
        st.warning("Please enter some text.")