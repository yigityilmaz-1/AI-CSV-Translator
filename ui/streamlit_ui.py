import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(page_title="AI CSV Translator", layout="centered")
st.title(" AI-Powered CSV Translator")

st.markdown("""
Upload a CSV file, choose the column you want to translate, enter your custom prompt using `{lang}` as a placeholder for the language code, and specify target languages separated by commas (e.g., `tr,de,fr`).
""")

api_key = st.text_input(" OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded CSV", df.head())
    column_to_translate = st.selectbox(" Column to Translate", df.columns)
    prompt = st.text_area(" Custom Prompt (use {lang} where language should go)",
                          "This text is part of a multilingual dataset. Please translate it into {lang}.")
    languages = st.text_input(" Target Languages (ISO codes, comma-separated)", "tr,de")

    if st.button(" Start Translation"):
        with st.spinner("Translation in progress. This may take a while..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as input_tmp:
                df.to_csv(input_tmp.name, index=False)
                input_path = input_tmp.name

            output_path = os.path.join("output", "translated_result.csv")
            os.makedirs("output", exist_ok=True)

            command = [
                "python", "./core/translator_generic.py",
                "--input", input_path,
                "--output", output_path,
                "--column", column_to_translate,
                "--languages", languages,
                "--prompt", prompt,
                "--api_key", api_key
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Translation complete ")
                    st.download_button(" Download Translated CSV", open(output_path, "rb").read(),
                                       file_name="translated_result.csv", mime="text/csv")
                else:
                    st.error("An error occurred during translation.")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
