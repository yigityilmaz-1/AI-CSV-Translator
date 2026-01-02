import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import logging
import traceback
import threading
import time
import streamlit.components.v1 as components

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_supported_languages, get_supported_file_types, get_model_config
from core.translator_generic import translate_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Page config
st.set_page_config(
    page_title="AI CSV Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme customization
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    model_config = get_model_config()
    model_options = model_config.get("available_models", ["gpt-3.5-turbo", "gpt-4"])
    # Find index of default model
    default_index = 0
    if model_config["model"] in model_options:
        default_index = model_options.index(model_config["model"])
        
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=default_index
    )
    
    batch_size = st.slider("Batch Size", 1, 50, model_config["batch_size"])
    rate_limit = st.slider("Rate Limit (sleep per item)", 0.0, 5.0, model_config["rate_limit"], 0.1)
    max_parallel = st.slider("Max Parallel Clicks (Speed)", 1, 20, model_config.get("max_parallel_requests", 5))
    
    # Config overrides for this session
    req_config = {
        "batch_size": batch_size,
        "rate_limit": rate_limit,
        "max_parallel_requests": max_parallel
    }
    
# Main content
st.title("🌐 AI-Powered CSV Translator")

# File upload
uploaded_file = st.file_uploader(
    "📄 Upload File",
    type=get_supported_file_types(),
    help=f"Supported formats: {', '.join(get_supported_file_types())}"
)

if uploaded_file:
    try:
        # Read file based on extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # CSV specific settings
        if file_ext == '.csv':
            col1, col2 = st.columns(2)
            with col1:
                delimiter = st.selectbox(
                    "CSV Delimiter",
                    options=[',', ';', '\t', '|'],
                    format_func=lambda x: {
                        ',': 'Comma (,)',
                        ';': 'Semicolon (;)',
                        '\t': 'Tab',
                        '|': 'Pipe (|)'
                    }[x]
                )
            with col2:
                encoding = st.selectbox(
                    "File Encoding",
                    options=['utf-8', 'latin1', 'cp1252'],
                    format_func=lambda x: {
                        'utf-8': 'UTF-8',
                        'latin1': 'Latin-1',
                        'cp1252': 'Windows-1252'
                    }[x]
                )
            
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
        else:  # Excel files
            df = pd.read_excel(uploaded_file)
        
        # Preview
        st.write("### 📊 Preview of Uploaded File")
        st.dataframe(df.head())
        
        # heuristic to find best column
        best_col_idx = 0
        max_avg_len = 0
        
        for i, col in enumerate(df.columns):
            # Skip likely non-content columns by name
            lower_col = col.lower()
            if "id" in lower_col and "content" not in lower_col: continue
            if "date" in lower_col: continue
            if "guid" in lower_col: continue
            if "code" in lower_col: continue # 'common_languages' code is usually not what we want
            
            # Check content
            try:
                if df[col].dtype == 'object':
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > max_avg_len:
                        max_avg_len = avg_len
                        best_col_idx = i
            except:
                pass

        # Column selection
        column_to_translate = st.selectbox(
            "Column to Translate", 
            df.columns,
            index=best_col_idx,
            help="Select the column containing the text you want to translate"
        )
        
        # Language selection
        supported_langs = get_supported_languages()
        
        # Source language selection
        st.write("### Source Language")
        st.write("Enter the ISO code of the source language (e.g., 'en' for English, 'fr' for French)")
        source_lang = st.text_input(
            "Source Language ISO Code",
            value="en",
            help="Enter the ISO 639-1 language code of your source text"
        )
        
        # Show supported languages as reference
        with st.expander("📚 Supported Language Codes Reference"):
            st.write("Common ISO 639-1 language codes:")
            lang_cols = st.columns(3)
            for i, (code, name) in enumerate(supported_langs.items()):
                lang_cols[i % 3].write(f"**{code}** - {name}")
        
        # Target languages selection
        st.write("### Target Languages")
        st.write("Enter the ISO codes of target languages, separated by commas (e.g., 'es,fr,de')")
        target_langs_input = st.text_input(
            "Target Language ISO Codes",
            value="es,fr",
            help="Enter comma-separated ISO 639-1 language codes"
        )
        
        # Process target languages
        selected_langs = [lang.strip() for lang in target_langs_input.split(',') if lang.strip()]
        
        # Prompt customization
        source_lang_name = supported_langs.get(source_lang, source_lang)
        default_prompt = f"""You are a professional localization expert specializing in the aviation and travel industry. You are translating content for martigo.com, a premier global flight search engine.

Task:
Please translate the following {source_lang_name} text into {{lang}}.

Context & Tone: 
The translation is for an FAQ and Booking flow. The tone should be professional, user-friendly, and technically accurate.

Industry Terminology: 
You must use standard aviation industry terminology. Ensure the following terms are translated using their standard ecommerce equivalents in {{lang}}:
- Stopover / Layover (Distinguish between short transfers and long stays).
- One Way / Round Trip / Multi-City (Standard search types).
- Cabin Classes (Economy, Premium Economy, Business, First Class).
- Check-in / Boarding Pass / PNR.
- Self-Transfer (Specifically for 'Virtual Interlining' cases like Kiwi.com).

Constraint: 
> Keep the translation natural for a flight ticket website. Do not translate brand names like 'martigo'."""

        prompt = st.text_area(
            "Custom Prompt",
            value=default_prompt,
            height=300,
            help="Use {lang} as a placeholder for the target language"
        )
        
        # --- Translation Logic using Threading for Stop Functionality ---
        if "is_translating" not in st.session_state:
            st.session_state.is_translating = False
        if "translation_thread" not in st.session_state:
            st.session_state.translation_thread = None
        if "cancel_event" not in st.session_state:
            st.session_state.cancel_event = None
        if "translation_status" not in st.session_state:
            st.session_state.translation_status = {"msg": "", "progress": 0.0, "done": False, "result_df": None, "error": None}

        def run_translation_thread(df, col, langs, prmt, key, mdl, status_dict, stop_event, r_cfg, original_file_name):
            try:
                def callback(msg, prog):
                    status_dict["msg"] = msg
                    status_dict["progress"] = prog
                    
                # Generate a recovery path in the system's temporary directory
                recovery_path = os.path.join(tempfile.gettempdir(), f"recovery_{original_file_name}")
                
                res_df = translate_dataframe(
                    df=df,
                    column=col,
                    languages=langs,
                    prompt=prmt,
                    api_key=key,
                    model=mdl,
                    progress_callback=callback,
                    cancel_event=stop_event,
                    request_config=r_cfg,
                    auto_save_path=recovery_path
                )
                status_dict["result_df"] = res_df
                status_dict["done"] = True
            except Exception as e:
                status_dict["error"] = str(e)
                status_dict["done"] = True

        # Start Button
        if not st.session_state.is_translating:
            if st.button("🚀 Start Translation", type="primary"):
                if not api_key:
                    st.error("Please enter your OpenAI API key")
                elif not source_lang:
                    st.error("Please enter a source language code")
                elif not selected_langs:
                    st.error("Please enter at least one target language code")
                elif source_lang in selected_langs:
                    st.error("Source language cannot be in target languages")
                else:
                    # Initialize
                    st.session_state.is_translating = True
                    st.session_state.cancel_event = threading.Event()
                    st.session_state.translation_status = {"msg": "Starting...", "progress": 0.0, "done": False, "result_df": None, "error": None}
                    
                    # Start Thread
                    t = threading.Thread(
                        target=run_translation_thread,
                        args=(df, column_to_translate, selected_langs, prompt, api_key, selected_model, st.session_state.translation_status, st.session_state.cancel_event, req_config, uploaded_file.name)
                    )
                    st.session_state.translation_thread = t
                    t.start()
                    st.rerun()

        # Cancel Button & Progress Loop
        if st.session_state.is_translating:
            col_status, col_gif, col_stop = st.columns([3, 1, 1])
            
            with col_status:
                status_text = st.empty()
                progress_bar = st.progress(0)

            with col_gif:
                components.html(
                    """
                    <div class="tenor-gif-embed" data-postid="17461713593828281310" data-share-method="host" data-aspect-ratio="2.51515" data-width="100%">
                        <a href="https://tenor.com/view/cat-gif-17461713593828281310">Cat Sticker</a>from <a href="https://tenor.com/search/cat-stickers">Cat Stickers</a>
                    </div>
                    <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
                    """,
                    height=150
                )
                
            with col_stop:
                if st.button("🛑 Stop & Save", type="secondary"):
                    if st.session_state.cancel_event:
                        st.session_state.cancel_event.set()
                        st.warning("Stopping... please wait for current item.")
            
            # Update Loop
            status = st.session_state.translation_status
            status_text.text(status["msg"])
            progress_bar.progress(min(status["progress"], 1.0))
            
            if status["done"]:
                st.session_state.is_translating = False
                
                if status["error"]:
                    st.error(f"An error occurred: {status['error']}")
                elif status["result_df"] is not None:
                    # Success or Partial Success
                    if st.session_state.cancel_event.is_set():
                        st.warning("Translation stopped. Saving partial results.")
                    else:
                        st.success("✅ Translation complete!")
                    
                    translated_df = status["result_df"]
                    st.write("### 🏁 Translation Results")
                    st.dataframe(translated_df.head())
                    
                    csv = translated_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Translated CSV",
                        data=csv,
                        file_name=f"translated_{os.path.splitext(uploaded_file.name)[0]}.csv",
                        mime="text/csv"
                    )
                
                # Clean up
                st.session_state.translation_thread = None
                st.session_state.cancel_event = None
                
            else:
                time.sleep(0.5)
                st.rerun()

    except Exception as e:
        logger.error(f"File processing error: {e}")
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Yiğit'ten sevgilerle ❤️...</p>
    </div>
""", unsafe_allow_html=True)
