import argparse
import pandas as pd
import openai
import time
import os
import sys
from typing import List, Dict, Any, Optional, Callable
import logging
from tqdm import tqdm

# Add the project root directory to Python path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_model_config, get_supported_languages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure UTF-8 encoding on Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def validate_languages(languages: List[str]) -> List[str]:
    """Validate language codes against supported languages."""
    supported_langs = get_supported_languages()
    valid_langs = []
    for lang in languages:
        lang = lang.strip().lower()
        if lang in supported_langs:
            valid_langs.append(lang)
        else:
            logger.warning(f"Language code '{lang}' not in predefined list, but will attempt translation anyway")
            valid_langs.append(lang)
    return valid_langs

def process_batch(texts: List[str], lang: str, prompt_template: str, model_config: Dict[str, Any]) -> List[str]:
    """Process a batch of texts for translation."""
    translations = []
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == "":
            translations.append("")
            continue
            
        try:
            prompt_full = prompt_template.replace("{lang}", lang)
            full_prompt = f"{prompt_full}\n\nText:\n{text}"

            # Updated OpenAI API call
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=[{"role": "user", "content": full_prompt}]
            )

            translation = response.choices[0].message.content.strip()
            translations.append(translation)
            
        except Exception as e:
            logger.error(f"Error translating item: {e}")
            translations.append("")  # Add empty string for failed translations
            
    return translations

def translate_dataframe(
    df: pd.DataFrame, 
    column: str, 
    languages: List[str], 
    prompt: str, 
    api_key: str,
    model: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_event: Optional[Any] = None
) -> pd.DataFrame:
    """
    Translate a specific column in a DataFrame to multiple languages.
    
    Args:
        df: Input DataFrame
        column: Column name to translate
        languages: List of target language codes
        prompt: Prompt template with {lang} placeholder
        api_key: OpenAI API Key
        model: Optional model name to override config
        progress_callback: Optional callback function(message, percentage)
        cancel_event: Optional threading.Event to signal cancellation
    
    Returns:
        DataFrame with added translation columns
    """
    openai.api_key = api_key
    model_config = get_model_config()
    
    # Override model if provided
    if model:
        model_config["model"] = model
        
    languages = validate_languages(languages)
    
    df_result = df.copy()

    if not languages:
        raise ValueError("No valid languages specified")

    supported_languages_dict = get_supported_languages()
    
    # Initialize output columns
    for lang in languages:
        out_col = f"{column}_Translated_{lang}"
        df_result[out_col] = ""

    # Create target map for prompt
    # e.g., "es": "Spanish", "fr": "French"
    targets_desc = {lang: supported_languages_dict.get(lang, lang) for lang in languages}
    targets_json_str = str(targets_desc)

    # Process in batches
    texts = df_result[column].tolist()
    
    total_items = len(texts)
    
    logger.info(f"Starting multi-language translation for {len(languages)} languages...")
    
    for i, text in enumerate(texts):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            logger.info("Translation cancelled by user")
            break

        if progress_callback:
            progress_callback(f"Translating row {i+1}/{total_items}...", (i) / total_items)
            
        if pd.isna(text) or str(text).strip() == "":
            continue

        try:
            # Construct system prompt for JSON
            system_prompt = (
                f"You are a professional translator. Translate the text into the following languages: {targets_json_str}.\n"
                "Return the result as a valid JSON object where keys are the 2-letter language codes and values are the translations.\n"
                "Example output: {\"es\": \"Hola\", \"fr\": \"Bonjour\"}\n"
                "Do NOT add any other text."
            )
            
            full_prompt = f"Text to translate:\n{text}"
            if "{lang}" not in prompt and len(prompt) > 10:
                 full_prompt = f"Style instructions: {prompt}\n\n{full_prompt}"

            client = openai.OpenAI(api_key=api_key)
            
            kwargs = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
            }
            
            # Enable JSON mode for compatible models
            if "gpt-4" in model_config["model"] or "gpt-3.5-turbo-1106" in model_config["model"] or "gpt-3.5-turbo-0125" in model_config["model"]:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            try:
                translations_map = json.loads(content)
                
                # Assign to columns
                for lang in languages:
                    translation = translations_map.get(lang, "")
                    out_col = f"{column}_Translated_{lang}"
                    df_result.at[i, out_col] = translation
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response for row {i}: {content}")
                
            # Rate limit
            time.sleep(model_config["rate_limit"])

        except openai.AuthenticationError as e:
             logger.error(f"Authentication error: {e}")
             raise e
        except Exception as e:
            logger.error(f"Error translating row {i}: {e}")
            
    if progress_callback:
        progress_callback("Translation complete!", 1.0)
            
    return df_result

def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="AI CSV Translator")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--column', required=True, help='Column to translate')
    parser.add_argument('--languages', required=True, help='Target languages (comma-separated ISO codes)')
    parser.add_argument('--prompt', required=True, help='Custom prompt template with {lang} placeholder')
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter')
    parser.add_argument('--encoding', default='utf-8', help='File encoding')
    args = parser.parse_args()

    # --- Setup ---
    try:
        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, delimiter=args.delimiter, encoding=args.encoding)
        logger.info(f"Successfully read {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        sys.exit(1)

    # --- Translation ---
    try:
        languages_list = [l.strip() for l in args.languages.split(',')]
        translated_df = translate_dataframe(
            df=df,
            column=args.column,
            languages=languages_list,
            prompt=args.prompt,
            api_key=args.api_key
        )
        
        # --- Save results ---
        logger.info(f"Saving results to {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        translated_df.to_csv(args.output, index=False, sep=args.delimiter, encoding=args.encoding)
        logger.info("✓ Translation complete!")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
