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
    cancel_event: Optional[Any] = None,
    request_config: Optional[Dict[str, Any]] = None,
    auto_save_path: Optional[str] = None,
    auto_save_interval: int = 10
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
        request_config: Optional dictionary to override batch_size, rate_limit, max_parallel
        auto_save_path: Optional path to save progress CSV
        auto_save_interval: Save every N rows
    
    Returns:
        DataFrame with added translation columns
    """
    openai.api_key = api_key
    model_config = get_model_config()
    
    # Override model if provided
    if model:
        model_config["model"] = model
        
    # Apply request-specific config overrides (this ensures session isolation)
    if request_config:
        model_config.update(request_config)
        
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
    
    import concurrent.futures
    
    # Helper for single row processing
    def process_row(index, text, targets_json, prmt, cfg, apikey, c_event):
        # Stop immediately if cancelled
        if c_event and c_event.is_set():
            return index, None
            
        if pd.isna(text) or str(text).strip() == "":
            return index, None

        # Rate limit jitter or small sleep to prevent burst usage
        # With parallel requests, we might hit rate limits faster. But 429s should be handled by retries ideally.
        # For simple logic, we rely on the executor not launching too many at once.
        
        max_retries = cfg.get("max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                # Construct system prompt for JSON
                system_prompt = (
                    f"You are a professional translator. Translate the text into the following languages: {targets_json}.\n"
                    "Return the result as a valid JSON object where keys are the 2-letter language codes and values are the translations.\n"
                    "Example output: {\"es\": \"Hola\", \"fr\": \"Bonjour\"}\n"
                    "Do NOT add any other text."
                )
                
                full_prompt = f"Text to translate:\n{text}"
                
                # Add user instructions
                style_instructions = prmt.replace("{lang}", "the target languages")
                if len(style_instructions) > 10:
                     full_prompt = f"Instructions:\n{style_instructions}\n\n{full_prompt}"
    
                client = openai.OpenAI(api_key=apikey)
                
                kwargs = {
                    "model": cfg["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ]
                }
                
                if "gpt-4" in cfg["model"] or "gpt-5" in cfg["model"] or "gpt-3.5-turbo-1106" in cfg["model"] or "gpt-3.5-turbo-0125" in cfg["model"]:
                    kwargs["response_format"] = {"type": "json_object"}
    
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content.strip()
                
                import json
                translations_map = json.loads(content)
                return index, translations_map
                
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1 + 1 # 2, 3, 5 seconds...
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded for row {index} after {max_retries} retries.")
                    return index, {}
            except Exception as e:
                logger.error(f"Error translating row {index}: {e}")
                return index, {} # Return empty on failure
        
        return index, {}
            
    # --- Parallel Execution ---
    max_workers = model_config.get("max_parallel_requests", 5)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Submit tasks
        for i, text in enumerate(texts):
            # Check cancel before submitting too many (not perfect but helps)
            if cancel_event and cancel_event.is_set():
                break
            
            # Skip empty early to save threads, or let worker handle it (worker handles it for consistency)
            future = executor.submit(process_row, i, text, targets_json_str, prompt, model_config, api_key, cancel_event)
            futures.append(future)

        # Process results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            # Check cancel
            if cancel_event and cancel_event.is_set():
                logger.info("Translation cancelled by user")
                break # Stop processing results, context manager will wait for pending but we stop UI updates
                
            try:
                idx, result_map = future.result()
                
                if result_map:
                    for lang in languages:
                        translation = result_map.get(lang, "")
                        out_col = f"{column}_Translated_{lang}"
                        df_result.at[idx, out_col] = translation
            except Exception as e:
                logger.error(f"Worker exception: {e}")
                
            completed_count += 1
            if progress_callback:
                progress_callback(f"Translated row {completed_count}/{total_items}...", completed_count / total_items)
                
            # Auto-save
            if auto_save_path and completed_count % auto_save_interval == 0:
                try:
                    df_result.to_csv(auto_save_path, index=False)
                    # logger.info(f"Auto-saved progress to {auto_save_path}") 
                except Exception as e:
                    logger.warning(f"Auto-save failed: {e}")

    if cancel_event and cancel_event.is_set():
        if progress_callback:
             progress_callback("Cancelled.", 1.0)
    elif progress_callback:
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
