import argparse
import pandas as pd
import openai
import time
import os
import sys

# Ensure UTF-8 encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- Argümanları al ---
parser = argparse.ArgumentParser(description="Genel Amaçlı AI CSV Çevirici")
parser.add_argument('--input', required=True, help='Giriş CSV dosyası')
parser.add_argument('--output', required=True, help='Çıkış CSV dosyası')
parser.add_argument('--column', required=True, help='Çevrilecek kolon adı')
parser.add_argument('--languages', required=True, help='Hedef diller (virgül ile ayrılmış ISO kodları: tr,de,fr)')
parser.add_argument('--prompt', required=True, help='Kullanıcı tanımlı prompt, {lang} yerine dil kodu gelecek')
parser.add_argument('--api_key', required=True, help='OpenAI API anahtarı')
args = parser.parse_args()

# --- Check input file ---
if not os.path.isfile(args.input):
    print(f"Error: Input file '{args.input}' does not exist.")
    sys.exit(1)

# --- Ayarlar ---
openai.api_key = args.api_key
languages = [lang.strip() for lang in args.languages.split(',')]

df = pd.read_csv(args.input, encoding='utf-8')

# --- Çeviri ---
for lang in languages:
    lang_name = lang  # ISO kodu doğrudan kullanılıyor
    out_col = f"{args.column}_Translated_{lang}"
    df[out_col] = ""
    print(f"\nTranslating to {lang_name.upper()}...")

    for idx, row in df.iterrows():
        try:
            prompt_full = args.prompt.replace("{lang}", lang_name)
            full_prompt = f"{prompt_full}\n\nText:\n{row[args.column]}"

            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": full_prompt}]
            )

            translation = response.choices[0].message.content.strip()
            df.at[idx, out_col] = translation
            time.sleep(1.2)

        except Exception as e:
            print(f"Error at row {idx}, lang {lang_name}: {e}")
            continue

# --- Kaydet ---
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df.to_csv(args.output, index=False, encoding='utf-8')
print(f"\nTranslation complete. Output saved to {args.output}")
