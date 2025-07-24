from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import json

# Modeli yükle
model_name = "Helsinki-NLP/opus-mt-tr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Giriş dosyasını yükle
with open("data/entries.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

translated_entries = []

# Entry'leri çevir
for text in tqdm(entries, desc="Çeviriliyor"):
    if not isinstance(text, str) or not text.strip():
        continue
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", max_length=512, truncation=True)
    translated = model.generate(**batch)
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated_entries.append({"original": text, "translated": english_text})

# Sonuçları kaydet
with open("data/translated_entries.json", "w", encoding="utf-8") as f:
    json.dump(translated_entries, f, ensure_ascii=False, indent=2)
