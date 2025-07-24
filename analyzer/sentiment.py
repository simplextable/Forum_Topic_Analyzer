import json
import pandas as pd
from transformers import pipeline

# 📥 Load translated entries
with open("data/translated_entries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract English text
entries = [item["translated"] for item in data if "translated" in item]

df = pd.DataFrame(entries, columns=["entry"])

# 🤗 Load English Sentiment Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # binary: POSITIVE / NEGATIVE
sentiment_pipe = pipeline("sentiment-analysis", model=model_name)

# ⏳ Apply Sentiment Analysis
df["sentiment"] = df["entry"].apply(lambda x: sentiment_pipe(x[:512])[0]['label'])

# 📊 Save Results
df.to_csv("data/sentiment_results_en.csv", index=False, encoding="utf-8-sig")
print(df[["entry", "sentiment"]].head())
