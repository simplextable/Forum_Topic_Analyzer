from transformers import pipeline
import pandas as pd

# 1. Import the File
df = pd.read_json("../data/translated_entries.json")

# 2. Take only 'translated' Column
texts = df["translated"].tolist()

# 3. Import the Models
default_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
finetuned_model = pipeline("sentiment-analysis", model="finetuned_model")

# 4. Get the Predictions
default_preds = default_model(texts)
finetuned_preds = finetuned_model(texts)

# 5. Add the Results
df["default_label"] = [pred["label"] for pred in default_preds]
df["default_score"] = [round(pred["score"], 4) for pred in default_preds]

df["finetuned_label"] = [pred["label"] for pred in finetuned_preds]
df["finetuned_score"] = [round(pred["score"], 4) for pred in finetuned_preds]


# 6. Give Information: How much Positive Predicted Each Model?
default_positive_ratio = (df["default_label"] == "POSITIVE").mean()
finetuned_positive_ratio = (df["finetuned_label"] == "POSITIVE").mean()

# 7. Print
print(f"Default model POSITIVE oranı: %{default_positive_ratio * 100:.2f}")
print(f"Fine-tuned model POSITIVE oranı: %{finetuned_positive_ratio * 100:.2f}")

# 8. Give Result
if finetuned_positive_ratio < default_positive_ratio:
    print("✅ Seems Finetuning has a Impact")
else:
    print("⚠️ Fine Tuning's Impact cannot be Observed")

df[["translated", 
    "default_label", "default_score", 
    "finetuned_label", "finetuned_score"
]].to_csv("data/sentiment_comparison_results.csv", index=False, encoding="utf-8-sig")
print("📁 Results are saved as 'sentiment_comparison_results.csv'")


