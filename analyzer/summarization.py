import json
from transformers import pipeline, AutoTokenizer

print("🔍 translated_entries.json okunuyor...")

# 📥 Load translated entries
with open("data/translated_entries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["translated"] for item in data if "translated" in item and isinstance(item["translated"], str)]

# 🤖 Load model and tokenizer
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ⚖️ Token limit
MAX_TOKENS = 850  # Daha güvenli limit (modelin 1024 token sınırına yaklaşmadan)

# 🔧 Helper: Group texts by token count
def group_texts_by_token_limit(texts, tokenizer, max_tokens):
    groups = []
    current_group = []
    current_tokens = 0

    for text in texts:
        token_count = len(tokenizer.encode(text, truncation=False))
        if current_tokens + token_count > max_tokens:
            groups.append(" ".join(current_group))
            current_group = [text]
            current_tokens = token_count
        else:
            current_group.append(text)
            current_tokens += token_count

    if current_group:
        groups.append(" ".join(current_group))

    return groups

# 🔧 Helper: Summarize each group
def summarize_groups(groups, tag="pass"):
    summaries = []
    for i, group in enumerate(groups):
        print(f"🧠 Summarizing {tag} group {i+1}/{len(groups)}")
        summary = summarizer(group, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

# 🔄 Multi-stage summarization
def multi_stage_summarize(texts, tokenizer, max_tokens):
    stage = 1
    summaries = texts
    while True:
        groups = group_texts_by_token_limit(summaries, tokenizer, max_tokens)
        if len(groups) == 1:
            final_summary = summarize_groups(groups, tag=f"final")[0]
            return final_summary
        summaries = summarize_groups(groups, tag=f"stage {stage}")
        stage += 1

# 🚀 Run summarization
print(f"📊 Loaded {len(texts)} entries.")
final_summary = multi_stage_summarize(texts, tokenizer, MAX_TOKENS)

# 💾 Save output
with open("data/final_summary.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print("\n✅ Final summary saved to 'final_summary.txt'")
print("\n📌 Preview:\n")
print(final_summary)
