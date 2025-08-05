import gradio as gr
import json
from scraper.eksi_scraper import get_eksi_entries
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoTokenizer
import pandas as pd

# Translation model
tr_en_model_name = "Helsinki-NLP/opus-mt-tr-en"
tr_en_tokenizer = MarianTokenizer.from_pretrained(tr_en_model_name)
tr_en_model = MarianMTModel.from_pretrained(tr_en_model_name)

# Sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")



def scrape_entries(url):
    if not url.startswith("http"):
        return "❌ Invalid URL", None

    try:
        entries = get_eksi_entries(url, max_entry=250)
        with open("data/entries.json", "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return "✅ Scraping successful!", "data/entries.json"
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def translate_entries():
    try:
        with open("data/entries.json", "r", encoding="utf-8") as f:
            entries = json.load(f)

        translated = []
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                continue
            batch = tr_en_tokenizer.prepare_seq2seq_batch([entry], return_tensors="pt", truncation=True)
            gen = tr_en_model.generate(**batch)
            translated_text = tr_en_tokenizer.decode(gen[0], skip_special_tokens=True)
            translated.append({"original": entry, "translated": translated_text})

        with open("data/translated_entries.json", "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        return "✅ Translation complete!", "data/translated_entries.json"

    except Exception as e:
        return f"❌ Error during translation: {str(e)}", None

def analyze_sentiment():
    try:
        with open("data/translated_entries.json", "r", encoding="utf-8") as f:
            entries = json.load(f)

        results = []
        for item in entries:
            text = item.get("translated", "")
            if not text.strip():
                continue
            prediction = sentiment_pipeline(text[:512])[0]  # truncate long ones
            results.append({
                "entry": text,
                "label": prediction["label"],
                "score": round(prediction["score"], 4)
            })

        df = pd.DataFrame(results)
        df.to_csv("data/sentiment_results_en.csv", index=False)

        return "✅ Sentiment analysis completed!", "data/sentiment_results_en.csv"

    except Exception as e:
        return f"❌ Error during sentiment analysis: {str(e)}", None

def summarize_entries():
    try:
        with open("data/translated_entries.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item["translated"] for item in data if "translated" in item]

        # Token bazlı grupla
        MAX_TOKENS = 850
        groups, group, tokens = [], [], 0
        for text in texts:
            token_count = len(summarizer_tokenizer.encode(text, truncation=False))
            if tokens + token_count > MAX_TOKENS:
                groups.append(" ".join(group))
                group = [text]
                tokens = token_count
            else:
                group.append(text)
                tokens += token_count
        if group:
            groups.append(" ".join(group))

        summaries = []
        for chunk in groups:
            summary = summarizer(chunk, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        # Özetlerin özeti
        final_input = " ".join(summaries)
        final_summary = summarizer(final_input, max_length=200, min_length=80, do_sample=False)[0]['summary_text']

        with open("data/final_summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)

        return final_summary, "data/final_summary.txt"

    except Exception as e:
        return f"❌ Error: {str(e)}", None


with gr.Blocks(title="Forum Topic Analyzer") as demo:
    gr.Markdown("# ✨ Forum Topic Analyzer")
    gr.Markdown("Enter a topic URL from Ekşi Sözlük and scrape the entries.")

    with gr.Row():
        url_input = gr.Textbox(label="Topic URL", placeholder="https://eksisozluk.com/...", scale=4)
        scrape_button = gr.Button("Scrape Entries")

    scrape_status = gr.Textbox(label="Scrape Status", interactive=False)
    entries_file = gr.File(label="Download entries.json")

    scrape_button.click(fn=scrape_entries, inputs=url_input, outputs=[scrape_status, entries_file])

    with gr.Row():
        translate_button = gr.Button("Translate Entries")
    translate_status = gr.Textbox(label="Translation Status", interactive=False)
    translated_file = gr.File(label="Download translated_entries.json")

    translate_button.click(fn=translate_entries, outputs=[translate_status, translated_file])

    with gr.Row():
        sentiment_button = gr.Button("Analyze Sentiment")
    sentiment_status = gr.Textbox(label="Sentiment Status", interactive=False)
    sentiment_file = gr.File(label="Download sentiment_results_en.csv")

    sentiment_button.click(fn=analyze_sentiment, outputs=[sentiment_status, sentiment_file])

    with gr.Row():
        summarize_button = gr.Button("Summarize Entries")
    summary_text = gr.Textbox(label="Final Summary", lines=10)
    summary_file = gr.File(label="Download final_summary.txt")

    summarize_button.click(fn=summarize_entries, outputs=[summary_text, summary_file])


demo.launch(server_name="0.0.0.0", server_port=7860)

