from fastapi import FastAPI, Query
import subprocess

app = FastAPI(
    title="Forum Topic Analyzer API",
    description="Scraping + Translation + Sentiment + Summarization pipeline",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Forum Topic Analyzer API is running 🎯"}

@app.get("/scrape")
def scrape_entries(url: str = Query(..., description="Ekşi Sözlük başlık URL'si")):
    result = subprocess.run(["python", "scraper/eksi_scraper.py", url], capture_output=True, text=True)
    return {
        "status": "scraping completed",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

@app.get("/translate")
def translate():
    result = subprocess.run(["python", "analyzer/translate_entries.py"], capture_output=True, text=True)
    return {"status": "translation completed", "log": result.stdout}

@app.get("/sentiment")
def sentiment():
    result = subprocess.run(["python", "analyzer/sentiment.py"], capture_output=True, text=True)
    return {"status": "sentiment analysis completed", "log": result.stdout}

@app.get("/summarize")
def summarize():
    result = subprocess.run(["python", "analyzer/summarization.py"], capture_output=True, text=True)
    return {"status": "summarization completed", "log": result.stdout}
