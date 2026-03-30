import os
import time
import requests
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import trafilatura
from duckduckgo_search import DDGS

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import nltk
import spacy
import wikipedia

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ----------------------------------------------
# INITIAL SETUP
# ----------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

nltk.download("punkt")

print("Loading AI models...")

model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

print("Models loaded!")


# ----------------------------------------------
# SEARCH FUNCTIONS
# ----------------------------------------------

def search_duckduckgo(query):

    results = []

    try:
        with DDGS() as ddgs:

            data = ddgs.text(query, max_results=4)

            for r in data:
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                    "source": "DuckDuckGo"
                })

    except Exception as e:
        logging.warning(f"DuckDuckGo search error: {e}")

    return results


def search_wikipedia(query):

    try:

        summary = wikipedia.summary(query, sentences=2)

        return [{
            "title": "Wikipedia",
            "url": "https://wikipedia.org",
            "snippet": summary,
            "source": "Wikipedia"
        }]

    except Exception:
        return []


# ----------------------------------------------
# WEB SCRAPER
# ----------------------------------------------

def scrape_content(url):

    try:

        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, headers=headers, timeout=10)

        text = trafilatura.extract(res.text)

        return text[:1500] if text else None

    except Exception as e:

        logging.warning(f"Scraping failed: {e}")

        return None


# ----------------------------------------------
# CHAT LINK EXTRACTION
# ----------------------------------------------

def extract_chat_from_link(url):

    try:

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        driver.get(url)

        wait = WebDriverWait(driver, 20)

        wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        time.sleep(4)

        texts = []

        # METHOD 1: ChatGPT style
        try:

            messages = driver.find_elements(
                By.CSS_SELECTOR,
                "[data-message-author-role]"
            )

            for m in messages:
                t = m.text.strip()
                if len(t) > 40:
                    texts.append(t)

        except:
            pass

        # METHOD 2: generic markdown blocks
        if not texts:

            elements = driver.find_elements(
                By.CSS_SELECTOR,
                "div.markdown, article, p"
            )

            for e in elements:
                t = e.text.strip()
                if len(t) > 40:
                    texts.append(t)

        # METHOD 3: fallback full page
        if not texts:

            body = driver.find_element(By.TAG_NAME, "body").text

            if body:
                texts.append(body[:4000])

        driver.quit()

        if texts:
            return " ".join(texts)

        return None

    except Exception as e:

        logging.error(f"Selenium extraction error: {e}")

        return None


# ----------------------------------------------
# CLAIM EXTRACTION
# ----------------------------------------------

def extract_claims(text):

    doc = nlp(text)

    claims = []

    for sent in doc.sents:

        s = sent.text.strip()

        if len(s.split()) < 6:
            continue

        if any(token.pos_ == "VERB" for token in nlp(s)):
            claims.append(s)

    return claims[:10]


# ----------------------------------------------
# EVIDENCE COLLECTION
# ----------------------------------------------

def get_evidence(claim):

    evidence = []

    evidence.extend(search_duckduckgo(claim))
    evidence.extend(search_wikipedia(claim))

    results = []

    for r in evidence[:5]:

        content = scrape_content(r["url"]) if r.get("url") else None

        results.append({
            "title": r["title"],
            "url": r.get("url"),
            "snippet": r["snippet"],
            "content": content,
            "source": r.get("source")
        })

    return results


# ----------------------------------------------
# FACT CHECK ENGINE
# ----------------------------------------------

def verify_claim(claim):

    evidence = get_evidence(claim)

    if not evidence:

        return {
            "claim": claim,
            "status": "no_evidence",
            "trust_score": 0,
            "hallucination_score": 100
        }

    claim_emb = model.encode([claim])

    best_score = 0
    best = None

    for e in evidence:

        text = f"{e['snippet']} {e['content']}"

        if not text:
            continue

        emb = model.encode([text])

        score = cosine_similarity(claim_emb, emb)[0][0]

        if score > best_score:
            best_score = score
            best = e

    trust = round(best_score * 100, 1)
    hallucination = round((1 - best_score) * 100, 1)

    if best_score >= 0.65:
        status = "trusted"
    elif best_score >= 0.45:
        status = "uncertain"
    else:
        status = "hallucination"

    return {

        "claim": claim,
        "status": status,
        "trust_score": trust,
        "hallucination_score": hallucination,
        "source_title": best["title"] if best else None,
        "source_url": best["url"] if best else None,
        "correct_answer": best["snippet"] if best else None
    }


# ----------------------------------------------
# ANALYSIS PIPELINE
# ----------------------------------------------

def analyze_text(text):

    start = time.time()

    claims = extract_claims(text)

    results = []

    for c in claims:
        results.append(verify_claim(c))

    total = len(results)

    hallucinated = sum(
        1 for r in results if r["status"] == "hallucination"
    )

    rate = (hallucinated / total) * 100 if total else 0

    return {

        "total_claims": total,
        "hallucination_rate": round(rate,1),
        "results": results,
        "processing_time": round(time.time()-start,2)
    }


# ----------------------------------------------
# API
# ----------------------------------------------

@app.route("/")
def index():

    return send_from_directory("../frontend","index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json(force=True)

        text = data.get("text")
        url = data.get("url")

        if not text and not url:

            return jsonify({
                "success": False,
                "error": "No input provided"
            })

        # URL processing
        if url:

            logging.info("Extracting text from link...")

            text = extract_chat_from_link(url)

            if not text:

                logging.info("Trying fallback HTTP extraction")

                try:
                    res = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})
                    text = trafilatura.extract(res.text)
                except:
                    return jsonify({
                        "success": False,
                        "error": "Failed to extract chat"
                    })

        analysis = analyze_text(text)

        return jsonify({

            "success": True,

            "analysis": {

                "total_claims": analysis["total_claims"],
                "hallucination_rate": analysis["hallucination_rate"],
                "trust_rate": 100 - analysis["hallucination_rate"],
                "processing_time": analysis["processing_time"],
                "results": analysis["results"]
            }

        })

    except Exception as e:

        logging.error(e)

        return jsonify({
            "success": False,
            "error": str(e)
        })


# ----------------------------------------------

if __name__ == "__main__":

    print("🚀 VerifAI Server Running")
    print("http://localhost:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)