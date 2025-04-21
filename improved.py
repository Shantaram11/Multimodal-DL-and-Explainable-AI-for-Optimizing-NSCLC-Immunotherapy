"""
Version_2.1_(2025-04-20)
  • Fixes FeedbackInput 
  • Stores explanations as dict → prevents KeyErrors later
  • Cleans duplicate import, adds optional root route
Run:
    pip install fastapi uvicorn openai pandas requests
    uvicorn nsclc_glossary:app --reload
"""

import os, uuid, shelve, logging, csv, textwrap
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ────────── CONFIG ──────────
API_KEY       = "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"  #API key goes here
MODEL         = "sonar-pro" #adjust as needed to your favorite model, im using sonar pro
GLOSSARY_CSV  = "final_glossary.csv" #set the glossary path here

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")
app    = FastAPI(title="NSCLC Explainability_API_Module")

# ────────── LOAD GLOSSARY ──────────
def load_glossary(path: str) -> Dict[str, str]:
    gloss = {}
    if not os.path.exists(path):
        logging.warning("Glossary CSV not found: %s", path)
        return gloss
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                gloss[row[0].strip()] = row[1].strip()
    logging.info("Loaded %d glossary entries", len(gloss))
    return gloss

GLOSSARY = load_glossary(GLOSSARY_CSV)

# ────────── DATA MODELS ──────────
class FeatureInput(BaseModel):
    genes:        List[str] = []
    radiological: List[str] = []
    pathological: List[str] = []
    clinical:     List[str] = []

class FeedbackInput(BaseModel):
    explanation_id: str
    rating: int
    comment: Optional[str] = None          # ← new (prevents AttributeError)

class QueryInput(BaseModel):
    question: str
    explanation_id: Optional[str] = None

# ────────── PROMPT BUILDER (unchanged) ──────────
def construct_prompt(features: FeatureInput) -> str:
    f_all = (features.genes + features.radiological +
             features.pathological + features.clinical)
    feature_str = ", ".join(f_all)

    matched_defs = [f"* **{f}** — {GLOSSARY[f]}" for f in f_all if f in GLOSSARY]
    glossary_block = ("#### Glossary (auto‑extracted)\n" +
                      "\n".join(matched_defs) + "\n\n") if matched_defs else ""

    original_prompt = textwrap.dedent("""\
        You are an expert biomedical researcher specializing in non-small cell lung cancer (NSCLC) and in-depth analysis of scientific literature.

        You are provided with the following features for deep analysis:
    """)
    #I put the description of predicted features in here. Not sure if that is the best. Could be useful in glossary or maybe can be defined globally. 
    return (original_prompt + feature_str + "\n\n" + glossary_block +
            textwrap.dedent("""\
               These features represent the most predictive SHAP values identified by a machine learning model trained to predict patient outcomes following immunotherapy for NSCLC.
            
            
            The model predicts the following outcomes:
            - Best response: Best observed RECIST [1] response (i.e., progression, stable disease, partial
              response, or complete response) during the patient's follow-up after first-line immunotherapy
              initiation.
            - OS:  Vital status: Duration from the initiation of first-line immunotherapy (with or without
              chemotherapy) to the patient's death or last available status update. 
            - PFS:  Progression: Duration from the initiation of first-line immunotherapy to the occurrence
              of the first progression event or last available status update, including the emergence of new
              lesions or the progression of pre-existing ones.
                          
            Please search and synthesize the most recent, high-quality, peer-reviewed literature — from sources such as PubMed, NEJM, Lancet Oncology, ClinicalTrials.gov, Nature Medicine, and The Cancer Genome Atlas (TCGA) — to provide a detailed explanation of how these features contribute to NSCLC outcomes following immunotherapy.

            Your response should include:
            - A summary of key findings
            - A detailed explanation of relevant biological pathways
            - For each feature:
              - Its prognostic or predictive role (e.g., risk or protective factor)
              - Its mechanism of action, when known
              - The level of evidence (e.g., preclinical, retrospective, prospective, RCT, meta-analysis)
            - Inline citations using the format: [PubMed ID: XXXXX] or [DOI: XXXXX]
            - Prioritize studies from 2020–2024, but include landmark earlier findings when appropriate

            Present your findings clearly and relatively concisely, using accessible language suitable for a mixed audience of clinicians and patients. Avoid overwhelming technical jargon. Use headings and bullet points for readability.
        """))

# ────────── CHAT WRAPPER ──────────
def ask_perplexity(messages):
    try:
        resp = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            temperature = 0.2,
            max_tokens  = 800     # cost guard
        )
        return resp.choices[0].message.content, resp.model_dump()
    except Exception as e:
        logging.exception("Perplexity API error")
        raise HTTPException(503, detail=str(e))

# ────────── ENDPOINTS ──────────
@app.post("/predict_explain")
async def predict_explain(features: FeatureInput):
    explanation_id = str(uuid.uuid4())
    prompt         = construct_prompt(features)
    messages = [
        {"role": "system", "content": "Be precise and relatively concise."},
        {"role": "user",   "content": prompt}
    ]
    explanation, raw = ask_perplexity(messages)

    # store as dict so we can later add rating/comment
    with shelve.open("explanations.db") as db:
        db[explanation_id] = {
            "text":    explanation,
            "rating":  0,
            "comment": None
        }

    return {"explanation_id": explanation_id,
            "explanation":    explanation,
            "details":        raw}

@app.post("/feedback")
async def feedback(fb: FeedbackInput):
    # legacy ratings.db (optional)
    with shelve.open("ratings.db") as rdb:
        rdb[fb.explanation_id] = {"rating": fb.rating, "comment": fb.comment}

    # update main record
    with shelve.open("explanations.db") as edb:
        rec = edb.get(fb.explanation_id)
        if rec:
            rec["rating"]  = fb.rating
            rec["comment"] = fb.comment
            edb[fb.explanation_id] = rec
    return {"status": "recorded"}

@app.post("/ask") #the initial responce can be referenced or a question can be asked separately.
async def ask(query: QueryInput):
    history = [
        {"role": "system",
         "content": "Answer concisely with citations."}
    ]

    if query.explanation_id:
        with shelve.open("explanations.db") as db:
            prev = db.get(query.explanation_id, {}).get("text")
        if prev:                              # add as SECOND system msg
            history.append({
                "role": "system",
                "content": f"Previous explanation for context:\n\n{prev}"
            })

    # now the first non‑system message is from the user → OK
    history.append({"role": "user", "content": query.question})

    print("DEBUG:", history)                  # optional: verify order
    answer, raw = ask_perplexity(history)
    return {"answer": answer, "details": raw}


# Optional root route for quick “alive” check
@app.get("/")
def root():
    return {"message": "NSCLC Explainability API – see /docs for Swagger UI"}