"""
nsclc_local.py  •  Version 1.0  (local‑only CLI)
────────────────────────────────────────────────────
Run:
    pip install "openai>=1.10" pandas requests
    python nsclc_local.py
"""

import os, uuid, shelve, logging, csv, textwrap
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# ────────── CONFIG ──────────
API_KEY      = ""   # <─ Perplexity key
MODEL        = "sonar-pro"
GLOSSARY_CSV = "final_glossary.csv"   # <─ path to glossary CSV  (feature,definition)

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

# ────────── GLOSSARY ──────────
def load_glossary(path: str) -> Dict[str, str]:
    gloss: Dict[str, str] = {}
    if not os.path.isfile(path):
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

# ────────── PROMPT BUILDER ──────────
def construct_prompt(f: FeatureInput) -> str:
    f_all = f.genes + f.radiological + f.pathological + f.clinical
    feature_str = ", ".join(f_all)

    matched_defs = [f"* **{feat}** — {GLOSSARY[feat]}" for feat in f_all if feat in GLOSSARY]
    glossary_block = ("#### Glossary (auto‑extracted)\n" +
                      "\n".join(matched_defs) + "\n\n") if matched_defs else ""

    original_prompt = textwrap.dedent("""\
        You are an expert biomedical researcher specializing in non‑small‑cell lung cancer (NSCLC)
        and in‑depth analysis of scientific literature.

        You are provided with the following features for deep analysis:
    """)

    body = textwrap.dedent("""\
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
    """)

    return original_prompt + feature_str + "\n\n" + glossary_block + body

# ────────── PERPLEXITY CALL ──────────
def ask_perplexity(messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model       = MODEL,
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 800
    )
    return resp.choices[0].message.content

# ────────── STORAGE HELPERS ──────────
def store_explanation(eid: str, text: str):
    with shelve.open("explanations.db") as db:
        db[eid] = {"text": text, "rating": 0, "comment": None}

def update_feedback(eid: str, rating: int, comment: str | None):
    with shelve.open("ratings.db") as rdb:
        rdb[eid] = {"rating": rating, "comment": comment}
    with shelve.open("explanations.db") as edb:
        rec = edb.get(eid)
        if rec:
            rec["rating"] = rating
            rec["comment"] = comment
            edb[eid] = rec

# ────────── CLI HELPERS ──────────
def prompt_list(label: str) -> list[str]:
    raw = input(f"{label} (comma‑separated, blank=none): ").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else []

def explain_local(feature_dict: dict) -> tuple[str, str]:
    try:
        features = FeatureInput(**feature_dict)
    except ValidationError as e:
        print("Feature input validation error:", e)
        raise SystemExit

    prompt = construct_prompt(features)
    messages = [
        {"role": "system", "content": "Be precise and relatively concise."},
        {"role": "user",   "content": prompt}
    ]
    explanation = ask_perplexity(messages)
    eid = str(uuid.uuid4())
    store_explanation(eid, explanation)

    print("\n=== EXPLANATION (ID:", eid, ") ===\n")
    print(explanation)
    print("\n==============================\n")
    return eid, explanation

def ask_local(question: str, eid: Optional[str] = None):
    history = [{"role": "system", "content": "Answer concisely with citations."}]
    if eid:
        with shelve.open("explanations.db") as db:
            prev = db.get(eid, {}).get("text")
        if prev:
            history.append({"role": "system",
                            "content": f"Previous explanation for context:\n\n{prev}"})
    history.append({"role": "user", "content": question})
    answer = ask_perplexity(history)

    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n==============\n")

# ────────── MAIN CLI ──────────

if __name__ == "__main__":
    print("╭────────────────────────────────────────────╮")
    print("│ NSCLC Explainability • Local CLI          │")
    print("╰────────────────────────────────────────────╯\n")

    genes        = prompt_list("Genes")
    radiological = prompt_list("Radiological features")
    pathological = prompt_list("Pathological features")
    clinical     = prompt_list("Clinical features")

    if not any([genes, radiological, pathological, clinical]):
        print("No features entered. Exiting.")
        raise SystemExit

    eid, _ = explain_local({
        "genes": genes,
        "radiological": radiological,
        "pathological": pathological,
        "clinical": clinical
    })

    # Rating
    while True:
        r_raw = input("Rate the explanation 1‑5 (0 to skip): ").strip() or "0"
        if r_raw.isdigit() and 0 <= int(r_raw) <= 5:
            rating = int(r_raw); break
        print("Please enter a number 0‑5.")
    comment = None
    if rating:
        comment = input("Optional comment (blank to skip): ").strip() or None
        update_feedback(eid, rating, comment)
        print("✓ Feedback recorded\n")

    # Follow‑up loop
    while True:
        q = input("Follow‑up question (blank to quit): ").strip()
        if not q:
            print("Bye!")
            break
        ask_local(q, eid)
