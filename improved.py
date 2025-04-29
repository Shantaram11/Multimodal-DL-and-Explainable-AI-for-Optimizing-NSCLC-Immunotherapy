import os, uuid, shelve, csv, logging, textwrap
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# ── CONFIG ─────────────────────────────────────────────────────────────
API_KEY      = "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"   #  <-- Perplexity API key ("pplx-...")
MODEL        = "sonar-pro"
GLOSSARY_CSV = "final_glossary.csv"   #  <-- path to glossary CSV (feature,definition)

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

# ── GLOSSARY ───────────────────────────────────────────────────────────
def load_glossary(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        logging.warning("Glossary CSV not found: %s", path)
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row[0].strip(): row[1].strip() for row in csv.reader(fh) if len(row) >= 2}

GLOSSARY = load_glossary(GLOSSARY_CSV)

# ── DATA MODEL ─────────────────────────────────────────────────────────
class FeatureInput(BaseModel):
    genes:        List[str] = []
    radiological: List[str] = []
    pathological: List[str] = []
    clinical:     List[str] = []

# ── PROMPT BUILDER (citations enforced) ────────────────────────────────
def construct_prompt(f: FeatureInput) -> str:
    all_feats = f.genes + f.radiological + f.pathological + f.clinical
    feature_str = ", ".join(all_feats)
    defs = [f"* **{k}** — {GLOSSARY[k]}" for k in all_feats if k in GLOSSARY]
    glossary_block = ("#### Glossary\n" + "\n".join(defs) + "\n\n") if defs else ""

    prompt = textwrap.dedent(f"""\
        You are an expert biomedical researcher specializing in non‑small‑cell lung cancer (NSCLC).

        The SHAP‑ranked input features are:
        {feature_str}

        {glossary_block}These features come from a machine‑learning predictor of:
        • Best RECIST response • Overall survival • Progression‑free survival

        TASK: Synthesize the most recent, peer‑reviewed literature (PubMed, NEJM, Lancet Oncology,
        Nature Medicine, TCGA, ClinicalTrials.gov). Provide:

        1. Executive summary
        2. Feature‑by‑feature analysis (role, mechanism, evidence tier)
        3. Cross‑feature interactions
        4. Knowledge gaps

        **Every factual claim MUST have an inline citation** in one of these formats:
        [PubMed ID: 12345678]   •OR•   [DOI: 10.xxxx/xxxxx]

        Use bullet points and sub‑headings. Avoid jargon.

        Return output in Markdown. Do not fabricate citations.
    """)
    return prompt

# ── LLM CALL ───────────────────────────────────────────────────────────
def llm(messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model       = MODEL,
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 800
    )
    return resp.choices[0].message.content

# ── STORAGE ────────────────────────────────────────────────────────────
def store_explanation(eid: str, text: str):
    with shelve.open("explanations.db") as db:
        db[eid] = {"text": text, "rating": 0, "comment": None}

def update_feedback(eid: str, rating: int, comment: str | None):
    with shelve.open("ratings.db") as r:
        r[eid] = {"rating": rating, "comment": comment}
    with shelve.open("explanations.db") as e:
        rec = e.get(eid)
        if rec:
            rec["rating"], rec["comment"] = rating, comment
            e[eid] = rec

# ── PUBLIC HELPERS ─────────────────────────────────────────────────────
def explain_local(features: dict) -> tuple[str, str]:
    features_obj = FeatureInput(**features)
    prompt = construct_prompt(features_obj)
    text   = llm([{"role":"system","content":"Be clear and concise."},
                  {"role":"user","content":prompt}])
    eid = str(uuid.uuid4())
    store_explanation(eid, text)
    return eid, text

def ask_local(question: str, explanation_id: Optional[str] = None) -> str:
    history = [{"role":"system","content":"Answer concisely; cite PubMed/DOI inline."}]
    if explanation_id:
        with shelve.open("explanations.db") as db:
            prev = db.get(explanation_id, {}).get("text")
        if prev:
            history.append({"role":"system","content":f"Context:\n\n{prev}"})
    history.append({"role":"user","content":question})
    return llm(history)

def feedback_local(eid: str, rating: int, comment: str | None = None):
    update_feedback(eid, rating, comment)

# ── CLI UI ─────────────────────────────────────────────────────────────
def prompt_list(label):
    raw = input(f"{label} (comma‑sep): ").strip()
    return [s.strip() for s in raw.split(",") if s.strip()]

if __name__ == "__main__":
    print("=== NSCLC Explainability – CLI ===")
    genes = prompt_list("Genes")
    radi = prompt_list("Radiological")
    path = prompt_list("Pathological")
    clin = prompt_list("Clinical")

    if not any([genes, radi, path, clin]):
        print("No features provided. Exiting.")
        raise SystemExit

    eid, explanation = explain_local({
        "genes": genes, "radiological": radi,
        "pathological": path, "clinical": clin
    })
    print("\n--- EXPLANATION ---\n", explanation, "\n-------------------\n")

    # rating
    r = int(input("Rate this 1‑5 (0 skip): ").strip() or "0")
    c = None
    if r: c = input("Comment (opt): ").strip() or None
    if r: feedback_local(eid, r, c)

    while True:
        q = input("Follow‑up (blank to quit): ").strip()
        if not q: break
        ans = ask_local(q, eid)
        print("\n--- ANSWER ---\n", ans, "\n--------------\n")