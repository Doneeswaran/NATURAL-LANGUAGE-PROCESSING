import os
import uvicorn
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="NLP Evaluator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models. This may take a moment (especially the first time downloading mpnet)...")
# all-mpnet-base-v2 is the highest performing pre-trained sentence transformer for semantic similarity
# We force device='cpu' because the Apple Silicon (MPS) PyTorch backend can sometimes deadlock on encoding
model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

# Load spaCy for advanced Natural Language Processing
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

print("Models loaded successfully!")

class EvaluationRequest(BaseModel):
    questionText: str = ""
    referenceText: str
    studentText: str
    mode: str = "balanced"

class EvaluationResponse(BaseModel):
    scores: Dict[str, Any]
    overall: int
    strengths: str
    gaps: str
    suggestion: str
    keywords_matched: List[str]
    keywords_missed: List[str]

def extract_keywords(text: str):
    # Advanced NLP extraction using spaCy
    # We extract noun phrases and named entities to understand core concepts
    doc = nlp(text)
    
    concepts = []
    # Extract noun chunks (e.g. "mitochondria", "the process of photosynthesis")
    for chunk in doc.noun_chunks:
        # Ignore pronouns and very short chunks
        if chunk.root.pos_ != "PRON" and len(chunk.text) > 2:
            concepts.append(chunk.text.lower())
            
    # Extract named entities (e.g. "Albert Einstein", "Paris", "World War II")
    for ent in doc.ents:
        concepts.append(ent.text.lower())
        
    return list(set(concepts))

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_answer(req: EvaluationRequest):
    # Calculate Semantic Similarity using high-dimensional embeddings
    emb_ref = model.encode(req.referenceText, convert_to_tensor=True)
    emb_stu = model.encode(req.studentText, convert_to_tensor=True)
    
    # Cosine similarity yields a value between -1.0 and 1.0
    cosine_score = util.cos_sim(emb_ref, emb_stu).item()
    
    # Mathematical curve to map cosine similarity to a realistic 0-100 score.
    # mpnet tends to score between 0.3 and 1.0 for valid texts.
    semantic_score: int = 0
    if cosine_score < 0.2:
        semantic_score = 0
    else:
        # We use a non-linear scaling to make it fairer
        scaled = (cosine_score - 0.2) / 0.8
        semantic_score = int(min(100, (scaled ** 0.8) * 100))
    
    # Factual & Completeness heuristics using spaCy concept matching
    ref_concepts = extract_keywords(req.referenceText)
    stu_concepts = extract_keywords(req.studentText)
    
    matched: List[str] = []
    for rc in ref_concepts:
        # Check if the reference concept exists in any student concept (or vice versa)
        if any(rc in sc or sc in rc for sc in stu_concepts):
            matched.append(rc)
            
    missed: List[str] = [rc for rc in ref_concepts if rc not in matched]
    
    completeness_score = int(len(matched) / max(1, len(ref_concepts)) * 100)
    
    # Mode modifiers
    if req.mode == "strict":
        overall = int(semantic_score * 0.4 + completeness_score * 0.6)
    elif req.mode == "lenient":
        overall = int(semantic_score * 0.7 + completeness_score * 0.3)
    else: # balanced
        overall = int(semantic_score * 0.5 + completeness_score * 0.5)
        
    overall = min(100, max(0, overall))
    
    # Heuristics for Clarity and Relevance based on answer length & semantic score
    stu_len = len(req.studentText.split())
    
    clarity_score = 90 if stu_len > 10 else 50
    # A highly semantic answer that is concise is considered highly relevant
    relevance_score = min(100, int(semantic_score * 1.1)) if stu_len < len(req.referenceText.split()) * 2 else semantic_score
    
    # Generate text feedback
    strengths = "Good attempt." if overall < 50 else "The answer demonstrates a solid understanding of the core concepts."
    gaps = "Missing key concepts." if len(missed) > 0 else "No major gaps identified."
    suggestion = "Review the key reference material." if overall < 80 else "Keep up the excellent work!"

    if len(missed) > 0:
        gaps = f"Missing important terminology or entities such as: {', '.join(missed[:3])}."
        
    return {
        "scores": {
            "semantic": semantic_score,
            "factual": completeness_score, 
            "completeness": completeness_score,
            "clarity": min(100, clarity_score),
            "relevance": min(100, relevance_score)
        },
        "overall": overall,
        "strengths": strengths,
        "gaps": gaps,
        "suggestion": suggestion,
        "keywords_matched": matched[:10],
        "keywords_missed": missed[:10]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
