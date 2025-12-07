import nltk
import streamlit as st
import spacy
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import language_tool_python

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Load tools
# ✅ Efficiently cache heavy tools and models
@st.cache_resource
def load_nlp_tools():
    nlp = spacy.load("en_core_web_sm")
    tool = language_tool_python.LanguageTool('en-US')
    return nlp, tool

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_tone_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 🔃 Load all tools
nlp, tool = load_nlp_tools()
bert_model = load_embedding_model()
tone_classifier = load_tone_classifier()
hf_tone_classifier = tone_classifier 

def score_relevance(question, answer):
    q_emb = bert_model.encode(question, convert_to_tensor=True)
    a_emb = bert_model.encode(answer, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(q_emb, a_emb).item(), 2)

def score_depth(answer):
    word_count = len(word_tokenize(answer))
    return round(min(word_count / 50, 1.0), 2)

def score_answer(analysis):
    score = (analysis['relevance'] + analysis['depth'] + analysis['clarity']) / 3
    return max(0, int(score))

def suggest_improvement(analysis):
    suggestions = []
    if analysis["relevance"] < 60:
        suggestions.append("Focus more on answering the question directly.")
    if analysis["depth"] < 60:
        suggestions.append("Provide more detailed examples.")
    if analysis["clarity"] < 60:
        suggestions.append("Improve clarity or sentence structure.")
    return suggestions or ["Well done! Keep going."]

def score_clarity(answer):
    matches = tool.check(answer)
    errors = len(matches)
    return round(1 - min(errors / max(len(answer.split()), 1), 1.0), 2)

def extract_entities(answer):
    doc = nlp(answer)
    return ", ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents])

def extract_concepts(answer):
    doc = nlp(answer)
    return ", ".join([chunk.text for chunk in doc.noun_chunks])

def rate_tone(answer):
    labels = ["confident", "uncertain", "professional", "casual"]
    result = hf_tone_classifier(answer, candidate_labels=labels)
    top = result["labels"][0]
    score = round(result["scores"][0], 2)
    return (top, score)

def is_unclear_answer(answer):
    unclear_labels = ["unclear", "incomplete", "confused", "irrelevant"]
    result = hf_tone_classifier(answer, candidate_labels=unclear_labels)
    top_label = result["labels"][0]
    score = result["scores"][0]
    return top_label in unclear_labels and score > 0.6

def analyze_answer(question, answer):
    relevance_score = score_relevance(question, answer) * 100  # convert to 0-100
    depth_score = score_depth(answer) * 100
    clarity_score = score_clarity(answer) * 100
    tone = rate_tone(answer)
    entities = extract_entities(answer)
    concepts = extract_concepts(answer)
    
    return {
        "relevance": relevance_score,
        "depth": depth_score,
        "clarity": clarity_score,
        "tone": tone,
        "entities": entities,
        "concepts": concepts
    }

def tone_classifier(text, labels):
    # Simple simulation for local models
    for label in labels:
        if any(word in text.lower() for word in ["idk", "no idea", "not sure", "maybe", "nothing", "na", "don't know"]):
            return {"labels": [label], "scores": [0.9]}
    return {"labels": ["clear"], "scores": [0.1]}
