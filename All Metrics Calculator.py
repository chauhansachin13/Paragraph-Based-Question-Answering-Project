import os
import re
import json
import csv
from transformers import pipeline, logging
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Environment Setup
# =============================
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.set_verbosity_error()

# =============================
# Load NLP Model for KG
# =============================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =============================
# Load QA Models
# =============================
def load_all_models():
    models = {}
    models["DistilBERT"] = pipeline("question-answering",
                                   model="distilbert/distilbert-base-cased-distilled-squad",
                                   revision="564e9b5",
                                   device=-1,
                                   framework="pt")
    models["BERT"] = pipeline("question-answering",
                             model="bert-large-uncased-whole-word-masking-finetuned-squad",
                             device=-1,
                             framework="pt")
    models["RoBERTa"] = pipeline("question-answering",
                                model="deepset/roberta-base-squad2",
                                device=-1,
                                framework="pt")
    models["Sentence-BERT"] = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return models

qa_pipelines = load_all_models()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =============================
# LDA (keyword) based QA
# =============================
def lda_keyword_answer(question, context):
    question_keywords = question.lower().split()
    best_sentence = ""
    max_overlap = 0
    for sentence in re.split(r'[.!?]', context):
        words = sentence.lower().split()
        overlap = len(set(words) & set(question_keywords))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence.strip()
    return best_sentence if best_sentence else "No matching content found."

# =============================
# Evaluation Metrics
# =============================
def compute_exact_match(prediction, ground_truth):
    return int(prediction.strip().lower() == ground_truth.strip().lower())

def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# =============================
# Process CSV File
# =============================
def process_csv_file(input_path, output_path):
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare output data
    output_data = []
    
    for item in data:
        context = item['context']
        questions = item['questions']
        answers = item['answers']
        
        for q, expected in zip(questions, answers):
            answers = {}
            try:
                # Get answers from all models
                for model_name, model in qa_pipelines.items():
                    try:
                        if model_name == "Sentence-BERT":
                            # Special handling for Sentence-BERT
                            sentences = [s.strip() for s in re.split(r'[.!?]', context) if s.strip()]
                            if not sentences:
                                answers[model_name] = "No sentences found in context."
                            else:
                                question_embedding = model.encode(q, convert_to_tensor=True)
                                sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
                                cos_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
                                best_idx = torch.argmax(cos_scores).item()
                                answers[model_name] = sentences[best_idx]
                        else:
                            # Standard QA models
                            result = model(question=q, context=context)
                            answers[model_name] = result['answer']
                    except Exception as e:
                        answers[model_name] = f"Error with {model_name}: {str(e)}"
                
                # Get LDA answer
                answers["LDA"] = lda_keyword_answer(q, context)
                
                # Calculate metrics for each model
                metrics = []
                expected_embedding = embedder.encode(expected)
                
                for model, ans in answers.items():
                    if not ans.startswith("Error"):
                        # Encode the model's answer
                        ans_embedding = embedder.encode(ans)
                        
                        # Calculate metrics
                        cos_sim = cosine_similarity(
                            expected_embedding.reshape(1, -1),
                            ans_embedding.reshape(1, -1))[0][0]
                        em = compute_exact_match(ans, expected)
                        f1 = compute_f1(ans, expected)
                        
                        metrics.append({
                            'model': model,
                            'answer': ans,
                            'cos_sim': cos_sim,
                            'em': em,
                            'f1': f1
                        })
                
                # Add to output data
                for m in metrics:
                    output_data.append({
                        'category': item['category'],
                        'context': context,
                        'question': q,
                        'expected_answer': expected,
                        'model': m['model'],
                        'model_answer': m['answer'],
                        'cosine_similarity': m['cos_sim'],
                        'exact_match': m['em'],
                        'f1_score': m['f1']
                    })
                        
            except Exception as e:
                print(f"Error processing question: {q}. Error: {e}")
    
    # Write to output CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['category', 'context', 'question', 'expected_answer', 
                     'model', 'model_answer', 'cosine_similarity', 'exact_match', 'f1_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

# =============================
# Main Execution
# =============================
if __name__ == '__main__':
    input_path = r"C:\Users\pragy\Downloads\momos2.csv"
    output_path = r"C:\Users\pragy\Downloads\momos_output3.csv"
    
    print("Starting processing...")
    process_csv_file(input_path, output_path)
    print(f"Processing complete. Results saved to {output_path}")