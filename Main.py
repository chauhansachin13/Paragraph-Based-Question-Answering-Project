import os
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from transformers import pipeline, logging
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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
# Enhanced Knowledge Graph Generator with Proper Relationships
# =============================
def generate_knowledge_graph(text, container=None, separate_window=False):
    doc = nlp(text)
    edges = []
    nodes = set()
    
    for sent in doc.sents:
        # Extract entities and important words
        entities = [ent.text for ent in sent.ents]
        important_words = [token.text for token in sent 
                         if token.pos_ in ["NOUN", "PROPN", "VERB"] 
                         and not token.is_stop 
                         and len(token.text) > 2]
        
        all_nodes = list(set(entities + important_words))
        
        # Create relationships between consecutive important words
        for i in range(len(all_nodes)-1):
            edges.append((all_nodes[i], "related to", all_nodes[i+1]))
            nodes.update([all_nodes[i], all_nodes[i+1]])
        
        # Extract subject-predicate-object relationships
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text
                predicate = token.head.text
                edges.append((subject, "is subject of", predicate))
                nodes.update([subject, predicate])
            elif token.dep_ in ("dobj", "pobj"):
                object_ = token.text
                predicate = token.head.text
                edges.append((predicate, "acts on", object_))
                nodes.update([predicate, object_])
            elif token.dep_ == "attr":
                attribute = token.text
                subject = token.head.text
                edges.append((subject, "has attribute", attribute))
                nodes.update([subject, attribute])

    # Create graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0], edge[2], label=edge[1])

    # Create figure with larger size
    fig = plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.0)  # Increased k for better spacing
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2500)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Draw edges with labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Knowledge Graph with Relationships", fontsize=14)
    plt.axis('off')

    if separate_window:
        # Create a new window for the graph
        graph_window = tk.Toplevel()
        graph_window.title("Knowledge Graph - Detailed View")
        graph_window.geometry("1200x900")
        
        # Add canvas with navigation toolbar
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        close_btn = tk.Button(graph_window, text="Close", command=graph_window.destroy)
        close_btn.pack(pady=10)
    else:
        # Display in the provided container
        if container:
            # Clear previous graph
            for widget in container.winfo_children():
                widget.destroy()
            
            # Create scrollable frame
            scroll_frame = tk.Frame(container)
            scroll_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add canvas
            canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            hscroll = tk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL, command=canvas.get_tk_widget().xview)
            vscroll = tk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=canvas.get_tk_widget().yview)
            canvas.get_tk_widget().configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
            
            hscroll.pack(side=tk.BOTTOM, fill=tk.X)
            vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add button to open in separate window
            open_btn = tk.Button(container, text="Open in Separate Window", 
                                command=lambda: generate_knowledge_graph(text, None, True))
            open_btn.pack(pady=5)

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
# Visualization Functions
# =============================
def show_metrics_visualization(metrics_data):
    """Display metrics in both table and graph formats"""
    if not metrics_data:
        messagebox.showwarning("No Data", "No metrics data available for visualization.")
        return
    
    # Create visualization window
    vis_window = tk.Toplevel()
    vis_window.title("Model Performance Metrics")
    vis_window.geometry("1200x800")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(vis_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Tab 1: Table View
    table_frame = tk.Frame(notebook)
    notebook.add(table_frame, text="Table View")
    
    # Create treeview widget with columns in the requested order
    tree = ttk.Treeview(table_frame, columns=('Model', 'Cosine Similarity', 'Exact Match', 'F1 Score'), show='headings')
    
    # Configure columns
    tree.heading('Model', text='Model')
    tree.heading('Cosine Similarity', text='Cosine Similarity')
    tree.heading('Exact Match', text='Exact Match')
    tree.heading('F1 Score', text='F1 Score')
    
    tree.column('Model', width=150, anchor='w')
    tree.column('Cosine Similarity', width=150, anchor='center')
    tree.column('Exact Match', width=150, anchor='center')
    tree.column('F1 Score', width=150, anchor='center')
    
    # Add data to table
    for metric in metrics_data:
        tree.insert('', 'end', values=(
            metric['model'],
            f"{metric['cos_sim']:.4f}",
            metric['em'],
            f"{metric['f1']:.4f}"
        ))
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Tab 2: Graph View
    graph_frame = tk.Frame(notebook)
    notebook.add(graph_frame, text="Graph View")
    
    # Create figure for the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting all three metrics
    models = [m['model'] for m in metrics_data]
    cos_sim = [m['cos_sim'] for m in metrics_data]
    em_scores = [m['em'] for m in metrics_data]
    f1_scores = [m['f1'] for m in metrics_data]
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width*2 for x in r1]
    
    # Create bars for all three metrics
    ax.bar(r1, cos_sim, color='#1f77b4', width=bar_width, edgecolor='white', label='Cosine Similarity')
    ax.bar(r2, em_scores, color='#2ca02c', width=bar_width, edgecolor='white', label='Exact Match')
    ax.bar(r3, f1_scores, color='#ff7f0e', width=bar_width, edgecolor='white', label='F1 Score')
    
    # Add labels and title
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(models))])
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Create canvas and add to frame
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, graph_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add close button
    close_btn = tk.Button(vis_window, text="Close", command=vis_window.destroy)
    close_btn.pack(pady=10)

# =============================
# GUI Class
# =============================
class QAGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("🧠 Multi-Model QA Tool with Knowledge Graph")
        self.model_name = "DistilBERT"
        self.metrics_data = []  # Store metrics for visualization
        
        # UI Configuration
        self.configure_ui()
        
        # Initialize models
        self.qa_pipelines = load_all_models()
        self.qa_pipeline = self.qa_pipelines[self.model_name]
        
    def configure_ui(self):
        self.master.geometry("1200x900")
        
        # Colors and fonts
        self.bg_color = "#f0f2f5"
        self.fg_color = "#1c1e21"
        self.button_color = "#1877f2"
        self.box_color = "#ffffff"
        
        # Main frame
        self.main_frame = tk.Frame(self.master, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Context input
        self.context_label = tk.Label(self.main_frame, text="📝 Context Paragraph:", 
                                    font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.context_label.pack(anchor='w', pady=(0, 5))
        
        self.context_box = scrolledtext.ScrolledText(self.main_frame, height=10, width=100,
                                                   bg=self.box_color, fg=self.fg_color,
                                                   font=("Segoe UI", 10))
        self.context_box.pack(fill=tk.X, pady=(0, 15))
        
        # Question input
        self.question_label = tk.Label(self.main_frame, text="❓ Your Question:", 
                                     font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.question_label.pack(anchor='w', pady=(0, 5))
        
        self.question_entry = tk.Entry(self.main_frame, width=100,
                                     bg=self.box_color, fg=self.fg_color,
                                     font=("Segoe UI", 10))
        self.question_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Expected answer (for evaluation)
        self.expected_label = tk.Label(self.main_frame, text="🎯 Expected Answer (optional):", 
                                     font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.expected_label.pack(anchor='w', pady=(0, 5))
        
        self.expected_entry = tk.Entry(self.main_frame, width=100,
                                     bg=self.box_color, fg=self.fg_color,
                                     font=("Segoe UI", 10))
        self.expected_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Button frame
        self.button_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        self.ask_button = tk.Button(self.button_frame, text="🔍 Get Answer", 
                                  command=self.answer_question,
                                  bg=self.button_color, fg="white",
                                  font=("Segoe UI", 10, "bold"))
        self.ask_button.pack(side=tk.LEFT, padx=5)
        
        self.kg_button = tk.Button(self.button_frame, text="🌐 Generate Knowledge Graph", 
                                 command=self.show_knowledge_graph,
                                 bg=self.button_color, fg="white",
                                 font=("Segoe UI", 10, "bold"))
        self.kg_button.pack(side=tk.LEFT, padx=5)
        
        # Model selection
        self.model_label = tk.Label(self.button_frame, text="Model:", 
                                  font=("Segoe UI", 10), bg=self.bg_color, fg=self.fg_color)
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        self.model_dropdown = ttk.Combobox(self.button_frame, 
                                         values=["DistilBERT", "BERT", "RoBERTa", "Sentence-BERT", "LDA"],
                                         state="readonly",
                                         font=("Segoe UI", 10))
        self.model_dropdown.set(self.model_name)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.change_model)
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Current model display
        self.model_display = tk.Label(self.button_frame, 
                                    text=f"Current: {self.model_name}",
                                    font=("Segoe UI", 10), bg=self.bg_color, fg="#606770")
        self.model_display.pack(side=tk.LEFT, padx=10)
        
        # Visualization button
        self.vis_button = tk.Button(self.button_frame, text="📊 Show Performance Metrics", 
                                  command=self.show_performance_metrics,
                                  bg="#4CAF50", fg="white",
                                  font=("Segoe UI", 10, "bold"))
        self.vis_button.pack(side=tk.LEFT, padx=5)
        
        # Answer display
        self.answer_label = tk.Label(self.main_frame, text="✅ Answer:", 
                                   font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.answer_label.pack(anchor='w', pady=(10, 5))
        
        self.answer_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=100,
                                                    bg=self.box_color, fg=self.fg_color,
                                                    font=("Segoe UI", 10))
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Knowledge Graph frame
        self.kg_label = tk.Label(self.main_frame, text="🧠 Knowledge Graph:", 
                                font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.kg_label.pack(anchor='w', pady=(10, 5))
        
        self.canvas_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def answer_question(self):
        context = self.context_box.get("1.0", tk.END).strip()
        question = self.question_entry.get().strip()
        expected = self.expected_entry.get().strip()

        if not context or not question:
            messagebox.showwarning("Input Missing", "Please enter both context and a question.")
            return

        answers = {}
        self.metrics_data = []  # Reset metrics data
        
        try:
            # Get answers from all models
            for model_name, model in self.qa_pipelines.items():
                try:
                    if model_name == "Sentence-BERT":
                        # Special handling for Sentence-BERT
                        sentences = [s.strip() for s in re.split(r'[.!?]', context) if s.strip()]
                        if not sentences:
                            answers[model_name] = "No sentences found in context."
                        else:
                            question_embedding = model.encode(question, convert_to_tensor=True)
                            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
                            cos_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
                            best_idx = torch.argmax(cos_scores).item()
                            answers[model_name] = sentences[best_idx]
                    else:
                        # Standard QA models
                        result = model(question=question, context=context)
                        answers[model_name] = result['answer']
                except Exception as e:
                    answers[model_name] = f"Error with {model_name}: {str(e)}"
            
            # Get LDA answer
            answers["LDA"] = lda_keyword_answer(question, context)
            
        except Exception as e:
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, f"Error occurred: {e}")
            return

        # Prepare result text
        result_text = ""
        
        # Show all answers first
        result_text += "📋 Model Answers:\n"
        for model, ans in answers.items():
            result_text += f"• {model}: {ans}\n"
        
        # Add evaluation if expected answer provided
        if expected:
            try:
                # Encode the expected answer once
                expected_embedding = embedder.encode(expected)
                
                result_text += "\n🔎 Evaluation Metrics (vs Expected Answer):\n"
                metrics = []
                
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
                
                # Store metrics for visualization
                self.metrics_data = metrics.copy()
                
                # Sort by cosine similarity (descending)
                metrics.sort(key=lambda x: x['cos_sim'], reverse=True)
                
                # Display metrics
                for m in metrics:
                    result_text += (
                        f"\n• {m['model']}:\n"
                        f"  - Cosine Similarity: {m['cos_sim']:.4f}\n"
                        f"  - Exact Match: {m['em']}\n"
                        f"  - F1 Score: {m['f1']:.4f}\n"
                    )
                
                # Highlight best model
                if metrics:
                    best = metrics[0]
                    result_text += (
                        f"\n🏆 Best Performing Model: {best['model']}\n"
                        f"  - Highest Cosine Similarity: {best['cos_sim']:.4f}\n"
                        f"  - F1 Score: {best['f1']:.4f}\n"
                    )
                    
            except Exception as e:
                result_text += f"\n\n⚠️ Evaluation error: {str(e)}"

        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, result_text)

    def change_model(self, event=None):
        selected = self.model_dropdown.get()
        if selected in self.qa_pipelines or selected == "LDA":
            self.model_name = selected
            self.model_display.config(text=f"Current: {self.model_name}")

    def show_knowledge_graph(self):
        context = self.context_box.get("1.0", tk.END).strip()
        if not context:
            messagebox.showwarning("Missing Context", "Please provide context to generate a knowledge graph.")
            return
        
        try:
            # Ask user if they want to open in separate window
            choice = messagebox.askyesno("Knowledge Graph", 
                                       "Generate knowledge graph in separate window for better visibility?")
            generate_knowledge_graph(context, None if choice else self.canvas_frame, choice)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate knowledge graph: {str(e)}")
    
    def show_performance_metrics(self):
        """Show performance metrics in table and graph formats"""
        if not self.metrics_data:
            messagebox.showwarning("No Data", "No metrics data available. Please run an evaluation first.")
            return
        
        show_metrics_visualization(self.metrics_data)

# =============================
# Launch GUI
# =============================
if __name__ == '__main__':
    root = tk.Tk()
    app = QAGUI(root)
    root.mainloop()