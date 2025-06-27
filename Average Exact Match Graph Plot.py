import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['DistilBERT', 'BERT', 'RoBERTa', 'Sentence-BERT', 'LDA']
exact_scores = [0.27027027, 0.297297297, 0.288288288, 0.0, 0.0]

# Create DataFrame
df = pd.DataFrame({
    'Model': models,
    'Exact Match Score': exact_scores
})

# Create figure with window title
fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title('Average Exact Match Score of All 5 QA Models')

# Plot
bars = plt.bar(df['Model'], df['Exact Match Score'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Customize plot
plt.title('Average Exact Match Score of All 5 QA Models', fontsize=14, pad=20)
plt.ylabel('Average Exact Match Score', fontsize=12)
plt.ylim(0, 0.35)  # Adjusted range for exact match scores
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add precise value labels
for bar in bars:
    height = bar.get_height()
    label = f'{height:.9f}'.rstrip('0').rstrip('.') if height != 0 else '0'
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
             label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()