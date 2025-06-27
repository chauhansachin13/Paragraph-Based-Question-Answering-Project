import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['DistilBERT', 'BERT', 'RoBERTa', 'Sentence-BERT', 'LDA']
cosine_scores = [0.705942784, 0.761427271, 0.756846418, 0.415019592, 0.332974718]

# Create DataFrame
df = pd.DataFrame({
    'Model': models,
    'Cosine Similarity': cosine_scores
})

# Create figure with window title
fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title('Average Cosine Similarity Score of All 5 QA Models')

# Plot
bars = plt.bar(df['Model'], df['Cosine Similarity'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Customize plot
plt.title('Average Cosine Similarity Score of All 5 QA Models', fontsize=14, pad=20)
plt.ylabel('Average Cosine Similarity Score', fontsize=12)
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.yticks([i/10 for i in range(0, 11)])  # Set y-axis ticks at 0.1 intervals
plt.xticks(rotation=45, ha='right')  # Rotate model names for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars (with full precision)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
             f'{height:.9f}'.rstrip('0').rstrip('.') if '.' in f'{height:.9f}' else f'{height:.9f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()