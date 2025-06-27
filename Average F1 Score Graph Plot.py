import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['DistilBERT', 'BERT', 'RoBERTa', 'Sentence-BERT', 'LDA']
f1_scores = [0.528133347, 0.589311842, 0.579169611, 0.187254978, 0.145105978]

# Create DataFrame
df = pd.DataFrame({
    'Model': models,
    'F1 Score': f1_scores
})

# Create figure with window title
fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title('Average F1 Score of All 5 QA Models')

# Plot
bars = plt.bar(df['Model'], df['F1 Score'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Customize plot
plt.title('Average F1 Score of All 5 QA Models', fontsize=14, pad=20)
plt.ylabel('Average F1 Score', fontsize=12)
plt.ylim(0, 1)  # Full range from 0 to 1
plt.yticks([i/10 for i in range(0, 11)])  # Ticks at 0.1 intervals
plt.xticks(rotation=45, ha='right')  # Rotate model names for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars (with full precision)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
             f'{height:.6f}'.rstrip('0').rstrip('.') if '.' in f'{height:.6f}' else f'{height:.6f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()