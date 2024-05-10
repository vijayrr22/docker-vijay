import matplotlib.pyplot as plt

# Data
models = [
    "POS+Naïve Bayes",
    "Feature-Based SVM",
    "Gini-Index SVM",
    "L+N+I+D+R+S",
    "Lexicon Based SVM",
    "Rule-Based (ATE) + Fine tuned BERT model (OUR Model)"
]
precision = [0.70, 0.66, 0.70, 0.72, 0.74, 0.75]
recall = [0.72, 0.61, 0.79, 0.79, 0.72, 0.75]

# Highlight color for "Our Model"
colors = ['blue' if 'OUR Model' not in model else 'red' for model in models]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(precision, recall, marker='o', c=colors, edgecolors='black')

# Annotate points with model names with custom positions
for i, model in enumerate(models):
    if 'Gini-Index SVM' in model:
        offset = (-0.02, -0.01)  # Adjust the position for "Gini-Index SVM"
    elif 'L+N+I+D+R+S' in model:
        offset = (0.002, -0.005)  # Adjust the position for "L+N+I+D+R+S"
    elif 'OUR Model' in model:
        offset = (-0.04, 0.005)  # Adjust the position for "Our Model"
    else:
        offset = (0.002, -0.005)  # Default position
    plt.annotate(model, (precision[i], recall[i]), fontsize=12 if 'OUR Model' not in model else 14,
                 xytext=(precision[i] + offset[0], recall[i] + offset[1]), textcoords='data', color='black')

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall for Different Models')
plt.grid(True)
plt.tight_layout()
plt.show()
