import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read the CSV file
df = pd.read_csv('fairface_label_val_expanded.csv')

# Mapping dictionary for converting labels to binary format
label_mapping = {'Female': 0, 'Male': 1}

# Calculate FPR, FNR, TPR, TNR, and AUC for each race
unique_races = df['race'].unique()
results = []

# Define colors and labels for each race
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'gray']
labels = []

for idx, race in enumerate(unique_races):
    race_df = df[df['race'] == race]
    
    # Convert labels to binary format
    race_df['gender_binary'] = race_df['gender'].map(label_mapping)
    race_df['predicted_gender_binary'] = race_df['predicted_gender'].map(label_mapping)
    
    true_positive = race_df[(race_df['predicted_gender_binary'] == 1) & (race_df['gender_binary'] == 1)].shape[0]
    false_positive = race_df[(race_df['predicted_gender_binary'] == 1) & (race_df['gender_binary'] == 0)].shape[0]
    true_negative = race_df[(race_df['predicted_gender_binary'] == 0) & (race_df['gender_binary'] == 0)].shape[0]
    false_negative = race_df[(race_df['predicted_gender_binary'] == 0) & (race_df['gender_binary'] == 1)].shape[0]
    
    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (false_negative + true_positive)
    tnr = true_negative / (true_negative + false_positive)
    
    # Calculate AUC
    fpr_values, tpr_values, _ = roc_curve(race_df['gender_binary'], race_df['predicted_gender_binary'])
    auc_value = auc(fpr_values, tpr_values)
    
    results.append({
        'Race': race,
        'TPR': tpr,
        'FPR': fpr,
        'FNR': fnr,
        'TNR': tnr,
        'AUC': auc_value
    })
    
    labels.append(race)

# Create a DataFrame with the results
result_df = pd.DataFrame(results)

# Plot the graphs
plt.figure(figsize=(12, 6))

# False Positive Rate (FPR) vs. True Positive Rate (TPR)
plt.subplot(1, 2, 1)
for idx, race in enumerate(unique_races):
    plt.plot(result_df.loc[result_df['Race'] == race, 'FPR'], result_df.loc[result_df['Race'] == race, 'TPR'],
             color=colors[idx], marker='o', label=f'{labels[idx]} (AUC = {result_df.loc[result_df["Race"] == race, "AUC"].values[0]:.2f})')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

# False Negative Rate (FNR) vs. True Negative Rate (TNR)
plt.subplot(1, 2, 2)
for idx, race in enumerate(unique_races):
    plt.plot(result_df.loc[result_df['Race'] == race, 'FNR'], result_df.loc[result_df['Race'] == race, 'TNR'],
             color=colors[idx], marker='o', label=f'{labels[idx]} (AUC = {result_df.loc[result_df["Race"] == race, "AUC"].values[0]:.2f})')

plt.xlabel('False Negative Rate (FNR)')
plt.ylabel('True Negative Rate (TNR)')
plt.title('Error Tradeoff Curve')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
