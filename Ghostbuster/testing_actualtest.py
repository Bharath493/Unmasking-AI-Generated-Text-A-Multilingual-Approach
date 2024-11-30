import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score,ConfusionMatrixDisplay

test_df = pd.read_json("subtaskA_test_multilingual_labels.jsonl", lines=True)
test_df = test_df[:14940]

pred_df = pd.read_csv('output_actualtest.txt', header=None, names=['Value'])

#Finding all the Nil indices
nil_indices = pred_df.index[pred_df['Value'] == 'Nil'].tolist()

# Remove 'Nil' values from pred_df and reset the index
filtered_pred = pred_df[pred_df['Value'] != 'Nil'].reset_index(drop=True)

# Remove corresponding rows in test_df
filtered_df = test_df.drop(nil_indices).reset_index(drop=True)


filtered_pred['Value'] = pd.to_numeric(filtered_pred['Value'], errors='coerce')

# Apply a lambda function to transform the values
filtered_pred['Value'] = filtered_pred['Value'].apply(lambda x: 1 if x > 50 else 0)
print(len(filtered_pred.Value))
print(len(filtered_df.label))
# print(filtered_pred.Value)
# print(filtered_df.label[:9])




accuracy = accuracy_score(filtered_df.label, filtered_pred.Value)
precision = precision_score(filtered_df.label, filtered_pred.Value, pos_label=1, average='binary')
recall = recall_score(filtered_df.label, filtered_pred.Value,pos_label=1, average='binary')
score = f1_score(filtered_df.label, filtered_pred.Value, pos_label=1, average='binary')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print(f"F1 Score: {score}")