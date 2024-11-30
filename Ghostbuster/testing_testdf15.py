import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score,ConfusionMatrixDisplay

train_df = pd.read_json("subtaskA_train_multilingual.jsonl", lines=True)
dev_df = pd.read_json("subtaskA_dev_multilingual.jsonl", lines=True)
train_df_merged = pd.concat([train_df,dev_df])

source_dict={}
for source in train_df_merged.source.unique():
  source_dict[source] = train_df_merged[train_df_merged.source == source]
print(source_dict.keys())

model_dict = {}
for source,df in source_dict.items():
  model_dict[f'{source}_0'] = df[df.model == "human"]
  model_dict[f'{source}_1'] = df[(df.model.isin(["chatGPT","davinci","bloomz","dolly","cohere"]))]

  
#Creating train_data,val_data,test_data
first = model_dict["wikihow_0"]
train_df_70 = first[:int(0.7 * len(first))]
val_df_15 = first[int(0.7 * len(first)): int(0.85 * len(first))]
test_df_15 = first[int(0.85 * len(first)):]
for k , v in model_dict.items():
  if k == "wikihow_0":
    continue
  slice_train= v[:int(0.7*len(v))]
  slice_val = v[int(0.7 * len(v)): int(0.85 * len(v))]
  slice_test= v[int(0.85 * len(v)):]
  train_df_70 = pd.concat([train_df_70,slice_train])
  val_df_15 = pd.concat([val_df_15,slice_val])
  test_df_15 = pd.concat([test_df_15,slice_test])

print(len(test_df_15))


#train_data= train_df_70
#val_data = val_df_15
#test_data = test_df_10

test_df_15 = test_df_15.sample(frac=1, random_state=42).reset_index(drop=True)
test_df_15 = test_df_15[:11370]  ####Partial run
pred_df = pd.read_csv('output_testdf15.txt', header=None, names=['Value'])

print(len(test_df_15))
print(len(pred_df))

# print(test_df_15.text[8580])
nil_indices = pred_df.index[pred_df['Value'] == 'Nil'].tolist()
whatis = test_df_15.loc[nil_indices]
print(whatis.source.value_counts())

# print(len(nil_indices))
filtered_pred = pred_df[pred_df['Value'] != 'Nil'].reset_index(drop=True)
filtered_df = test_df_15.drop(nil_indices).reset_index(drop=True)
print(filtered_df[filtered_df.source=="chinese"]['label'])

filtered_pred['Value'] = pd.to_numeric(filtered_pred['Value'], errors='coerce')
filtered_pred['Value'] = filtered_pred['Value'].apply(lambda x: 1 if x > 50 else 0)
print(len(filtered_pred.Value))
print(len(filtered_df.label))

#filtered_pred   -> predicted
#filtered_df     -> groundtruth

#Testing on test_df_15
print("Testing on test_df_15")

accuracy = accuracy_score(filtered_df.label, filtered_pred.Value)
precision = precision_score(filtered_df.label, filtered_pred.Value, pos_label=1, average='binary')
recall = recall_score(filtered_df.label, filtered_pred.Value,pos_label=1, average='binary')
score = f1_score(filtered_df.label, filtered_pred.Value, pos_label=1, average='binary')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print(f"F1 Score: {score}")
print("----------------------------------")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Testing by Language
lang_list = ["English","bulgarian","chinese","indonesian","urdu","russian","arabic","german"]
accuracy_list=[]
precision_list=[]
recall_list=[]
for lang in lang_list:
    if lang== "English":
        print("Model performance in English")
        l_indices = filtered_df.index[filtered_df.source.isin(["reddit","wikihow","arxiv","wikipedia","peerread"])].tolist()

    else:
        print(f"Model performance in {lang}")
        l_indices = filtered_df.index[filtered_df.source== lang].tolist()

    print("----------------------------------")
    pred = filtered_pred.loc[l_indices]
    trueth = filtered_df.loc[l_indices]

    accuracy = accuracy_score(trueth.label, pred.Value)
    precision = precision_score(trueth.label, pred.Value, pos_label=1, average='binary')
    recall = recall_score(trueth.label, pred.Value,pos_label=1, average='binary')
    
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
# Create DataFrame
accuracy_list = [round(x*100,1) for x in accuracy_list] 
precision_list = [round(x*100,1) for x in precision_list]  
recall_list = [round(x*100,1) for x in recall_list]     
df = pd.DataFrame({
    'Language': lang_list,
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list
})
# # Function to create bar plots for a given metric
# def plot_metric(data, metric, ax, color,bar_width=0.5):
#     data[metric] = data[metric] * 100
#     bars = ax.bar(data['Language'], data[metric], color=color,width=bar_width)
#     ax.set_title(f'Model performance on each language', fontsize=11)
#     ax.set_xlabel('Language', fontsize=10)
#     ax.set_ylabel(f'{metric}(%)', fontsize=10)
#     ax.set_ylim(0, 120)  # Set the y-axis limits to make differences more noticeable
#     ax.set_xticklabels(data['Language'], rotation=45)
#     # Adding data labels on top of each bar
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom',ha='center')  # va: vertical alignment

# # Create figure and axes
# fig, ax = plt.subplots(1, 3, figsize=(14, 4))  # 3 rows, 1 column
# # print(accuracy_list)
# # print(precision_list)
# # print(recall_list)
# # Plot each metric in a separate subplot
# plot_metric(df, 'Accuracy', ax[0], 'indigo')
# plot_metric(df, 'Precision', ax[1], 'green')
# plot_metric(df, 'Recall', ax[2], 'darkorange')

# plt.tight_layout()
# plt.show()

# Plotting directly from DataFrame
fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted figure size for better fit
# Plotting with specified colors
color_dict = {'Accuracy': 'indigo', 'Precision': 'green', 'Recall': 'orange'}
df.set_index('Language').plot(kind='bar', ax=ax, width=0.7, color=[color_dict[x] for x in df.columns[1:]])

# Customize plot with labels and title
ax.set_title('Ghostbuster Performance by Language', fontsize=13)
ax.set_xlabel('Language', fontsize=12)
ax.set_ylabel('Metrics (in %)', fontsize=12)
ax.set_ylim(0, 140)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Function to add labels on top of each bar
def add_labels(ax):
    for p in ax.patches:  # Access each bar that's plotted
        ax.annotate(f"{p.get_height():.1f}",  # Get the height and format it as a decimal
                   (p.get_x() + p.get_width() / 2., p.get_height()),  # Position for the label
                   ha='center', va='center', fontsize=10, color='black', rotation=0, xytext=(0, 4),
                   textcoords='offset points')

add_labels(ax)  # Call the function to add labels to the bars

# Adding legend
ax.legend(title='Metric Type')

plt.tight_layout()
plt.show()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Domain specific for English
domain_list = ["reddit","wikihow","arxiv","wikipedia","peerread"]
accuracy_list=[]
precision_list=[]
recall_list=[]
for domain in domain_list:
   print(f"Model Performance on {domain}")
   print("-----------------------------\n")
   d_indices = filtered_df.index[filtered_df.source== domain].tolist()
   pred = filtered_pred.loc[d_indices]
   trueth = filtered_df.loc[d_indices]

   accuracy = accuracy_score(trueth.label, pred.Value)
   precision = precision_score(trueth.label, pred.Value, pos_label=1, average='binary')
   recall = recall_score(trueth.label, pred.Value,pos_label=1, average='binary')
  
   accuracy_list.append(accuracy)
   precision_list.append(precision)
   recall_list.append(recall)

   print("Accuracy:", accuracy)
   print("Precision:", precision)
   print("Recall:", recall)

accuracy_list = [round(x*100,1) for x in accuracy_list] 
precision_list = [round(x*100,1) for x in precision_list]  
recall_list = [round(x*100,1) for x in recall_list]

# Create DataFrame
df = pd.DataFrame({
    'Domain': domain_list,
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list
})
# # Function to create bar plots for a given metric
# def plot_metric(data, metric, ax, color,bar_width=0.5):
#     data[metric] = data[metric] * 100
#     bars = ax.bar(data['domain'], data[metric], color=color,width=bar_width)
#     ax.set_title(f'Model performance on each domain for English', fontsize=11)
#     ax.set_xlabel('domain', fontsize=10)
#     ax.set_ylabel(f'{metric}(%)', fontsize=10)
#     ax.set_ylim(0, 120)  # Set the y-axis limits to make differences more noticeable
#     ax.set_xticklabels(data['domain'], rotation=45)
#     # Adding data labels on top of each bar
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom',ha='center')  # va: vertical alignment

# # Create figure and axes
# fig, ax = plt.subplots(1, 3, figsize=(14, 4))  # 3 rows, 1 column

# # Plot each metric in a separate subplot
# plot_metric(df, 'Accuracy', ax[0], 'indigo')
# plot_metric(df, 'Precision', ax[1], 'green')
# plot_metric(df, 'Recall', ax[2], 'darkorange')

# plt.tight_layout()
# plt.show()

# Plotting directly from DataFrame
fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted figure size for better fit
# Plotting with specified colors
color_dict = {'Accuracy': 'indigo', 'Precision': 'green', 'Recall': 'orange'}
df.set_index('Domain').plot(kind='bar', ax=ax, width=0.7, color=[color_dict[x] for x in df.columns[1:]])

# Customize plot with labels and title
ax.set_title('Ghostbuster Performance by Domain for English', fontsize=13)
ax.set_xlabel('Domain', fontsize=12)
ax.set_ylabel('Metrics (in %)', fontsize=12)
ax.set_ylim(0, 140)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Function to add labels on top of each bar
def add_labels(ax):
    for p in ax.patches:  # Access each bar that's plotted
        ax.annotate(f"{p.get_height():.1f}",  # Get the height and format it as a decimal
                   (p.get_x() + p.get_width() / 2., p.get_height()),  # Position for the label
                   ha='center', va='center', fontsize=10, color='black', rotation=0, xytext=(0, 4),
                   textcoords='offset points')

add_labels(ax)  # Call the function to add labels to the bars

# Adding legend
ax.legend(title='Metric Type')

plt.tight_layout()
plt.show()


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#LLM specific performance

accuracy_list = []
llm_list = ["chatGPT","davinci","bloomz","dolly","cohere"]
for llm in llm_list:
    print(f"Model Performance on data generated by {llm} ")
    print("---------------------------------------------\n")
  
    llm_indices = filtered_df.index[filtered_df.model== llm].tolist()
    pred = filtered_pred.loc[llm_indices]
    trueth = filtered_df.loc[llm_indices]

    accuracy = accuracy_score(trueth.label, pred.Value)
    accuracy_list.append(accuracy)
    print("Accuracy:", accuracy)


# Create DataFrame
df = pd.DataFrame({
    'Model': llm_list,
    'Accuracy': accuracy_list,
})
# Function to create bar plots for a given metric
def plot_metric(data, metric, ax, color,bar_width=0.4):
    data[metric] = data[metric] * 100
    bars = ax.bar(data['Model'], data[metric], color=color,width=bar_width)
    ax.set_title(f'Accuracy on each model for Ghostbuster', fontsize=12)
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(f'{metric}(%)', fontsize=10)
    ax.set_ylim(0, 120)  # Set the y-axis limits to make differences more noticeable
    ax.set_xticklabels(data['Model'], rotation=45)
    # Adding data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom',ha='center')  # va: vertical alignment

# Create figure and axes
fig, ax = plt.subplots(figsize=(5, 4))  # 3 rows, 1 column

# Plot each metric in a separate subplot
plot_metric(df, 'Accuracy', ax, 'indigo')
# plot_metric(df, 'Precision', ax[1], 'green')
# plot_metric(df, 'Recall', ax[2], 'darkorange')

plt.tight_layout()
plt.show()



