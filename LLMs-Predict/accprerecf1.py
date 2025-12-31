import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

output_path = 'result/output_SST_gpt35.csv'


df = pd.read_csv(output_path)
true_labels = df['true_label']
pred_labels = df['pred_label']

true_labels = true_labels.astype(int)
pred_labels = pred_labels.astype(int)

acc = accuracy_score(true_labels,pred_labels)
pre = precision_score(true_labels,pred_labels,average='macro')
rec = recall_score(true_labels,pred_labels,average='macro')
f1 = f1_score(true_labels,pred_labels,average='macro')

print(f'acc:{acc:.4f}')
print(f'pre:{pre:.4f}')
print(f'rec:{rec:.4f}')
print(f'f1:{f1:.4f}')