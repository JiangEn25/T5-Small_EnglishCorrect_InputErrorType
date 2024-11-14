import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, fbeta_score
import chardet

# 读取数据
class GrammarCorrectionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, encoding='utf-8'):
        # 自动检测文件编码
        with open(csv_file, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        
        # 读取 CSV 文件
        self.data = pd.read_csv(csv_file, encoding=encoding)
        
        # 打印 CSV 文件的前几行，以便确认列名和内容
        print("CSV file content:")
        print(self.data.head())
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        incorrect = self.data.iloc[idx]['incorrect']
        correct = self.data.iloc[idx]['correct']
        
        input_ids = self.tokenizer.encode("grammar: " + incorrect, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(correct, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        
        return {
            'input_ids': input_ids[0],
            'labels': labels[0]
        }

# 加载模型和分词器
model_name = 'd:/jupyter/transformer/grammar_correction_model2'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载验证集
val_data_path = 'd:/jupyter/transformer/val_data.csv'
val_dataset = GrammarCorrectionDataset(val_data_path, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义评估函数
def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model.generate(input_ids)
            predicted_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            correct_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            all_labels.extend(correct_texts)
            all_predictions.extend(predicted_texts)

    return all_labels, all_predictions

# 评估模型
all_labels, all_predictions = evaluate_model(model, val_dataloader, tokenizer, device)

# 计算评估指标
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f0_5 = fbeta_score(all_labels, all_predictions, beta=0.5, average='weighted')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F0.5: {f0_5}')