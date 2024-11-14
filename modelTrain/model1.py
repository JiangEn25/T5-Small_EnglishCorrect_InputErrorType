import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class GrammarCorrectionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        incorrect = self.data.iloc[idx]['incorrect']
        correct = self.data.iloc[idx]['correct']
        error_type_idx = self.data.iloc[idx]['error_type_idx']
        
        # 将错误类型索引拼接到错误句子中
        input_text = f"grammar: {error_type_idx} {incorrect}"
        
        # 编码输入句子
        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码正确句子
        label_encoding = self.tokenizer.encode_plus(
            correct,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': label_encoding['input_ids'].squeeze(0)
        }
    
from transformers import T5ForConditionalGeneration

# 使用标准的 T5ForConditionalGeneration 模型
model = T5ForConditionalGeneration.from_pretrained('t5-small')

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm.auto import tqdm  # 导入 tqdm

# 读取数据
data_path = 'd:/jupyter/transformer/encoded_labeled_data2.csv'
data = pd.read_csv(data_path)
num_error_types = data['error_type_idx'].nunique()
print(f"Number of unique error types: {num_error_types}")

# 初始化模型和分词器
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载数据
dataset = GrammarCorrectionDataset(data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=1e-4)

# 训练模型
model.train()
for epoch in range(1):  # 训练1个epoch
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):  # 使用 tqdm 显示进度条
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        except Exception as e:
            print(f"Error in batch {i + 1}: {e}")
            break
        
        # 打印每10个批次的损失值
        if (i + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}')

# 保存模型
model.save_pretrained('d:/jupyter/transformer/grammar_correction_model1')
tokenizer.save_pretrained('d:/jupyter/transformer/grammar_correction_model1')

