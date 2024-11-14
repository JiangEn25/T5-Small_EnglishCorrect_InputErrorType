import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

# 读取数据
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
        
        input_ids = self.tokenizer.encode("grammar: " + incorrect, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(correct, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        
        return {
            'input_ids': input_ids[0],
            'labels': labels[0]
        }

# 初始化模型和分词器
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 检查模型的输入和输出形状
dummy_input = tokenizer.encode("grammar: This is a test sentence.", return_tensors='pt')
dummy_output = model.generate(dummy_input)
print(f"Dummy input shape: {dummy_input.shape}, Dummy output shape: {dummy_output.shape}")

# 加载数据
data_path = 'd:/jupyter/transformer/data.csv'
dataset = GrammarCorrectionDataset(data_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=1e-4)

# 训练模型
model.train()
for epoch in range(1):  # 训练3个epoch
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        print(f"Batch {i + 1} input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
        
        optimizer.zero_grad()
        try:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Error in batch {i + 1}: {e}")
            break
        
        # 打印每10个批次的损失值
        if (i + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')
    
    print(f'Epoch {epoch + 1}, Final Loss: {loss.item()}')

# 保存模型
model.save_pretrained('d:/jupyter/transformer/grammar_correction_model2')
tokenizer.save_pretrained('d:/jupyter/transformer/grammar_correction_model2')
