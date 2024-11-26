import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import language_tool_python
from torch.utils.data import Dataset, DataLoader
import chardet
from sklearn.metrics import accuracy_score, recall_score
import re

# 1. 加载模型和分词器
model_load_path = 'd:/jupyter/transformer/grammar_correction_model1'
tokenizer = T5Tokenizer.from_pretrained(model_load_path)
model = T5ForConditionalGeneration.from_pretrained(model_load_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载第二个模型
model_name = 'd:/jupyter/transformer/grammar_correction_model2'
tokenizer2 = T5Tokenizer.from_pretrained(model_name)
model2 = T5ForConditionalGeneration.from_pretrained(model_name)
model2.to(device)

# 2. 定义错误类型映射
error_type_mapping = {
    'ADMIT_ENJOY_VB': 0, 'AGREEMENT_SENT_START': 1, 'APOS_SPACE_CONTRACTION': 2, 'A_NNS': 3, 
    'BEEN_PART_AGREEMENT': 4, 'CD_NN': 5, 'COMMA_COMPOUND_SENTENCE': 6, 'COMMA_COMPOUND_SENTENCE_2': 7,
    'COMMA_PARENTHESIS_WHITESPACE': 8, 'DELETE_SPACE': 9, 'ENGLISH_WORD_REPEAT_RULE': 10, 'EN_A_VS_AN': 11,
    'EN_COMPOUNDS_PART_TIME': 12, 'EN_CONTRACTION_SPELLING': 13, 'EN_SPECIFIC_CASE': 14, 'EVERYDAY_EVERY_DAY': 15,
    'EXTREME_ADJECTIVES': 16, 'FIRST_OF_ALL': 17, 'HAVE_PART_AGREEMENT': 18, 'HE_VERB_AGR': 19, 'IT_VBZ': 20,
    'JAPAN': 21, 'MANY_NN': 22, 'MD_BASEFORM': 23, 'MISSING_COMMA_AFTER_INTRODUCTORY_PHRASE': 24,
    'MISSING_TO_BEFORE_A_VERB': 25, 'MORFOLOGIK_RULE_EN_US': 26, 'PRP_JJ': 27, 'PRP_PAST_PART': 28, 'PRP_VBG': 29,
    'RB_RB_COMMA': 30, 'SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA': 31, 'SPACE_BETWEEN_NUMBER_AND_WORD': 32,
    'THIS_NNS': 33, 'UPPERCASE_SENTENCE_START': 34, 'WANNA': 35, 'WRONG_APOSTROPHE': 36, 'none': 37
}

# 3. 编写缩写转全写的函数
CONTRACTIONS = {
    "isn't": "is not",
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "we're": "we are",
    "you've": "you have",
    "you'll": "you will",
    "i'm": "i am"
}

def expand_contractions(text):
    pattern = re.compile(r"\b(" + "|".join(CONTRACTIONS.keys()) + r")\b")
    return pattern.sub(lambda x: CONTRACTIONS[x.group()], text)

def standardize_contractions(texts):
    return [expand_contractions(text) for text in texts]

# 4. Grammar correction using T5 model with error type input
def identify_error_type(sentence, tool, error_type_mapping):
    matches = tool.check(sentence)
    if matches:
        rule_id = matches[0].ruleId
        error_type_idx = error_type_mapping.get(rule_id, error_type_mapping['none'])
    else:
        error_type_idx = error_type_mapping['none']
    return error_type_idx

def correct_grammar_with_type(text, tool, error_type_mapping):
    error_type_idx = identify_error_type(text, tool, error_type_mapping)
    input_text = f"grammar: {error_type_idx} {text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected_text = corrected_text.replace(f"grammar: {error_type_idx} ", "")
    return corrected_text

# 5. Grammar correction using the second model
class GrammarCorrectionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, encoding='utf-8'):
        with open(csv_file, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        self.data = pd.read_csv(csv_file, encoding=encoding)
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
            'labels': labels[0],
            'incorrect': incorrect,
            'correct': correct
        }

# 6. 手动计算 F0.5 分数
def f05_score(precision, recall):
    beta = 0.5
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

# 7. 评估模型并获取预测与实际标签
def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            incorrect_sentences = batch['incorrect']
            correct_sentences = batch['correct']
            outputs = model.generate(input_ids)
            predicted_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            correct_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            # 标准化缩写词
            standardized_predictions = standardize_contractions(predicted_texts)
            standardized_labels = standardize_contractions(correct_texts)
            
            all_labels.extend(standardized_labels)
            all_predictions.extend(standardized_predictions)
    
    true_labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions, average='macro')
    
    # 计算 precision
    precision = recall_score(true_labels, predictions, average='macro', zero_division=1)
    
    # 计算 F0.5 分数
    f05 = f05_score(precision, recall)
    
    return accuracy, recall, f05, all_labels, all_predictions

# 8. 加载验证集数据
val_data_path = 'd:/jupyter/transformer/testdata/SpellingErrors_Test2.csv'  # 请替换为你的文件路径
val_dataset = GrammarCorrectionDataset(val_data_path, tokenizer2)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 9. 评估两个模型
accuracy_1, recall_1, f05_1, all_labels1, all_predictions1 = evaluate_model(model, val_dataloader, tokenizer, device)
accuracy_2, recall_2, f05_2, all_labels2, all_predictions2 = evaluate_model(model2, val_dataloader, tokenizer2, device)

# 10. 打印准确率、召回率和 F0.5 分数
print(f"Model 1 - Accuracy: {accuracy_1:.4f}, Recall: {recall_1:.4f}, F0.5: {f05_1:.4f}")
print(f"Model 2 - Accuracy: {accuracy_2:.4f}, Recall: {recall_2:.4f}, F0.5: {f05_2:.4f}")

# 11. 比较两个方法的输出并显示差异
