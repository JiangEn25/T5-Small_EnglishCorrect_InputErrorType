from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import language_tool_python

# 1. 加载模型和分词器
model_load_path = 'd:/jupyter/transformer/grammar_correction_model1'

# 加载分词器和模型
tokenizer = T5Tokenizer.from_pretrained(model_load_path)
model = T5ForConditionalGeneration.from_pretrained(model_load_path)

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 定义错误类型映射
error_type_mapping = {
    'ADMIT_ENJOY_VB': 0,
    'AGREEMENT_SENT_START': 1,
    'APOS_SPACE_CONTRACTION': 2,
    'A_NNS': 3,
    'BEEN_PART_AGREEMENT': 4,
    'CD_NN': 5,
    'COMMA_COMPOUND_SENTENCE': 6,
    'COMMA_COMPOUND_SENTENCE_2': 7,
    'COMMA_PARENTHESIS_WHITESPACE': 8,
    'DELETE_SPACE': 9,
    'ENGLISH_WORD_REPEAT_RULE': 10,
    'EN_A_VS_AN': 11,
    'EN_COMPOUNDS_PART_TIME': 12,
    'EN_CONTRACTION_SPELLING': 13,
    'EN_SPECIFIC_CASE': 14,
    'EVERYDAY_EVERY_DAY': 15,
    'EXTREME_ADJECTIVES': 16,
    'FIRST_OF_ALL': 17,
    'HAVE_PART_AGREEMENT': 18,
    'HE_VERB_AGR': 19,
    'IT_VBZ': 20,
    'JAPAN': 21,
    'MANY_NN': 22,
    'MD_BASEFORM': 23,
    'MISSING_COMMA_AFTER_INTRODUCTORY_PHRASE': 24,
    'MISSING_TO_BEFORE_A_VERB': 25,
    'MORFOLOGIK_RULE_EN_US': 26,
    'PRP_JJ': 27,
    'PRP_PAST_PART': 28,
    'PRP_VBG': 29,
    'RB_RB_COMMA': 30,
    'SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA': 31,
    'SPACE_BETWEEN_NUMBER_AND_WORD': 32,
    'THIS_NNS': 33,
    'UPPERCASE_SENTENCE_START': 34,
    'WANNA': 35,
    'WRONG_APOSTROPHE': 36,
    'none': 37
}

def identify_error_type(sentence, tool, error_type_mapping):
    # 使用 language_tool_python 识别错误
    matches = tool.check(sentence)
    
    # 获取第一个错误的规则ID
    if matches:
        rule_id = matches[0].ruleId
        # 尝试将规则ID映射到错误类型索引
        error_type_idx = error_type_mapping.get(rule_id, error_type_mapping['none'])
    else:
        error_type_idx = error_type_mapping['none']
    
    return error_type_idx

def correct_grammar(text, tool, error_type_mapping):
    # 识别错误类型索引
    error_type_idx = identify_error_type(text, tool, error_type_mapping)
    
    # 将错误类型索引拼接到错误句子中
    input_text = f"grammar: {error_type_idx} {text}"
    
    # 对输入文本进行编码
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # 生成纠正后的文本
    with torch.no_grad():
        outputs = model.generate(input_ids)
    
    # 解码生成的输出
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 去掉错误类型索引
    corrected_text = corrected_text.replace(f"grammar: {error_type_idx} ", "")
    
    return corrected_text

if __name__ == "__main__":
    # 初始化 language_tool_python 工具
    tool = language_tool_python.LanguageTool('en-US')
    
    while True:
        # 用户输入句子
        user_input = input("请输入要纠正的句子（输入'exit'退出）: ")
        
        # 检查是否退出
        if user_input.lower() == 'exit':
            print("退出程序。")
            break
        
        # 生成纠正后的句子
        corrected_text = correct_grammar(user_input, tool, error_type_mapping)
        
        # 打印结果
        print(f"输入: {user_input}")
        print(f"纠正后: {corrected_text}")
        print("-" * 50)