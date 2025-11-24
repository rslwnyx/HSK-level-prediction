import re
import jieba
import pandas as pd
import numpy as np

def clamp_level(level):
    return min(level, 6)


def clean_chinese_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    return text.strip()

def load_hsk_data(file_path):
    
    df = pd.read_csv(file_path, header=None)
    df.columns = ['content', 'level', 'type']
    
    word_dict = {}
    grammar_patterns = []

    for _, row in df.iterrows():
        content = str(row['content']).strip()
        try:
            level = int(row['level'])
        except:
            continue
            
        dtype = str(row['type']).strip().lower()

        if dtype == 'word':
            word_dict[content] = level
        
        elif dtype == 'grammar':
            safe_pattern = content.replace('(', '\\(').replace(')', '\\)')
            safe_pattern = safe_pattern.replace('[', '\\[').replace(']', '\\]')
            safe_pattern = safe_pattern.replace('?', '\\?')
            safe_pattern = safe_pattern.replace('+', '\\+')
            safe_pattern = safe_pattern.replace('*', '\\*')
            safe_pattern = safe_pattern.replace('...', '.+')

            regex_pattern = re.sub(r'\s*[A-Z]+\s*', '.+', safe_pattern)

            grammar_patterns.append((regex_pattern, clamp_level(level)))

    return word_dict, grammar_patterns

def extract_hsk_features(text, word_dict, grammar_patterns):
    
    text = clean_chinese_text(text)
    words = jieba.lcut(text)
    total_words = len(words) if len(words) > 0 else 1
    
    #Counter
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for w in words:
        lvl = word_dict.get(w, 0)
        if 1 <= lvl <= 6:
            level_counts[lvl] += 1
            
    for pattern, lvl in grammar_patterns:
        if re.search(pattern, text):
            if 1 <= lvl <= 6:
                level_counts[lvl] += 2 
    

    features = [level_counts[i] / total_words for i in range(1, 7)]
    features.append(total_words)
    
    weighted_score = 0
    for lvl in range(1, 7):
        weight = lvl * lvl 
        weighted_score += level_counts[lvl] * weight
        
    features.append(weighted_score / total_words)

    return features

def hsk_tokenizer(text):
    return jieba.lcut(text)