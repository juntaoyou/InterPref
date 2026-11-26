import spacy
import textstat
from textblob import TextBlob
from collections import Counter
import numpy as np
import re
import json
# 全局加载 spaCy 模型（英文）
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("请先运行: python -m spacy download en_core_web_sm")

# ----------------------------
# 外部词典加载函数（占位，可替换为实际路径）
# ----------------------------

def load_concreteness_dict(path=None):
    """
    加载抽象性词典（如 Brysbaert Concreteness Norms）
    格式示例: {"word": 3.2, ...}，值越大越具体
    若未提供，返回空字典（跳过抽象性分析）
    """
    if path is None:
        return {}  # 跳过
    # 示例：实际实现可从 CSV/JSON 读取
    concreteness = {}
    # with open(path) as f:
    #     for line in f:
    #         word, score = line.strip().split('\t')
    #         concreteness[word.lower()] = float(score)
    return concreteness

def load_subtlex_freq(path=None):
    """
    加载 SUBTLEX 词频（log10 频率）
    若未提供，返回空字典（跳过罕见度分析）
    """
    if path is None:
        return {}
    # 实际实现类似上文
    return {}

# ----------------------------
# 核心分析函数
# ----------------------------

def semantic_density(doc):
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    content_tokens = [t for t in doc if t.pos_ in content_pos and not t.is_stop and t.is_alpha]
    total_tokens = [t for t in doc if t.is_alpha]
    return len(content_tokens) / len(total_tokens) if total_tokens else 0

def abstractness_ratio(doc, concreteness_dict, threshold=2.5):
    if not concreteness_dict:
        return None
    abstract_count = 0
    valid_words = 0
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ"}:
            word = token.lemma_.lower()
            if word in concreteness_dict:
                valid_words += 1
                if concreteness_dict[word] < threshold:
                    abstractcount += 1
    return abstract_count / valid_words if valid_words > 0 else 0

def sentiment_analysis(text):
    blob = TextBlob(text)
    return {
        "polarity": float(blob.sentiment.polarity),
        "subjectivity": float(blob.sentiment.subjectivity)
    }

def discourse_act(doc):
    sents = list(doc.sents)
    if not sents:
        return "unknown"
    sent = sents[0]
    text = sent.text.strip()
    if text.endswith("?"):
        return "question"
    if len(sent) > 0 and sent[0].pos_ == "VERB" and sent[0].lemma_ not in {"be", "have", "do"}:
        return "command"
    return "statement"

def count_connectives(doc):
    CONNECTIVES = {
        "addition": {"and", "also", "furthermore", "moreover", "additionally"},
        "contrast": {"but", "however", "although", "nevertheless", "yet", "though"},
        "cause": {"because", "since", "therefore", "thus", "hence", "consequently"},
        "condition": {"if", "unless", "provided", "assuming", "given"}
    }
    tokens_lower = {token.text.lower() for token in doc if token.is_alpha}
    counts = {}
    for cat, words in CONNECTIVES.items():
        counts[cat] = len(words & tokens_lower)
    return counts

def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"e$", "", word)
    vowels = len(re.findall(r"[aeiouy]+", word))
    return max(1, vowels)

def total_syllables(doc):
    return sum(count_syllables(token.text) for token in doc if token.is_alpha)

def word_rarity(doc, subtlex_freq, percentile=10):
    if not subtlex_freq:
        return None
    valid_freqs = []
    for token in doc:
        word = token.text.lower()
        if word in subtlex_freq:
            valid_freqs.append(subtlex_freq[word])
    if not valid_freqs:
        return 1.0
    cutoff = np.percentile(valid_freqs, percentile)
    rare_count = sum(1 for f in valid_freqs if f <= cutoff)
    return rare_count / len(valid_freqs)

def dependency_depth(doc):
    depths = []
    for token in doc:
        depth = 0
        curr = token
        while curr.head != curr:
            curr = curr.head
            depth += 1
        depths.append(depth)
    return np.mean(depths) if depths else 0

def has_passive_voice(doc):
    flag = any(token.dep_ == "nsubjpass" for token in doc)
    if flag: return 1
    else: return 0

def collect_statistics(results):
    """
    收集所有分析结果的统计数据
    
    参数:
        results (list): 多个 analyze_sentence 结果的列表
        
    返回:
        dict: 收集的统计数据，用于后续计算平均值
    """
    if not results:
        return {}
    
    # 初始化收集器
    collected = {
        "lexical": {
            "type_token_ratio": [],
            "average_word_length": [],
            "semantic_density": [],
            "word_rarity_ratio": [],
            "abstractness_ratio": [],
            "pos_distribution": Counter(),
        },
        "syntactic": {
            "sentence_length": [],
            "average_dependency_depth": [],
            "has_passive_voice": [],
            "connectives": Counter(),
        },
        "stylistic": {
            "flesch_reading_ease": [],
            "dale_chall_score": [],
            "total_syllables": [],
            "discourse_act": {},
            "sentiment": {
                "polarity": [],
                "subjectivity": []
            },
        }
    }
    
    # 收集所有结果
    for result in results:
        # 收集词汇统计
        for key in ["type_token_ratio", "average_word_length", "semantic_density", "word_rarity_ratio", "abstractness_ratio"]:
            value = result["lexical"][key]
            if value is not None:
                collected["lexical"][key].append(value)
        
        # 收集词性分布
        for pos, count in result["lexical"]["pos_distribution"].items():
            collected["lexical"]["pos_distribution"][pos] += count
        
        # 收集句法统计
        for key in ["sentence_length", "average_dependency_depth"]:
            collected["syntactic"][key].append(result["syntactic"][key])
        collected["syntactic"]["has_passive_voice"].append(result["syntactic"]["has_passive_voice"])
        
        # 收集连接词统计
        for cat, count in result["syntactic"]["connectives"].items():
            collected["syntactic"]["connectives"][cat] += count
        
        # 收集风格统计
        for key in ["flesch_reading_ease", "dale_chall_score", "total_syllables"]:
            collected["stylistic"][key].append(result["stylistic"][key])
        
        # 收集话语行为统计
        discourse = result["stylistic"]["discourse_act"]
        if discourse not in collected["stylistic"]["discourse_act"]:
            collected["stylistic"]["discourse_act"][discourse] = 0
        collected["stylistic"]["discourse_act"][discourse] += 1
        
        # 收集情感统计
        for key in ["polarity", "subjectivity"]:
            collected["stylistic"]["sentiment"][key].append(result["stylistic"]["sentiment"][key])
    
    return collected

def calculate_averages(collected, total_results):
    """
    计算所有统计数据的平均值
    
    参数:
        collected (dict): 收集的统计数据
        total_results (int): 总结果数量
        
    返回:
        dict: 包含所有统计数据平均值的字典
    """
    if not collected:
        return {}
    
    averages = {
        "lexical": {},
        "syntactic": {},
        "stylistic": {},
        "total_documents": total_results
    }
    
    # 计算词汇统计平均值
    for key in ["type_token_ratio", "average_word_length", "semantic_density"]:
        values = collected["lexical"][key]
        if values:
            averages["lexical"][key] = round(np.mean(values), 3)
        else:
            averages["lexical"][key] = 0.0
    
    # 处理可能为 None 的词汇统计
    for key in ["word_rarity_ratio", "abstractness_ratio"]:
        values = collected["lexical"][key]
        if values:
            averages["lexical"][key] = round(np.mean(values), 3)
        else:
            averages["lexical"][key] = None
    
    # 词性分布（保持为计数器）
    averages["lexical"]["pos_distribution"] = dict(collected["lexical"]["pos_distribution"])
    
    # 计算句法统计平均值
    for key in ["sentence_length", "average_dependency_depth"]:
        values = collected["syntactic"][key]
        if values:
            averages["syntactic"][key] = round(np.mean(values), 2)
        else:
            averages["syntactic"][key] = 0.0
    
    # 被动语态比例
    passive_values = collected["syntactic"]["has_passive_voice"]
    if passive_values:
        averages["syntactic"]["has_passive_voice_ratio"] = round(sum(passive_values) / len(passive_values), 3)
    else:
        averages["syntactic"]["has_passive_voice_ratio"] = 0.0
    
    # 连接词统计（保持为计数器）
    averages["syntactic"]["connectives_total"] = dict(collected["syntactic"]["connectives"])
    
    # 计算风格统计平均值
    for key in ["flesch_reading_ease", "dale_chall_score", "total_syllables"]:
        values = collected["stylistic"][key]
        if values:
            averages["stylistic"][key] = round(np.mean(values), 2)
        else:
            averages["stylistic"][key] = 0.0
    
    # 话语行为分布（保持为字典）
    averages["stylistic"]["discourse_act_distribution"] = collected["stylistic"]["discourse_act"]
    
    # 情感统计平均值
    averages["stylistic"]["sentiment"] = {}
    for key in ["polarity", "subjectivity"]:
        values = collected["stylistic"]["sentiment"][key]
        if values:
            averages["stylistic"]["sentiment"][key] = round(np.mean(values), 3)
        else:
            averages["stylistic"]["sentiment"][key] = 0.0
    
    return averages

# ----------------------------
# 主分析函数
# ----------------------------

def analyze_sentence(
    text: str,
    concreteness_path=None,
    subtlex_path=None
):
    """
    对单个句子进行全面语言学分析。
    
    参数:
        text (str): 输入句子
        concreteness_path (str, optional): 抽象性词典路径
        subtlex_path (str, optional): 词频词典路径
    
    返回:
        dict: 包含 lexical / syntactic / stylistic 三大类指标
    """
    doc = nlp(text)
    
    # 加载外部词典（按需）
    concreteness_dict = load_concreteness_dict(concreteness_path)
    subtlex_freq = load_subtlex_freq(subtlex_path)
    
    # 词汇分析
    tokens_alpha = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in tokens_alpha]
    ttr = len(set(lemmas)) / len(lemmas) if lemmas else 0
    avg_word_len = np.mean([len(t) for t in tokens_alpha]) if tokens_alpha else 0
    pos_counts = dict(Counter([t.pos_ for t in tokens_alpha]))
    
    # 句法分析
    sent_len = len(tokens_alpha)
    avg_depth = dependency_depth(doc)
    passive = has_passive_voice(doc)
    connectives = count_connectives(doc)
    
    # 风格与语义
    flesch = textstat.flesch_reading_ease(text)
    dale_chall = textstat.dale_chall_readability_score(text)
    syllables = total_syllables(doc)
    discourse = discourse_act(doc)
    sentiment = sentiment_analysis(text)
    sem_density = semantic_density(doc)
    abstractness = abstractness_ratio(doc, concreteness_dict)
    rarity = word_rarity(doc, subtlex_freq)

    return {
        "lexical": {
            "type_token_ratio": round(ttr, 3),
            "average_word_length": round(avg_word_len, 2),
            "semantic_density": round(sem_density, 3),
            "word_rarity_ratio": round(rarity, 3) if rarity is not None else None,
            "abstractness_ratio": round(abstractness, 3) if abstractness is not None else None,
            "pos_distribution": pos_counts,
        },
        "syntactic": {
            "sentence_length": sent_len,
            "average_dependency_depth": round(avg_depth, 2),
            "has_passive_voice": passive,
            "connectives": connectives,
        },
        "stylistic": {
            "flesch_reading_ease": round(flesch, 2),
            "dale_chall_score": round(dale_chall, 2),
            "total_syllables": syllables,
            "discourse_act": discourse,
            "sentiment": sentiment,
        }
    }

# 示例
def get_data(filepath):
    datas = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                data = json.loads(line)
                datas.append(data)
    return datas

# 主执行部分
def main():
    # 文件路径
    file_path1 = '<REPO_ROOT>/detect_with_two/eval/qwen_alpaca/Nsteer/Expertise_high_0.9.json'
    # file_path2 = '<REPO_ROOT>/detect_with_two/eval/qwen_alpaca/P1/None.json'
    
    # 读取数据
    data1 = get_data(file_path1)
    # data2 = get_data(file_path2)
    
    # 确保两个文件有相同数量的数据
    # if len(data1) != len(data2):
    #     print(f"警告：两个文件的数据行数不同 - 文件1: {len(data1)}行, 文件2: {len(data2)}行")
    #     min_len = min(len(data1), len(data2))
    #     data1 = data1[:min_len]
    #     data2 = data2[:min_len]
    
    # print(f"总共处理 {len(data1)} 个样本")
    
    # 处理数据并收集结果
    results1 = []
    # results2 = []
    
    for d1 in tqdm(data1, desc="处理样本"):
        text1 = d1['answer']
        
        result1 = analyze_sentence(text1)

        results1.append(result1)
    
    # 收集统计数据
    collected1 = collect_statistics(results1)
    # collected2 = collect_statistics(results2)
    
    # 计算平均值
    averages1 = calculate_averages(collected1, len(results1))
    # averages2 = calculate_averages(collected2, len(results2))
    
    # 打印结果
    print("\n=" * 70)
    print("文件1统计结果平均值 (<REPO_ROOT>/detect_with_two/eval/qwen_alpaca/Nsteer/Expertise_high_0.9.json)")
    print("=" * 70)
    with open('Nsteer_Expertise_high_0.9.json', 'w', encoding='utf-8') as f:
        json.dump(averages1, f, ensure_ascii=False, indent=2)
    
    # print("\n=" * 70)
    # print("文件2统计结果平均值 (<REPO_ROOT>/detect_with_two/eval/qwen_alpaca/P1/None.json)")
    # print("=" * 70)
    # with open('P1_None.json', 'w', encoding='utf-8') as f:
    #     json.dump(averages2, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
