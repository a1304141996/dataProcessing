"""原功能词典的功能词还附带词性标识，本部分代码用于去掉这种标识，只保留功能词部分"""

import re
import os

# 正则表达式法
'''
def extract_function_words(file_path):
    function_words = []
    not_match_words = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match_words = re.findall(r'^[\u4e00-\u9fa5\w]+', line)

            if match_words:
                function_words.extend(match_words)
            else:
                not_match_words.append(line.strip())

    return function_words, not_match_words


def save_function_dictionary(function_dictionary, file_path):
    # 确保文件存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file:
        for word in function_dictionary:
            file.write(word + '\n')


source_file_path = r'D:\LIWC词典\功能词.txt'
target_file_path = r'D:\LIWC词典\功能词典.txt'

fundcation_word, words = extract_function_words(source_file_path)
print(fundcation_word)
print(words)
save_function_dictionary(fundcation_word, target_file_path)
'''


# 切片法
def extract_function_words(file_path):
    function_words = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            patrs = line.strip().split()
            if patrs:
                function_words.append(patrs[0])
            else:
                continue
    return function_words


def save_function_dictionary(function_dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for word in function_dictionary:
            file.write(word + '\n')


source_file_path = r'D:\数据集\LIWC中文词典\功能词.txt'
target_file_path = r'D:\数据集\LIWC中文词典\功能词典.txt'

function_words = extract_function_words(source_file_path)
save_function_dictionary(function_words, target_file_path)
