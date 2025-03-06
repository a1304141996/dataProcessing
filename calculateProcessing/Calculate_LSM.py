# 突然发现储存每个发言者的那个文件忘记加入有效对话分开的逻辑了，导致有效对话段没有被分开
import os
import text_processing_utils
import config


def calculate_lsm(value_a, value_b):
    lsm_score = 1 - abs(value_a - value_b) / (value_a + value_b + 0.0001)
    return lsm_score


# 执行处理文件的函数
def process_files(folder_path, function_words_file):
    word_counts = []
    func_word_counts = []
    total_func_word_counts = []
    matched_function_words = []

    function_words = text_processing_utils.load_function_words(function_words_file)

    fwc_values = []  # 存储所有C对话的FW值
    fwt_values = []  # 存储所有T对话的FW值
    lsm_values = []  # 初始化LSM值列表

    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        full_file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_file_path):
            paragraphs = text_processing_utils.split_paragraphs(full_file_path)

            # 处理每个段落
            for paragraph in paragraphs:
                total_func_word_count = 0
                total_word_count = 0
                # 分割每一行对话
                lines = paragraph.splitlines()
                for line in lines:
                    if ':' not in line:
                        continue
                    # 去除发送者标识，只保留对话内容
                    parts = line.split(': ', 1)
                    if len(parts) == 2:
                        statement = parts[1].strip()
                        cleaned_statement = text_processing_utils.clean_and_merge(statement)
                        func_count, word_count, count, func_list = text_processing_utils.process_sentence(
                            cleaned_statement,
                            function_words)
                        total_func_word_count += func_count
                        total_word_count += word_count

                        word_counts.append(total_word_count)
                        func_word_counts.append(total_func_word_count)
                        total_func_word_counts.append(func_count)
                        matched_function_words.extend(func_list)

                # print(matched_function_words)
                # print(word_counts)
                # print(len(matched_function_words))  # 匹配到的功能词个数，不等于功能词个数，因为有些功能词会重复出现
                # print(total_word_count)  # 一个有效对话词总数
                # print(total_func_word_count)  # 一个有效对话段功能词总个数
                # 如果总词数为0则跳过该段落
                if total_word_count == 0:
                    continue

                # 计算当前段落的FW值
                fw_value = text_processing_utils.calculate_fw_values(total_func_word_count, total_word_count)

                # 根据文件名区分并存储C对话和T对话的FW值
                if 'C' in file_name:
                    fwc_values.append(fw_value)
                elif 'T' in file_name:
                    fwt_values.append(fw_value)

            # 对当前文件计算LSM值
            for fwc, fwt in zip(fwc_values, fwt_values):
                lsm = calculate_lsm(fwc, fwt)
                lsm_values.append(round(lsm, 2))

    return lsm_values


def LSM(directory_path, function_words_file):
    results = process_files(directory_path, function_words_file)
    return results

# file_path = r'D:\毕业论文\对话数据\合并后的对话数据\test\对话'
# lsm_values = LSM(file_path, config.FUNCTION_WORDS_FILE)
# print(lsm_values)
