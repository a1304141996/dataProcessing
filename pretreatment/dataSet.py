import os
import pandas as pd
import json
from glob import glob


def process_excel_file(file_path, output_dir):
    """
    处理单个Excel文件并转换为JSON格式
    :param file_path: str, xlsx文件的路径
    :param output_dir: str, 输出目录
    :return: None
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 提取全局指标（LSM 和 M rLSM）
    global_metrics = {
        "LSM": df.loc[0, 'LSM'],  # 假设 LSM 在列 F
        "M_rLSM": df.loc[0, 'M rLSM']  # 假设 M_rLSM 在列 E
    }

    # 准备对话数据
    conversations = []
    for i in range(len(df)):
        conversation = {
            "speaker": df.loc[i, 'Speaker'],  # 假设 'Speaker' 在列 A
            "statement": df.loc[i, 'Statement'],  # 假设 'Statement' 在列 B
            "function_words_percentage": df.loc[i, 'Function Words (%)'],  # 假设 'Function Words (%)' 在列 C
            "rLSM": 0 if pd.isna(df.loc[i, 'rLSM']) else df.loc[i, 'rLSM']
        }
        conversations.append(conversation)

    # 用户数据
    user_data = {
        "label": 0,  # 假设整个用户的标签，1表示愿意付费
        "conversations": conversations
    }

    # 最终 JSON 结构
    data = {
        "global_metrics": global_metrics,
        "user_data": user_data
    }

    # 将文件名与路径分开，保留文件名并更改扩展名为.json
    output_filename = os.path.basename(file_path).replace('.xlsx', '.json')
    output_path = os.path.join(output_dir, output_filename)

    # 转换为 JSON 并保存到指定路径
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Processed {file_path} -> {output_path}")


def process_all_files(input_dir, output_dir):
    """
    处理给定目录下的所有xlsx文件并将其转换为JSON格式
    :param input_dir: str, 输入xlsx文件目录
    :param output_dir: str, 输出json文件目录
    :return: None
    """
    # 获取指定目录下所有的xlsx文件
    file_paths = glob(os.path.join(input_dir, '*.xlsx'))

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每一个文件
    for file_path in file_paths:
        process_excel_file(file_path, output_dir)


# 示例调用
input_directory = r"D:\毕业论文\对话数据\xlsx文件夹\不付费"  # 输入目录，包含xlsx文件
output_directory = r"D:\毕业论文\对话数据\json文件夹\不付费"  # 输出目录，保存处理后的json文件

process_all_files(input_directory, output_directory)
