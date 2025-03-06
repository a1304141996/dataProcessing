# 对于每个文件，创建一个新的文件夹，文件夹以原文件名为基础，且输出到一个总的文件夹目录中，该目录包含所有文件生成的文件夹
# 读取文件内容，区分咨询师与咨询者对话，并分别写入不同的文件夹中
# 此部分用于计算LSM
import os


# 获取咨询师与咨询者id
def extract_ids_from_filename(file_name):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    counselor_id, consultee_id = base_name.split('_')
    return counselor_id, consultee_id


def process_dialogues(file_path, counselor_id, consultee_id, target_folder):
    counselor_prefix = f'{counselor_id} -> {consultee_id}'
    conseltee_prefix = f'{consultee_id} -> {counselor_id}'

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        counselor_dialogues = [line for line in lines if counselor_prefix in line]  # 存储咨询师的全部对话
        consultee_dialogues = [line for line in lines if conseltee_prefix in line]  # 存储咨询者的全部对话

    # 创建目标文件夹
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dialogues_folder = os.path.join(target_folder, base_name)
    os.makedirs(dialogues_folder, exist_ok=True)

    # 写入咨询师与咨询者对话
    counselor_file = os.path.join(dialogues_folder, f'{counselor_id}_dialogue.txt')
    consultee_file = os.path.join(dialogues_folder, f'{consultee_id}_dialogue.txt')

    with open(counselor_file, 'w', encoding='utf-8') as cf:
        for dialogue in counselor_dialogues:
            cf.writelines(dialogue.strip() + '\n')

    with open(consultee_file, 'w', encoding='utf-8') as tf:
        for dialogue in consultee_dialogues:
            tf.writelines(dialogue.strip() + '\n')


def process_files(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(source_folder, file_name)
            counselor_id, consultee_id = extract_ids_from_filename(file_path)
            process_dialogues(file_path, counselor_id, consultee_id, target_folder)


def main():
    source_folder = r'D:\毕业论文\对话数据\付费'
    target_folder = r'D:\毕业论文\对话数据\分开的对话\付费'
    process_files(source_folder, target_folder)


if __name__ == '__main__':
    main()
