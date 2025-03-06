"""
将分散的对话合并为一人一句的形式
"""
import os.path


def merge_dialogue(file_contents):
    lines = file_contents.splitlines()
    # 初始化变量
    current_speaker = None
    current_message = []
    last_receiver = None  # 保存最后一次接收者信息

    results = []

    # 处理每一行数据
    for line in lines:
        parts = line.strip().split(': ', 1)  # ': '一定要是这种形式，冒号后面跟个空格，否则会匹配到时间后面的冒号
        if len(parts) < 2:
            continue

        speaker, message = parts[0].split(' -> ')[0].split()[-1], parts[1].strip()

        receiver = parts[0].split(' -> ')[1]

        if speaker == current_speaker:
            # 相同则合并对话
            current_message.append(message)
        else:
            # 不同则输出前一条消息
            if current_speaker is not None:
                results.append(f"{current_speaker} -> {last_receiver}: {'  '.join(current_message)}")
            # 重置对话记录
            current_speaker = speaker
            current_message = [message]
            last_receiver = receiver

    if current_speaker is not None:
        results.append(f"{current_speaker} -> {last_receiver}: {' '.join(current_message)}")

    return '\n'.join(results)


def process_files(source_dir, target_dir):
    # 确保目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历文件夹里的全部文件
    for filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, filename)
        target_file_path = os.path.join(target_dir, filename)

        # 阅读文件中的内容
        if os.path.isfile(source_file_path):
            with open(source_file_path, 'r', encoding='utf-8') as file:
                file_contents = file.read()  # 这里只能用read，将全部文本读取为字符串，然后按照valid分割为不同有效对话段

            segments = file_contents.split('valid Segment\n')  # 分割为不同有效对话段

            merged_segments = []

            for i, segment in enumerate(segments):
                if segment.strip():
                    merged_content = merge_dialogue(segment)

                    merged_segments.append(merged_content + '\n\n\n')

            # 将合并后的对话写入文件
            with open(target_file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(merged_segments))


source_dir = r'D:\毕业论文\对话数据\付费'
target_dir = r'D:\毕业论文\对话数据\合并后的对话数据\付费'

process_files(source_dir, target_dir)
