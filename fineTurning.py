import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import random
from sklearn.metrics import classification_report
import wandb
import time

config = wandb.config

#  wandb初始化
wandb.init(
    project="FineTuning_Qwen2-1.5b",
    name=time.strftime('%m%d%H%M%S'),
    config={
        "batch_size": 16,
        "epochs": 5,
        "learning_rate": 1e-4,
        "model_name": "Qwen2-1.5b"
    }
)


def load_conversation_data(data_path):
    conversations = []
    labels = []
    lsm_values = []
    m_rlsm_values = []

    # 遍历路径下的所有json文件
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(data_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 提取global_metrics中的 LSM 和 M_rLSM
                global_metrics = data.get("global_metrics", {})
                lsm = global_metrics.get('LSM', 0.0)
                m_rlsm = global_metrics.get('M_rLSM', 0.0)
                lsm_values.append(lsm)
                m_rlsm_values.append(m_rlsm)

                # 提取 user_data 中的 label
                user_data = data.get("user_data", {})
                label = user_data.get('label', 0)
                labels.append(label)

                # 提取对话数据 conversations
                conversation_data = []
                for conv in user_data.get("conversations", []):
                    statement = conv.get('statement', '')
                    function_words_percentage = conv.get('function_words_percentage', 0.0)
                    rlsm = conv.get('rLSM', None)  # 处理 rLSM 值，可能为空
                    conversation_data.append({
                        'statement': statement,
                        'function_words_percentage': function_words_percentage,
                        'rLSM': rlsm
                    })

                # 将提取的对话数据加入 conversations 列表
                conversations.append(conversation_data)

    return conversations, labels, lsm_values, m_rlsm_values


class CustomDataset(Dataset):
    def __init__(self, conversations, labels, lsm_values, m_rlsm_values, tokenizer, max_length=40,
                 max_sentences=20):  # max_length与max_sentences的设置需要根据实际情况进行调整,如何确定这两个值？
        # max_length表示每个句子分词后所包含的 token 的最大数量
        self.conversations = conversations
        self.labels = labels
        self.lsm_values = lsm_values
        self.m_rlsm_values = m_rlsm_values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_sentences = max_sentences  # 设置最大句子数

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        try:
            conv = self.conversations[idx]
            label = self.labels[idx]
            lsm = self.lsm_values[idx]
            m_rlsm = self.m_rlsm_values[idx]

            # 截断或填充句子数量
            # 如果对话集中的句子数量超过 max_sentences，则截断；如果少于，则填充。
            if len(conv) > self.max_sentences:
                conv = conv[:self.max_sentences]
            else:
                # 使用空句子进行填充，确保总共的句子数量为 max_sentences
                while len(conv) < self.max_sentences:
                    conv.append({'statement': '', 'function_words_percentage': 0.0, 'rLSM': 0.0})

            # 确保所有语句都是字符串类型，并处理可能的 None 或空值
            statements = [str(c['statement']) if c['statement'] else "" for c in conv]
            function_words_percentages = [c['function_words_percentage'] for c in conv]
            rlsm_values = [c['rLSM'] if c['rLSM'] is not None else 0 for c in conv]  # 处理最后一句没有rLSM的情况

            # 分别对每个语句进行编码，然后手动堆叠结果
            all_input_ids = []
            all_attention_masks = []

            for statement in statements:
                inputs = self.tokenizer(
                    statement,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,  # 当文本长度超过了设定的最大长度（max_length）时，自动将超出的部分截断，只保留前面指定长度的内容
                    return_tensors="pt"  # 指定返回的数据类型为PyTorch的张量（torch.Tensor），方便后续直接送入PyTorch模型中使用
                )
                all_input_ids.append(inputs['input_ids'].squeeze(0))
                all_attention_masks.append(inputs['attention_mask'].squeeze(0))

            # 堆叠所有的 tensors
            input_ids = torch.stack(all_input_ids)  # (max_sentences, max_length)
            attention_mask = torch.stack(all_attention_masks)  # (max_sentences, max_length)

            # 将 function_words_percentage 和 rLSM 作为逐句特征
            return {
                'input_ids': input_ids,  # (max_sentences, max_length)
                'attention_mask': attention_mask,  # (max_sentences, max_length)
                'function_words_percentage': torch.tensor(function_words_percentages, dtype=torch.float),
                # (max_sentences,)
                'rLSM': torch.tensor(rlsm_values, dtype=torch.float),  # (max_sentences,)
                'LSM': torch.tensor(lsm, dtype=torch.float),  # 单一数
                'M_rLSM': torch.tensor(m_rlsm, dtype=torch.float),  # 单一数值
                'labels': torch.tensor(label, dtype=torch.long)  # 单一数值
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            print(f"Conversation content: {self.conversations[idx]}")
            raise


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # 确定 hidden_size 和 embed_dim
        hidden_size = self.model.config.hidden_size
        embed_dim = hidden_size + 2

        # 动态选择 num_heads
        num_heads = max(
            [i for i in range(1, embed_dim + 1) if embed_dim % i == 0 and i % 2 == 0 and i <= embed_dim // 2])

        # 使用 TransformerEncoder 替代 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.aggregation_layer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1  # 使用单层TransformerEncoder，具体使用几层需要根据实际情况进行调整，用多少层效果最好？
        )

        # 分类层保持不变
        self.fc = nn.Linear(embed_dim + 2, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM):
        batch_size, max_sentences, max_length = input_ids.shape
        hidden_size = self.model.config.hidden_size

        # 重塑输入以进行批量处理
        input_ids_reshaped = input_ids.view(-1, max_length)
        attention_mask_reshaped = attention_mask.view(-1, max_length)

        # 一次性处理所有句子
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_reshaped,
                attention_mask=attention_mask_reshaped
            )

        '''这里的特征拼接是用cls特征拼接第一个fw和rlsm，应该是将每一个输出向量与自身的特征拼接后，再经过一个Transformer层，然后取其cls特征'''
        # 这里是取第一个fw和rlsm进行拼接么，如果是，还需要对这两个值进行截断或填充
        # 重塑输出为原始维度
        cls_tokens = outputs.last_hidden_state[:, 0, :].view(batch_size, max_sentences, hidden_size)

        # 拼接特征
        combined_features = torch.cat((
            cls_tokens,
            function_words_percentage.unsqueeze(2),
            rLSM.unsqueeze(2)
        ), dim=2)

        # 使用 TransformerEncoder 进行特征聚合
        aggregated_features = self.aggregation_layer(combined_features)

        # 取第一个 token 特征
        cls_token_features_for_dialogue = aggregated_features[:, 0, :]

        # 拼接全局特征
        final_features = torch.cat((
            cls_token_features_for_dialogue,
            LSM.unsqueeze(1),
            M_rLSM.unsqueeze(1)
        ), dim=1)

        final_features = self.dropout(final_features)
        logits = self.fc(final_features)

        return logits


if __name__ == '__main__':
    # 添加多进程支持
    # from multiprocessing import freeze_support
    #
    # freeze_support()

    # 修改为两个数据路径
    data_path_0 = r'D:\毕业论文\对话数据\test\不付费'  # 假设标签为0的数据存放路径
    data_path_1 = r'D:\毕业论文\对话数据\test\付费'  # 假设标签为1的数据存放路径

    # 加载两个数据集
    conversations_0, labels_0, lsm_values_0, m_rlsm_values_0 = load_conversation_data(data_path_0)
    conversations_1, labels_1, lsm_values_1, m_rlsm_values_1 = load_conversation_data(data_path_1)

    # 合并两个数据集
    conversations = conversations_0 + conversations_1
    labels = labels_0 + labels_1
    lsm_values = lsm_values_0 + lsm_values_1
    m_rlsm_values = m_rlsm_values_0 + m_rlsm_values_1

    # 随机打乱数据顺序
    combined = list(zip(conversations, labels, lsm_values, m_rlsm_values))
    random.shuffle(combined)
    conversations, labels, lsm_values, m_rlsm_values = zip(*combined)

    # 转换为列表形式
    conversations = list(conversations)
    labels = list(labels)
    lsm_values = list(lsm_values)
    m_rlsm_values = list(m_rlsm_values)

    # 本地模型存放路径
    local_model_path = r"D:\Large Models\QwenQwen2.5-1.5B-Instruct"

    # 加载本地的Qwen-2模型和分词器
    local_model_path = r"D:\Large Models\QwenQwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # 划分为训练集(60%)、验证集(20%)和测试集(20%)
    conv_train, conv_temp, labels_train, labels_temp, lsm_train, lsm_temp, m_rlsm_train, m_rlsm_temp = train_test_split(
        conversations, labels, lsm_values, m_rlsm_values, test_size=0.4, random_state=42
    )

    conv_val, conv_test, labels_val, labels_test, lsm_val, lsm_test, m_rlsm_val, m_rlsm_test = train_test_split(
        conv_temp, labels_temp, lsm_temp, m_rlsm_temp, test_size=0.5, random_state=42
    )

    # 创建数据集
    train_dataset = CustomDataset(conv_train, labels_train, lsm_train, m_rlsm_train, tokenizer)
    val_dataset = CustomDataset(conv_val, labels_val, lsm_val, m_rlsm_val, tokenizer)
    test_dataset = CustomDataset(conv_test, labels_test, lsm_test, m_rlsm_test, tokenizer)

    # 创建DataLoader
    # shuffle = True,  随机打乱顺序
    # num_workers = 4   多进程加载数据，利用多个CPU核心并行处理，提高数据加载速度，根据CPU核心数调整，最佳实践：设定为CPU核心数的2到4倍
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

    # 初始化模型
    model = CustomModel(model_name=local_model_path, num_labels=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # GPU设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练和验证过程
    num_epochs = 5
    for epoch in tqdm(range(num_epochs), desc="训练轮次进度"):
        model.train()
        total_loss = 0

        # 内层进度条（每个 epoch）
        batch_loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for batch in batch_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            function_words_percentage = batch['function_words_percentage'].to(device)
            rLSM = batch['rLSM'].to(device)
            LSM = batch['LSM'].to(device)
            M_rLSM = batch['M_rLSM'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM)
            loss = criterion(logits, labels)

            print(f"Loss: {loss.item()}")

            #  调试信息
            # 检查损失是否为 nan
            if torch.isnan(loss):
                print("Loss is nan!")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print(f"function_words_percentage: {function_words_percentage}")
                print(f"rLSM: {rLSM}")
                print(f"LSM: {LSM}")
                print(f"M_rLSM: {M_rLSM}")
                print(f"labels: {labels}")
                print(f"Logits: {logits}")
                break

            loss.backward()
            optimizer.step()

            # 实时更新batch_loss到进度条
            batch_loop.set_postfix(batch_loss=f"{loss.item():.4f}")
            wandb.log({"Train Loss": loss.item(), "Epoch": epoch})

            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # 验证集性能评估
        model.eval()
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}] Validation", leave=False)
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                function_words_percentage = batch['function_words_percentage'].to(device)
                rLSM = batch['rLSM'].to(device)
                LSM = batch['LSM'].to(device)
                M_rLSM = batch['M_rLSM'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM)
                val_loss = criterion(logits, labels)

                total_val_loss += val_loss.item()
                val_loop.set_postfix(batch_loss=f"{val_loss.item():.4f}")

                # 记录每个批次的验证损失到wandb
                wandb.log({"val_batch_loss": val_loss.item()})

        avg_val_loss = total_val_loss / len(val_loader)

        wandb.log({"train_epoch_loss": avg_train_loss, "val_epoch_loss": avg_val_loss})

        print(f"Epoch [{epoch + 1}/{num_epochs}] 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

    # 测试集最终评估
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="测试集评估", leave=True)
        for batch in test_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            function_words_percentage = batch['function_words_percentage'].to(device)
            rLSM = batch['rLSM'].to(device)
            LSM = batch['LSM'].to(device)
            M_rLSM = batch['M_rLSM'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 输出详细的测试集分类报告
    print("\n测试集最终性能报告：")
    print(classification_report(all_labels, all_preds, digits=4))
