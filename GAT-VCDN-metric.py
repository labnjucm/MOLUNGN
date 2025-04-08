import random

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GATConv

from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
class GATClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_heads):
        super(GATClassifier, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=nn.ReLU())
        self.gat2 = GATConv(hidden_feats * num_heads, num_classes, 1, activation=None)

    def forward(self, graph, features):
        h = self.gat1(graph, features)
        h = h.view(h.size(0), -1)  # Flatten heads
        h = self.gat2(graph, h)
        return h.squeeze(1)  # Output logits

# 根据GAT模型的注意力权重计算特征重要性（假设这里我们可以通过一些方法获得特征重要性）
def get_feature_importance(model, graph, features):
    # 获取第一层GAT的注意力权重
    gat1_attention_weights = model.gat1.attn_l  # 这里我们使用GAT第一层的attention权重
    # 假设你有办法计算或获取GAT中的注意力权重来评估特征的重要性
    # 对于示例，我们可以使用随机生成的权重来模拟
    feature_importance = np.random.rand(features.shape[1])  # 随机模拟特征重要性
    return feature_importance


# 提取前100个重要特征
def extract_top_features(data_file, feature_names_file, model, graph, features, top_n=100):
    # 加载数据
    data = pd.read_csv(data_file, header=None).values
    feature_names = pd.read_csv(feature_names_file, header=None).values.flatten()  # 获取特征名称

    # 计算特征重要性
    feature_importance = get_feature_importance(model, graph, features)

    # 获取前100个重要特征的索引
    top_indices = np.argsort(feature_importance)[-top_n:]

    # 提取对应的特征名称
    top_feature_names = feature_names[top_indices]

    # 提取对应的数据
    top_features = data[:, top_indices]

    return top_feature_names, top_features


# 示例调用：对于每个视角提取前100个特征
def extract_features_for_all_views(data_files, feature_names_files, model, graphs, all_features):
    all_top_features = []
    all_top_feature_names = []

    for i in range(len(data_files)):
        top_feature_names, top_features = extract_top_features(
            data_files[i], feature_names_files[i], model, graphs[i], all_features[i]
        )
        all_top_features.append(top_features)
        all_top_feature_names.append(top_feature_names)

    return all_top_feature_names, all_top_features
# 收集每个视角的预测概率
def collect_view_outputs(data_files, labels, model_params):
    outputs = []
    for data_file in data_files:
        _, _, _, _, logits, y_test = train_and_evaluate_gat(data_file, labels, **model_params)
        probs = torch.softmax(torch.tensor(logits), dim=1)  # 转换为概率分布
        outputs.append(probs)
    return torch.stack(outputs, dim=1), y_test  # [batch_size, num_views, num_classes], [batch_size]
# 构建图
def create_graph(data):
    num_nodes = data.shape[0]
    graph = dgl.graph(([], []), num_nodes=num_nodes)
    graph.add_edges(torch.arange(num_nodes), torch.arange(num_nodes))  # Self-loops
    return graph
# GAT 模型训练与测试

def train_and_evaluate_gat(data_file, labels, hidden_feats=16, num_heads=32, epochs=500, lr=0.008):
    # 加载数据
    data = pd.read_csv(data_file, header=None).values
    labels = labels.values.squeeze() - 1  # Adjust labels to start from 0

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # 转换为图
    train_graph = create_graph(X_train)
    test_graph = create_graph(X_test)

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 初始化模型
    model = GATClassifier(X_train.shape[1], hidden_feats, len(np.unique(labels)), num_heads)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(epochs):
        model.train()
        logits = model(train_graph, X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每轮计算并打印性能指标
        model.eval()
        with torch.no_grad():
            val_logits = model(test_graph, X_test)
            preds = torch.argmax(val_logits, dim=1)
            # 计算性能指标
            accuracy = accuracy_score(y_test, preds)
            recall = recall_score(y_test, preds, average='weighted')
            f1_weighted = f1_score(y_test, preds, average='weighted')
            f1_macro = f1_score(y_test, preds, average='macro')

            print(f"GAT:Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}, "
                  f"Recall: {recall:.2f}, F1_weighted: {f1_weighted:.2f}, F1_macro: {f1_macro:.2f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        logits = model(test_graph, X_test)
        preds = torch.argmax(logits, dim=1)

    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='weighted')
    f1_weighted = f1_score(y_test, preds, average='weighted')
    f1_macro = f1_score(y_test, preds, average='macro')

    return accuracy, recall, f1_weighted, f1_macro, logits, y_test


def write_to_file(content, prefix,seed,vepoch):
    # 获取当前时间，格式为 YYYY-MM-DD_HH-MM-SS
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_LUAD/result/seed-{seed}-vepoch-{vepoch}-{prefix}_{current_time}.txt"

    # 将内容写入文件
    with open(filename, 'w+') as f:
        f.write(content)


def set_seed(seed):
    random.seed(seed)  # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 的随机数生成器种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 如果使用 GPU，则设置 CUDA 的种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
        torch.backends.cudnn.deterministic = True  # 确保 CUDNN 的结果可复现
        torch.backends.cudnn.benchmark = False  # 避免动态优化
# 主函数中调用
if __name__ == "__main__":

    SEED = 42
    set_seed(SEED)

    # 数据路径和标签加载
    # data_files = [
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/1.csv',
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/2.csv',
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/3.csv'
    # ]
    # feature_names_files = [
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/1_featname.csv',
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/2_featname.csv',
    #     '/home/njucm/zdf/小论文2/MOGONET-main/LUAD/method-data/3_featname.csv'
    # ]
    data_files = [
        'data_LUAD/1.csv',
        'data_LUAD/2.csv',
        'data_LUAD/3.csv'
    ]
    feature_names_files = [
        'data_LUAD/1_featname.csv',
        'data_LUAD/2_featname.csv',
        'data_LUAD/3_featname.csv'
    ]
    labels = pd.read_csv('data_LUAD/label.csv', header=None,
                         names=['label'])

    # 模型参数
    model_params = {'hidden_feats': 16, 'num_heads': 32, 'epochs': 300, 'lr': 0.008}

    # 收集视角输出
    view_outputs, y_test = collect_view_outputs(data_files, labels, model_params)

    # 构建每个视角的图
    graphs = [create_graph(pd.read_csv(file, header=None).values) for file in data_files]
    all_features = [torch.tensor(pd.read_csv(file, header=None).values, dtype=torch.float32) for file in data_files]

    # 初始化 GAT 模型
    model = GATClassifier(all_features[0].shape[1], 16, len(np.unique(labels)), 32)

    # # 提取每个视角的前100个特征
    # all_top_feature_names, all_top_features = extract_features_for_all_views(data_files, feature_names_files, model,
    #                                                                          graphs, all_features)

    # # 输出每个视角的前100个特征名称
    # for i, top_feature_names in enumerate(all_top_feature_names):
    #     print(f"视角{i + 1}的前100个重要特征: {top_feature_names}")
    #
    # # 示例调用：将特征打印结果写入文件
    # top_features_content = "\n".join(
    #     [f"视角{i + 1}的前100个重要特征: {', '.join(map(str, top_feature_names))}"
    #      for i, top_feature_names in enumerate(all_top_feature_names)]
    # )
    # write_to_file(top_features_content, "top_features")

    # 初始化融合模型 VCDN
    num_views = view_outputs.shape[1]

    num_classes = view_outputs.shape[2]
    print(f"num_views:{num_views},num_classes:{num_classes}")
    vcdn_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_views * num_classes, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    optimizer = optim.Adam(vcdn_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # VCDN 训练
    performance_content = ""
    epochs = 300
    for epoch in range(epochs):
        vcdn_model.train()
        optimizer.zero_grad()
        logits = vcdn_model(view_outputs)
        loss = loss_fn(logits, y_test)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
            vcdn_model.eval()
            with torch.no_grad():
                logits = vcdn_model(view_outputs)
                preds = torch.argmax(logits, dim=1)
                # 计算总性能指标
                accuracy = accuracy_score(y_test, preds)
                recall = recall_score(y_test, preds, average='weighted')
                f1_weighted = f1_score(y_test, preds, average='weighted')
                f1_macro = f1_score(y_test, preds, average='macro')

                print(f"VCDN当前第{epoch}轮训练，模型融合后的性能指标:准确率: {accuracy:.2f},加权召回率: {recall:.2f}"
                      f"加权F1值: {f1_weighted:.2f},宏F1值: {f1_macro:.2f}")
                performance_content += ("模型融合后的性能指标:当前第{:d},准确率: {:.2f},加权召回率: {:.2f}"
                                        "加权F1值: {:.2f},宏F1值: {:.2f}\n").format(epoch,accuracy,recall,f1_weighted,f1_macro)
                # tmp_content = (
                #     f"模型融合后的性能指标:\n"
                #     f"准确率: {accuracy:.2f}\n"
                #     f"加权召回率: {recall:.2f}\n"
                #     f"加权F1值: {f1_weighted:.2f}\n"
                #     f"宏F1值: {f1_macro:.2f}\n"
                # )
                # print(f"准确率: {accuracy:.2f}")
                # print(f"加权召回率: {recall:.2f}")
                # print(f"加权F1值: {f1_weighted:.2f}")
                # print(f"宏F1值: {f1_macro:.2f}")
            vcdn_model.train()

    # VCDN 测试
    vcdn_model.eval()
    with torch.no_grad():
        logits = vcdn_model(view_outputs)
        preds = torch.argmax(logits, dim=1)

    # 计算总性能指标
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds, average='weighted')
    f1_weighted = f1_score(y_test, preds, average='weighted')
    f1_macro = f1_score(y_test, preds, average='macro')

    print(f"总性能指标:")
    print(f"准确率: {accuracy:.2f}")
    print(f"加权召回率: {recall:.2f}")
    print(f"加权F1值: {f1_weighted:.2f}")
    print(f"宏F1值: {f1_macro:.2f}")

    # 示例调用：将性能指标结果写入文件
    performance_content += "总性能指标:准确率: {:.2f},加权召回率: {:.2f},加权F1值: {:.2f},宏F1值: {:.2f}\n".format(accuracy, recall, f1_weighted, f1_macro)
    # performance_content = (
    #     f"模型融合后的性能指标:\n"
    #     f"准确率: {accuracy:.2f}\n"
    #     f"加权召回率: {recall:.2f}\n"
    #     f"加权F1值: {f1_weighted:.2f}\n"
    #     f"宏F1值: {f1_macro:.2f}\n"
    # )
    write_to_file(performance_content, "performance",SEED,epoch)