import pandas as pd

# 读取数据文件
label_file = 'data_LUSC/label.csv'
labels = pd.read_csv(label_file, header=None, names=['label'])

# 筛选分类为3和4的标签
filtered_labels = labels[labels['label'].isin([2, 3])]

# 保存到新文件label34.csv
filtered_labels.to_csv('../GAT-VCDN/data_LUSC/label23.csv', index=False, header=False)
print("分类为3和4的标签已保存到label34.csv中！")
