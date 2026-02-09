import jieba
import os
from config import *

conf=Config()
# 步骤 1：检查文件存在性
class_file = conf.class_datapath
# class_file = conf.c
# data_file = conf.train_datapath
data_file = conf.dev_datapath
# data_file = conf.test_datapath

if not os.path.exists(class_file) or not os.path.exists(data_file):
    print("类别文件及原始数据路径：", class_file, data_file)
    print("文件不存在，请检查文件是否存在！")
# print("类别文件及原始数据路径：", class_file, data_file)
# 步骤 2：设置预处理后文件保存路径
#是否使用字符分词
use_char_segmentation = False
if use_char_segmentation==True:
    #文件写入路径
    if "train" in data_file:
        output_file = conf.final_data+'/train_fastText_char.txt'
    elif "test" in data_file:
        output_file = conf.final_data+'/test_fastText_char.txt'
    else:
        output_file = conf.final_data+'/dev_fastText_char.txt'
else:
    if "train" in data_file:
        output_file = conf.final_data+'/train_fastText_jieba.txt'
    elif "test" in data_file:
        output_file = conf.final_data+'/test_fastText_jieba.txt'
    else:
        output_file = conf.final_data+'/dev_fastText_jieba.txt'
id2name = {}
print("类别文件:", class_file)

with open(class_file, 'r', encoding='utf-8') as f:
    for index,data in enumerate(f):
        # print(index,data)
        id2name[index] = data.strip('\n')  # 去除换行符，映射索引到类别名

print("类别映射:", id2name)
datas = []
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()  # 去除换行符和空白
        if not line:
            continue  # 跳过空行

        text,  label = line.split('\t')  # 以制表符分割
        label_name = f"__label__{id2name[int(label)]}"  # 转换标签，如 __label__education
        text = text.replace('：', '')  # 部分文本含有冒号，这里移除冒号
        # use_char_segmentation为 True，则是字符级分词；为False，则是jieba 词级分词
        words = list(text) if use_char_segmentation else jieba.cut(text)  # 字符分词或词级分词
        text_processed = ' '.join(word for word in words if word.strip())  # 拼接分词结果

        fasttext_line = f"{label_name} {text_processed}"
        datas.append(fasttext_line)  # 添加到列表
# print(datas[:5])
with open(output_file, 'w', encoding='utf-8') as f:
    for line in datas:
        f.write(line + '\n')  # 写入每行
print("前 5 行:", datas[:5])
print(f"数据已保存到 {output_file}")