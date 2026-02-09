# 导入工具包
import fasttext  # pip install fasttext-wheel
from config import Config
import datetime

# 获取时间
current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")
conf = Config()

# 1、模型训练
model = fasttext.train_supervised(input='result/train_fastText_jieba.txt',epoch=100)
# 2、模型保存
path = conf.ft_model_save_path
model.save_model(path + f"\\fastText_jieba_default_{str(current_time)}.bin")
# 3、模型预测
print(model.predict("《 赤 壁 O L 》 攻 城 战 诸 侯 战 硝 烟 又 起"))

# 4、模型词表查看
print(f"*查看模型词表[:10]：{model.words[:10]}")
## 单词向量表示
# print(f"单词的向量：{model.get_word_vector(model.words[:9][1])}")
print(f"单词的向量：{model.get_word_vector('中')}")
## 预测最接近的单词
print(f"*预测最接近的单词:{model.get_nearest_neighbors('中')}")

# 5、查看模型子词，上述训练未开启子词，所以这里查到还是词本身
print(f"*模型字词：{model.get_subwords('你')}")

# 6、模型测试评估
res = model.test('./result/test_fastText_jieba.txt')
# 参考官方说明：https://fasttext.cc/docs/en/supervised-tutorial.html
# (10000, 0.8761, 0.8761) 样本量 精确率 召回率
print(res)