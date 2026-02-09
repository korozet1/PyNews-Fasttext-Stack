class Config(object):
    def __init__(self):
        #原始数据路径
        self.train_datapath=r".\data\train.txt"
        self.test_datapath = r".\data\test.txt"
        self.dev_datapath = r".\data\dev.txt"

        #模型路径
        self.ft_model_save_path=r".\save_models"

        #样本类别文件
        self.class_datapath=r".\data\class.txt"

        #处理完的数据（用于训练）
        self.final_data=r".\result"

if __name__ == '__main__':
    conf=Config()
    print(conf.ft_model_save_path)
    print(conf.train_datapath)