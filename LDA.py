# LDA(线性判别分析)是一种很重要的有监督降维方法，可以将N个类别的数据降到N-1维
# 核心思想是将数据投影到一个超平面上，使得类内方差尽可能小，类间距离尽可能大
# 转化为广义瑞利商的优化问题，从而用拉格朗日方法求解
# 最终的投影直线的方向向量为Sw-1*(mean1-mean2),其中Sw为两个类别各自协方差矩阵的和
# 下面的代码解决的是一个简单的二分类问题
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 字体管理器

font = FontProperties(fname="/Users/apple/Library/Fonts/楷体_2312.ttf", size=15)


def read_data():
    data1 = []
    label1 = []
    data2 = []
    label2 = []
    with open("watermelon_data.csv", 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if int(line[-1]) == 0:
                data1.append(list((map(float, line[:-1]))))
                label1.append(int(line[-1]))
            else:
                data2.append(list((map(float, line[:-1]))))
                label2.append(int(line[-1]))
    return (np.array(data1), label1, np.array(data2), label2)


def Compute_mean_and_cov(data=[]):
    mean = np.mean(data, axis=0)
    data = data - mean
    d = len(data[0])
    cov = np.zeros((d, d), dtype=np.float32)
    for x in data:
        #print(x, d)
        x = np.array([x])
        cov = cov + x.T@x
    return mean, cov


def LDA(data1=[], label1=[], data2=[], label2=[]):
    mean1, cov1 = Compute_mean_and_cov(data=data1)
    mean2, cov2 = Compute_mean_and_cov(data=data2)
    Sw = cov1 + cov2
    w = np.linalg.inv(Sw)@(mean1 - mean2)
    w = -w
    result1 = []
    result2 = []
    for x in data1:
        result1.append(w@x)
    for x in data2:
        result2.append(w@x)
    x1 = [x[0] for x in data1]
    y1 = [x[1] for x in data1]
    x2 = [x[0] for x in data2]
    y2 = [x[1] for x in data2]

    t1 = [0, w[0]]
    t2 = [0, w[1]]
    plt.plot(x1, y1, '+b', x2, y2, '^g', t1, t2, '-r')
    plt.xlabel("含糖率", FontProperties=font)
    plt.ylabel("重量", FontProperties=font)
    plt.title("LDA投影向量", FontProperties=font)
    #plt.axis([0.2, 0.9, 0, 0.6])
    plt.show()
    #print(mean1, cov1)
    #print(data2, label2)


def main():
    #data1, label1, data2, label2 = read_data()
    LDA(*(read_data()))


if __name__ == '__main__':
    main()
