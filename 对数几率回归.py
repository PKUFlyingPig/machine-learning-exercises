# 对数几率回归是线性回归的一种变形，通过sigmoid函数将线性变换的值映射到【0，1】区间从而用来解决二分类问题
# 具体过程通过极大似然的方法得到需要优化的损失函数，然后通过牛顿迭代优化求解
# 由于数据线性不可分，所以最终准确率只有70%。对数几率回归本质上还是线性回归问题，只不过线性拟合的(wx+b)不是y而是In(y/(1-y))，即所谓的对数几率

import numpy as np
import csv
data = []
label = np.zeros((17,), dtype=np.int32)  # 标签值
beta = np.array([[0], [0], [1]], dtype=np.float32,)  # 初始化参数


def read_data():
    with open("watermelon_data.csv", 'r')as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            label[i] = int(line[2])
            X = [[line[0]], [line[1]], [1]]  # 数据加一个值为1的维度
            data.append(np.array(X, dtype=np.float32))


def calcu_loss():  # 计算误差
    loss = 0
    for i, X in enumerate(data):
        z = beta.T@X
        loss += (-label[i] * z + np.log(1 + np.exp(z)))
        return loss


def Newton_iterate(num):
    partial_beta_1 = np.zeros((3, 1), dtype=np.float32)  # loss对beta的一阶导数
    partial_beta_2 = np.zeros((3, 3), dtype=np.float32)
    old_loss = 0
    new_loss = 0
    time = 0
    while(1):
        new_loss = calcu_loss()
        if abs(new_loss - old_loss) < 1e-5 or time > num:
            break
        old_loss = new_loss
        time += 1
        for i, X in enumerate(data):
            global beta
            z = np.exp(beta.T@X)
            delta1 = (1 - label[i] - 1 / (1 + z))
            delta2 = (z / ((1 + z) * (1 + z)))
            partial_beta_1 += delta1 * X
            partial_beta_2 += delta2 * X@X.T
            delta = np.linalg.inv(partial_beta_2)@partial_beta_1
        beta = beta - delta
    print(beta)


def func(x):
    global beta
    z = beta.T@x
    #print(beta, z)
    ans = 1 / (1 + np.exp(-z))
    # print(ans)
    return ans


def test():
    right = 0
    for i, X in enumerate(data):
        #print(X, func(X))
        pred = 0 if func(X) < 0.5 else 1
        right += (pred == label[i])
    print(right / len(data))


def main():
    read_data()
    global beta
    Newton_iterate(1000)
    test()


if __name__ == '__main__':
    main()
