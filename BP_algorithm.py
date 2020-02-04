import numpy as np
import csv
from random import shuffle
data = []  # 每个数据是一个8维的列向量
label = np.zeros((17,), dtype=np.int32)  # 标签值


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s


class FC_network:
    def __init__(self):
        self.d = 8  # 输入空间的维度
        self.q = 4  # 隐层维度
        self.l = 1  # 输出层维度
        self.W1 = np.random.normal(size=(self.q, self.d))  # 初始化参数
        self.theta1 = np.random.normal(size=(self.q, 1))
        self.W2 = np.random.normal(size=(self.l, self.q))
        self.theta2 = np.random.normal(size=(self.l, 1))

    def print_para(self):
        print("W1:", self.W1)
        print("theta1:", self.theta1)
        print("W2:", self.W2)
        print("theta2:", self.theta2)

    def FP(self, X):  # 前向传播
        z1 = self.W1@X - self.theta1
        b = sigmoid(z1)
        z2 = self.W2@b - self.theta2
        y = sigmoid(z2)
        return y

    def standard_bp(self, label, data=[], learning_rate=1e-4, epoch=1000):
        for i in range(epoch):
            loss = 0
            for X, Y in zip(data, label):
                z1 = self.W1@X - self.theta1
                b = sigmoid(z1)
                z2 = self.W2@b - self.theta2
                y = sigmoid(z2)
                loss += 0.5 * np.sum((y - Y) ** 2)  # 均方误差损失函数
                delta2 = (y - Y) * y * (1 - y)  # 损失函数对第二个隐层输入向量的导数
                delta1 = (self.W2.T @ delta2) * b * \
                    (1 - b)  # 损失函数对第一个隐层输入向量的导数
                d_W2 = delta2@b.T
                d_theta2 = -delta2
                d_W1 = delta1@X.T
                d_theta1 = -delta1
                self.W1 -= learning_rate * d_W1
                self.W2 -= learning_rate * d_W2
                self.theta1 -= learning_rate * d_theta1
                self.theta2 -= learning_rate * d_theta2
            if i % 100 == 0:
                print(f"loss={loss}")

    def acc_bp(self, label, data=[], learning_rate=1e-1, epoch=1000):
        for i in range(epoch):
            loss = 0
            d_W1 = d_W2 = d_theta1 = d_theta2 = 0
            for X, Y in zip(data, label):
                z1 = self.W1@X - self.theta1
                b = sigmoid(z1)
                z2 = self.W2@b - self.theta2
                y = sigmoid(z2)
                loss += 0.5 * np.sum((y - Y) ** 2)
                delta2 = (y - Y) * y * (1 - y)
                delta1 = (self.W2.T @ delta2) * b * (1 - b)
                d_W2 += delta2@b.T
                d_theta2 += -delta2
                d_W1 += delta1@X.T
                d_theta1 += -delta1
            self.W1 -= learning_rate * d_W1
            self.W2 -= learning_rate * d_W2
            self.theta1 -= learning_rate * d_theta1
            self.theta2 -= learning_rate * d_theta2
            if i % 100 == 0:
                print(f"loss={loss}")

    def test(self, label, data=[]):
        correct = 0
        for X, Y in zip(data, label):
            pred_y = self.FP(X)
            pred = 0 if pred_y < 0.5 else 1
            correct += (pred == Y)
            print(pred_y, Y)
        return(correct / len(data))


def read_data():
    with open("watermelon.csv", 'r')as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            label[i] = int(line[-1])
            X = []
            for i in range(len(line) - 1):
                X.append([line[i]])
            data.append(np.array(X, dtype=np.float32))


def main():
    read_data()
    # for X, Y in zip(data, label):
    #     print(X, Y)
    print("standard_bp:\n")
    network = FC_network()
    network.acc_bp(label=label, data=data,
                   learning_rate=0.5, epoch=2000)
    print("test_result:")
    print(f"correct_ratio:{network.test(label=label, data=data)*100}%")

    print("accumulated_bp:\n")
    network = FC_network()
    network.standard_bp(label=label, data=data,
                        learning_rate=0.8, epoch=2000)
    print("test_result:")
    print(f"correct_ratio:{network.test(label=label, data=data)*100}%")


if __name__ == '__main__':
    main()
