# 高斯核对数几率回归利用核方法的表示定理解决线性不可分的问题
# 不知道为什么加上偏移量之后就会出现奇异矩阵不可逆的问题，改成没有偏移量之后拟合（估计是过拟合）得很好
# 本质上和对数几率回归的差别就在于求了一个核矩阵，其中kernal_matrix[i][j]=kernel(xi,xj)
import numpy as np
import csv
kernel_matrix = []
raw_data = []
label = np.zeros((17,), dtype=np.int32)  # 标签值
beta = np.zeros((17, 1))  # 初始化参数


def read_data():
    with open("watermelon.csv", 'r')as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            label[i] = int(line[-1])
            X = []
            for t in line[:-1]:
                X.append(float(t))
            raw_data.append(np.array(X, dtype=np.float32))


def Guass_kernel(x, y):
    sigma = 0.5
    return np.exp(-0.5 * np.sum((x - y)**2) / sigma)


def compute_kernel_matrix():
    for x in raw_data:
        product = []
        for y in raw_data:
            product.append([Guass_kernel(x, y)])
        # product.append([0.5])
        kernel_matrix.append(np.array(product, dtype=np.float32))


def calcu_loss():  # 计算误差
    loss = 0
    for i, X in enumerate(kernel_matrix):
        z = beta.T@X
        loss += (-label[i] * z + np.log(1 + np.exp(z)))
        return loss


def Newton_iterate(num):
    partial_beta_1 = np.zeros((17, 1), dtype=np.float32)  # loss对beta的一阶导数
    partial_beta_2 = np.zeros((17, 17), dtype=np.float32)
    old_loss = 0
    new_loss = 0
    time = 0
    while(1):
        new_loss = calcu_loss()
        if abs(new_loss - old_loss) < 1e-5 or time > num:
            break
        old_loss = new_loss
        time += 1
        for i, X in enumerate(kernel_matrix):
            global beta
            z = np.exp(beta.T@X)
            delta1 = (1 - label[i] - 1 / (1 + z))
            delta2 = (z / ((1 + z) * (1 + z)))
            partial_beta_1 += delta1 * X
            partial_beta_2 += delta2 * X@X.T
            # print(partial_beta_2)
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
    for i, X in enumerate(kernel_matrix):
        #print(X, func(X))
        pred = 0 if func(X) < 0.5 else 1
        right += (pred == label[i])
        print(func(X), label[i])
    print(right / len(kernel_matrix))


def main():
    read_data()
    compute_kernel_matrix()
    # print(kernel_matrix)
    Newton_iterate(100)
    test()


if __name__ == '__main__':
    main()
