from sklearn import svm
import numpy as np
import csv


def read_data():
    data = []
    label = []
    with open("watermelon.csv", 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(list(map(float, line[:-1])))
            label.append(int(line[-1]))
    return data, label


def SVM_clf(data=[], label=[]):  # SVM 二分类问题
    print("=== 高斯核支持向量 ===")
    clf = svm.SVC(C=50, kernel="rbf", gamma="auto")
    clf.fit(data, label)
    print(clf.support_vectors_)

    print("=== 线性核支持向量 ===")
    clf = svm.SVC(C=50, kernel="linear")
    clf.fit(data, label)
    print(clf.support_vectors_)
    # print("w:", clf.coef_)
    # print("b:", clf.intercept_)


def SVM_reg(data=[]):  # 支持向量回归
        # 把原来分类问题的输入向量的前7个分量作为输入，第8个分量作为输出
    newdata = []
    label = []
    for dot in data:
        label.append(dot[-1])
        newdata.append(dot[-1])
    print(newdata)
    print(label)
    clf = svm.SVR(C=1, epsilon=1e-2, gamma='scale')
    clf.fit(data[:10], label[:10])
    print(clf.predict(data))


def main():
    data, label = read_data()
    #SVM_clf(data, label)
    SVM_reg(data)


if __name__ == '__main__':
    main()
