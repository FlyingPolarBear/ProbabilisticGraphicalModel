import numpy as np

'''
序列长度为T，取值数为N
概率计算时增加strat->y1的连接
'''


class CRF():
    def __init__(self, lamda, mu):
        self.lamda = lamda
        self.mu = mu
        self.T = mu.shape[0]
        self.N = mu.shape[1]
        lamda0 = np.insert(lamda, 0, np.ones(self.N*self.N)
                           ).reshape(self.T, self.N, self.N)  # 增加strat->y1
        self.M = np.ones((self.T, self.N, self.N))
        for i in range(self.T):
            self.M[i] = np.exp(lamda0[i]+mu[i])

    def Alpha(self):
        self.alpha = np.zeros((self.T, self.N))
        self.alpha[0] = np.ones(self.N)
        for i in range(1, self.T):
            self.alpha[i] = np.dot(self.alpha[i-1], self.M[i])

    def Beta(self):
        self.beta = np.zeros((self.T, self.N))
        self.beta[self.T-1] = np.ones(self.N)
        for i in range(self.T-1, 0, -1):
            self.beta[i-1] = np.dot(self.beta[i], self.M[i])

    def Evaluation(self):  # 前向-后向算法求解概率计算问题
        self.Alpha()
        self.Beta()
        Z = np.sum(self.alpha[self.T-1])
        # Z = np.sum(self.beta[0])
        P1 = np.zeros(self.T)
        P2 = np.zeros(self.T-1)
        for i in range(self.T):
            P1[i] = np.dot(self.alpha[i].T, self.beta[i])/Z
        for i in range(self.T-1):
            P2[i] = np.dot(np.dot(self.alpha[i].T, self.M[i]),
                           self.beta[i+1])/Z
        print("P(y_i|x) =\n", P1)
        print("P(y_i-1,y_i|x) =", P2)

    def Viterbi(self):  # 维特比算法求解预测问题
        delta = np.zeros((self.T, self.N))
        si = np.zeros((self.T, self.N))
        y = np.zeros(self.T, dtype=np.int)
        delta[0] = self.mu[0]  # 初始化
        for i in range(1, self.T):  # 递推
            for l in range(self.N):
                delta[i][l] = np.max(
                    delta[i-1]+self.lamda[i-1][:, l]+self.mu[i][l], axis=0)
                si[i][l] = np.argmax(
                    delta[i-1]+self.lamda[i-1][:, l]+self.mu[i][l], axis=0)
        y[self.T-1] = np.argmax(delta[self.T-1])  # 终止
        for i in range(self.T-1, 0, -1):  # 返回路径
            y[i-1] = si[i][y[i]]
        print('Best Path using Viterbi:', y)


def CRF_manual():
    lamda = np.array([[[0.6, 1], [1, 0.0]],  # Edge: Y1=(1,2)->Y2=(1,2)
                      [[0.0, 1], [1, 0.2]]])  # Edge: Y2=(1,2)->Y3=(1,2)
    mu = np.array([[1.0, 0.5],  # X1->Y1=(1,2)
                   [0.8, 0.5],  # X2->Y2=(1,2)
                   [0.8, 0.5]])  # X3->Y3=(1,2)
    m = CRF(lamda, mu)
    m.Evaluation()
    m.Viterbi()


if __name__ == '__main__':
    CRF_manual()
