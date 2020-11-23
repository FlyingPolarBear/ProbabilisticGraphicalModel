import numpy as np

'''
lambda = (A, B, pi)
观测值为O，序列长度为T，状态集合取值数为N，观测集合取值数为K
A: NxN | B: NxK | pi: Nx1 | O: Tx1
'''


class HMM():
    def __init__(self, N, K, O):
        self.T = O.shape[0]  # 序列长度
        self.N = N  # 状态集合取值数
        self.K = K  # 观测集合取值数
        self.O = O  # 观测值O: Tx1
        self.A = np.ones((N, N))/N  # A: NxN
        self.B = np.ones((N, K))/K  # B: NxK
        self.pi = np.ones(N)/N  # pi: Nx1

    def ModelInit(self, A, B, pi, O):  # 适用于概率计算和预测问题的初始化
        self.T = O.shape[0]  # 序列长度
        self.N = B.shape[0]  # 状态集合取值数
        self.K = B.shape[1]  # 观测集合取值数
        self.O = O  # 观测值O: Tx1
        self.A = A  # A: NxN
        self.B = B  # B: NxK
        self.pi = pi  # pi: Nx1

    def Alpha(self):  # alpha: TxN
        alpha = np.zeros([self.T, self.N])
        alpha[0] = self.B[:, O[0]]*self.pi  # 初值
        for i in range(1, self.T):
            alpha[i] = np.dot(alpha[i-1], self.A) * self.B[:, O[i]]  # 递推
        self.alpha = alpha

    def Beta(self):  # beta: TxN
        beta = np.ones([self.T, self.N])  # 初值
        for i in range(self.T-1, 0, -1):
            beta[i-1] = np.dot(beta[i]*self.B[:, O[i]], self.A.T)  # 递推
        self.beta = beta

    def Gamma(self):  # gamma: TxN
        self.Alpha()
        self.Beta()
        self.gamma = np.zeros([self.T, self.N])
        for t in range(self.T):
            denominator = np.dot(self.alpha[t].T,self.beta[t])
            for i in range(self.N):
                self.gamma[t][i] = self.alpha[t][i]*self.beta[t][i]/denominator

    def Xi(self):  # xi: (T-1)xNxN
        self.Alpha()
        self.Beta()
        self.xi = np.zeros([self.T-1, self.N, self.N])
        for t in range(self.T-1):
            for i in range(self.N):
                for j in range(self.N):
                    self.xi[t][i][j] = self.alpha[t][i] * self.A[i][j] * \
                        self.B[j][self.O[t]] * self.beta[t+1][j]
            self.xi[t] /= np.sum(self.xi[t])
        pass

    def Forward(self, A, B, pi, O):
        self.ModelInit(A, B, pi, O)
        self.Alpha()
        self.P = np.sum(self.alpha[self.N-1])  # 终止
        print("Forward P(O|lamda) =", self.P)

    def Backward(self, A, B, pi, O):
        self.ModelInit(A, B, pi, O)
        self.Beta()
        self.P = np.dot(self.pi*B[:, O[0]],self.beta[0])  # 终止
        print("Backward P(O|lamda) =", self.P)

    def Baum_Welch(self, epoch):
        for n in range(epoch):
            self.Gamma()
            self.Xi()
            self.A = np.sum(self.xi, axis=0) / \
                np.sum(self.gamma[:-1], axis=0).reshape(-1, 1)
            for v in range(self.K):
                self.B[:, v] = np.sum(self.gamma[O == v], axis=0) / \
                    np.sum(self.gamma, axis=0)
            self.pi = self.gamma[0]
        print("Baum-Welch Algorithm:")
        print("A:\n", m.A, "\nB:\n", m.B, "\npi:\n", m.pi)

    def Approximate(self, A, B, pi, O):
        self.ModelInit(A, B, pi, O)
        self.Gamma()
        self.I = np.argmax(self.gamma, axis=1)
        print("Approximate I =", self.I)

    def Viterbi(self, A, B, pi, O):
        self.ModelInit(A, B, pi, O)
        delta = np.zeros((self.T, self.N))
        psi = np.zeros((self.T, self.N))
        delta[0] = self.pi*self.B[:, O[0]]  # 初始化
        for t in range(1, self.T):  # 递推
            for i in range(self.N):
                delta[t][i] = np.max(delta[t-1]*A[:, i])*B[i, O[t]]
                psi[t][i] = np.argmax(delta[t-1]*A[:, i])
        self.P = np.max(delta[self.T-1])
        self.I = np.zeros(self.T, dtype=np.int)
        self.I[self.T-1] = np.argmax(delta[self.T-1])  # 终止
        for t in range(self.T-2, -1, -1):  # 最优路径回溯
            self.I[t] = psi[t+1][self.I[t+1]]
        print("Viterbi I =", self.I)


# 《统计学习方法》中的示例
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])  # A: NxN
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])  # B: NxK
pi = np.array([0.2, 0.4, 0.4])  # pi: Nx1
O = np.array([0, 1, 0])  # O: Tx1

m = HMM(B.shape[0], B.shape[1], O)
m.Forward(A, B, pi, O)  # 概率计算：前向算法
m.Backward(A, B, pi, O)  # 概率计算：后向算法
m.Approximate(A, B, pi, O)  # 预测问题：近似算法
m.Viterbi(A, B, pi, O)  # 预测问题：维特比算法
m.Baum_Welch(10)  # 学习问题：Baum-Welch算法
