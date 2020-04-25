import math
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import random as rnd
from one_dim import OneDimMinimizator

def test_func(x):
    return 3 * x[0] + x[1] + 4 * math.sqrt(1 + x[0] * x[0] + 3 * x[1] * x[1])

def test_func_grad(x):
    x0 = 3 + 4 * x[0] / math.sqrt(1 + x[0] * x[0] + 3 * x[1] * x[1])
    x1 = 1 + 12 * x[1] / math.sqrt(1 + x[0] * x[0] + 3 * x[1] * x[1])
    return np.array([x0, x1])

def test_func_gess(x):
    sqrt = math.sqrt(1 + x[0] * x[0] + 3 * x[1] * x[1])
    a11 = 4 * (3 * x[1] * x[1] + 1) / (sqrt * (1 + x[0] * x[0] + 3 * x[1] * x[1]))
    a21 = - 12 * x[0] * x[1] / (sqrt * (1 + x[0] * x[0] + 3 * x[1] * x[1]))
    a22 = 12 * (x[0] * x[0] + 1) / (sqrt * (1 + x[0] * x[0] + 3 * x[1] * x[1]))
    return np.array([[a11, a21], [a21, a22]])

def inv(A):
    rev_det = 1 / (A[0,0] * A[1,1] - A[0,1] * A[1,0])
    return rev_det * np.array([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]])



class FuncWrapper:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def get_value(self, x):
        self.count += 1
        return self.func(x)

    def reset_count(self):
        self.count = 0

class MulDimMinimizator:
    def __init__(self, f, grad, gess, x0, eps):
        self.f = f
        self.grad = grad
        self.gess = gess
        self.x0 = x0
        self.eps = eps
        self.pk = []

    def Gradient1stOrder(self):
        x_k = self.x0
        grad_k = self.grad.get_value(x_k)
        eps = self.eps
        while (LA.norm(grad_k) * LA.norm(grad_k) >= eps):
            def cond(a):
                return self.f.get_value(x_k - a * grad_k)

            min = OneDimMinimizator(FuncWrapper(cond), 0, 1, self.eps)
            ak, bk = min.fibonacci_min()
            a = (bk + ak) / 2
            x_k1 = x_k - a * grad_k

            p_k = x_k1 - x_k
            self.pk.append(p_k / LA.norm(p_k))
            plt.plot([x_k[0], x_k1[0]], [x_k[1], x_k1[1]])

            x_k = x_k1
            grad_k = self.grad.get_value(x_k)

        return x_k

    def checkX_k(self):
        p_prev = self.pk[0]
        for p_next in self.pk[1:]:
            print("(x_prev, x_next) =", p_next.dot(p_prev))
            p_prev = p_next
        return

    def Gradient2ndOrder(self):
        x_k = self.x0
        grad_k = self.grad.get_value(x_k)

        while (LA.norm(grad_k) * LA.norm(grad_k) >= self.eps):
            H_k = inv(self.gess.get_value(x_k))
            d_k = - H_k.dot(grad_k)

            a = 1
            e = 0.25
            _right = e * grad_k.dot(d_k)
            x_k1 = x_k + a * d_k
            while ((self.f.get_value(x_k1) - self.f.get_value(x_k)) > a * _right):
                a /= 2
                x_k1 = x_k + a * d_k

            x_k = x_k1
            grad_k = self.grad.get_value(x_k)

        return x_k

def findLipsh(a, b, num, grad):
    max = 0
    for i in range(num):
        x = np.array([(rnd.random() - 0.5) * 2 * a, (rnd.random() - 0.5) * 2 * b])
        y = np.array([(rnd.random() - 0.5) * 2 * a, (rnd.random() - 0.5) * 2 * b])
        norm_g = LA.norm(grad.get_value(x) - grad.get_value(y))
        norm_v = LA.norm(x - y)
        if norm_v != 0:
            L = (norm_g / norm_v)
            if L > max:
                max = L
    return max

def find_m_M(a, b, num, gess):
    m_min = math.inf
    M_max = 0
    for i in range(num):
        x = np.array([(rnd.random() - 0.5) * 2 * a, (rnd.random() - 0.5) * 2 * b])
        y = np.array([(rnd.random() - 0.5) * 2 * a, (rnd.random() - 0.5) * 2 * b])
        norm_y = LA.norm(y)
        norm_y *= norm_y
        center = y.dot(gess.get_value(x)).dot(y)
        if norm_y != 0:
            rat = center / norm_y
            if rat < m_min:
                m_min = rat
            if rat > M_max:
                M_max = rat
    return m_min, M_max


func = FuncWrapper(test_func)
grad = FuncWrapper(test_func_grad)
gess = FuncWrapper(test_func_gess)
real = np.array([-1.5 * math.sqrt(0.6), - 1 / (2 * math.sqrt(15))])
min = MulDimMinimizator(func, grad, gess, np.array([1,1]), 0.1)

ans1 = min.Gradient1stOrder()
print("Answer:", ans1)
print("Difference", LA.norm(ans1 - real))
print(func.count, grad.count, gess.count)

func.reset_count()
grad.reset_count()
gess.reset_count()
print()

ans2 = min.Gradient2ndOrder()
print("Answer:", ans2)
print("Difference", LA.norm(ans2 - real))
print(func.count, grad.count, gess.count)
min.checkX_k()
plt.axis('equal')
plt.show()
#print("L =", findLipsh(0.002, 0.002, 1000, grad))
#print("M's =", find_m_M(5, 5, 1000, gess))