import numpy as np
import math


class TransportTask:
    def __init__(self, a, b, C):
        self.a = np.array(a)
        self.b = np.array(b)
        self.C = np.array(C)
        self.N = np.arange(a.size)
        self.M = np.arange(b.size)

    def check_solution(self, x):
        if (x.sum(axis=1).T == self.a).all() != True:
            return False

        if (x.sum(axis=0) == self.b).all() != True:
            return False

        cost = 0
        for i in self.N:
            for j in self.M:
                if self.C[i,j] != math.inf:
                    cost += x[i,j] * self.C[i,j]
                else:
                    if x[i,j] != 0:
                        return False
        return cost

    def get_canon_task(self):
        shape = self.C.shape

        A_eq = False
        id_buf = 1
        for i in self.N:
            mat_buf = np.zeros(shape)
            mat_buf[i,] = np.ones(shape[1])
            if isinstance(A_eq, bool):
                A_eq = mat_buf
                id_buf = np.identity(max(shape))
            else:
                A_eq = np.concatenate((A_eq, mat_buf), axis=1)
                id_buf = np.concatenate((id_buf, np.identity(max(shape))), axis=1)
        A_eq = np.concatenate((A_eq, id_buf), axis=0)[np.arange(shape[0] + shape[1] - 1),]
        b_eq = np.concatenate((self.a, self.b))[np.arange(shape[0] + shape[1] - 1)]
        c_func = self.C.flatten()

        return A_eq, b_eq, c_func

    def get_dual_task_canon(self):
        c_func = -np.concatenate((-self.a,self.b))
        b_meq = -self.C.flatten()
        A_meq = False
        id_buf = 1
        shape = self.C.shape
        for i in self.N:
            mat_buf = np.zeros(shape)
            mat_buf[i,] = np.ones(shape[1])
            if isinstance(A_meq, bool):
                A_meq = mat_buf
                id_buf = np.identity(max(shape))
            else:
                A_meq = np.concatenate((A_meq, mat_buf), axis=1)
                id_buf = np.concatenate((id_buf, np.identity(max(shape))), axis=1)
        A_meq = -np.concatenate((-A_meq, id_buf), axis=0)[np.arange(shape[0] + shape[1]),].T

        A_canon = np.concatenate((A_meq, -A_meq, -np.identity(A_meq.shape[0])), axis=1)
        b_canon = b_meq
        c_canon = np.concatenate((c_func, -c_func, np.zeros(A_meq.shape[0])))

        return A_canon, b_canon, c_canon
