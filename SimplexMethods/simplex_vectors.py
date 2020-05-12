import numpy as np
import numpy.linalg as la
import math
import utils as u


class SimplexVectors:

    class SimplexAlgorithm:

        def __init__(self, A, b, c, Nk, Bk, x):
            self.A = np.array(A)
            self.b = np.array(b)
            self.c = np.array(c)
            self.Nk = np.array(Nk)
            self.N = np.arange(c.size)
            self.M = np.arange(b.size)
            self.Bk = np.array(Bk)
            self.xk = np.array(x)

        def _ind_below_zero(self, vec):
            ind = []
            for i in range(0, vec.size):
                if vec[i] < -1e-10:
                    ind.append(i)
            return np.array(ind)

        def _ind_above_zero(self, vec):
            ind = []
            for i in range(0, vec.size):
                if vec[i] > 1e-10:
                    ind.append(i)
            return np.array(ind)

        def _ind_equals_zero(self, vec):
            ind = []
            for i in range(0, vec.size):
                if abs(vec[i]) < 1e-10:
                    ind.append(i)
            return np.array(ind)

        def _get_dk(self):
            #print(la.matrix_rank(self.A[:,self.Nk]))
            Lk = np.setdiff1d(self.N, self.Nk)
            c1 = self.c[Lk]
            c2 = self.c[self.Nk]
            A = self.A[:,Lk]
            return c1 - c2.dot(self.Bk.dot(A)), Lk

        def _get_uk(self, jk):
            uk = np.zeros(self.N.size)
            uk[jk] = -1
            for i in np.arange(self.Nk.size):
                uk[self.Nk[i]] = self.Bk[i, ].dot(self.A[:,jk])
            return uk

        def _get_theta(self, uk):
            NkPlus = self._ind_above_zero(self.xk)
            theta = math.inf
            ik = -1

            if np.array_equal(self.Nk, NkPlus):
                for i in self.Nk:
                    if uk[i] > 1e-10:
                        if theta > (self.xk[i] / uk[i]):
                            ik = i
                            theta = self.xk[i] / uk[i]
                return theta, ik
            else:
                uk_buf = u._vector_by_index(uk, np.setdiff1d(self.Nk, NkPlus))
                if u._check_below_equals_zero(uk_buf):
                    for i in self.Nk:
                        if uk[i] > 1e-10:
                            if theta > (self.xk[i] / uk[i]):
                                ik = i
                                theta = self.xk[i] / uk[i]
                    return theta, ik
                else:
                    return False, False

        def _change_basis(self):
            Lk = np.setdiff1d(self.N, self.Nk)
            NkPlus = self._ind_above_zero(self.xk)
            Nk0 = np.setdiff1d(self.Nk, NkPlus)
            for i in Nk0:
                for j in Lk:
                    ind_buf = np.sort(np.append(self.Nk[self.Nk != i], j))
                    if la.det(u._matrix_by_index(self.A, self.M, ind_buf)) != 0:
                        self.Nk = ind_buf
                        return
            print("Зациклились")
            return

        def _update_Bk_Nk(self, ik, jk, uk):
            N_buf = np.arange(self.Nk.size)
            ik_pos, = np.where(self.Nk == ik)
            self.Nk = self.Nk[self.Nk != ik]
            self.Nk = np.append(self.Nk, jk)
            self.Nk = np.sort(self.Nk)
            uk_buf = u._vector_by_index(uk, self.Nk)
            jk_pos, = np.where(self.Nk == jk)
            I = np.identity(N_buf.size)
            if jk_pos < ik_pos:
                I = np.insert(I, ik_pos + 1, 0, axis=1)
                I[:,ik_pos + 1] = np.transpose([-(uk_buf / uk[ik])])
                F = np.delete(I, jk_pos, axis=1)
                Bk = F.dot(self.Bk)
            else:
                I = np.delete(I, jk_pos, axis=1)
                F = np.insert(I, ik_pos, 0, axis=1)
                F[:, ik_pos] = np.transpose([-(uk_buf / uk[ik])])
                Bk = F.dot(self.Bk)
            self.Bk = Bk

        def _step(self):
            dk, Lk = self._get_dk()
            if u._check_above_equals_zero(dk):
                return True
            else:
                jk = 0
                for i in np.arange(Lk.size):
                    if dk[i] < -1e-10:
                        jk = Lk[i]
                        break
                uk = self._get_uk(jk)
                if u._check_below_equals_zero(uk[self.Nk]):
                    self.xk = True
                    return True
                else:
                    theta, ik = self._get_theta(uk)
                    if isinstance(theta, bool):
                        self._change_basis()
                        return False
                    else:
                        self.xk = self.xk - theta * uk
                        self._update_Bk_Nk(ik, jk, uk)
                        #print("Check conds2")
                        #print(self.A.dot(self.xk))
                        #print(self.b)
                        #print("Improvement", self.c.dot(self.xk))
                        #print("Rank and det", la.matrix_rank(self.A[:,self.Nk]), la.det(self.A[:,self.Nk]))
                        return False

        def _solve(self):
            brek = self._step()
            while brek != True:
                brek = self._step()
                #print("Check conds")
                #print(self.A.dot(self.xk))
                #print(self.b)
            return self.xk, self.Nk, self.Bk

    def artificial_basis(self):
        for i in self.M:
            if self.b[i] < 0:
                self.A[i,] = -self.A[i,]
                self.b[i] = -self.b[i]

        A_task = np.concatenate((self.A, np.identity(self.M.size)), axis=1)
        B_task = np.identity(self.M.size)
        c_task = np.concatenate((np.zeros(self.N.size), np.ones(self.M.size)))
        b_task = np.array(self.b)
        x_start = np.concatenate((np.zeros(self.N.size), b_task))
        Nk = np.arange(self.N.size, self.N.size + self.M.size)
        st = self.SimplexAlgorithm(A_task, b_task, c_task, Nk, B_task, x_start)
        return st._solve()

    def __init__(self, A, b, c):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.N = np.arange(c.size)
        self.M = np.arange(b.size)

    def solve(self):
        ans, Nk, Bk = self.artificial_basis()

        if isinstance(ans, bool):
            self.xk = ans
            return

        y = np.delete(ans, self.N)
        xk = np.delete(ans, np.arange(self.N.size, self.N.size + self.M.size))

        #I = u._matrix_by_index(A, M, Nk).dot(Bk)

        if u._check_any_above_zero(y):
            self.x = False
        else:
            st = self.SimplexAlgorithm(self.A, self.b, self.c, Nk, Bk, xk)
            self.x = st._solve()[0]

    def solution(self):
        return self.x