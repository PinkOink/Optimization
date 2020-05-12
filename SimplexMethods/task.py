import numpy as np
import simplex_vectors as sm
import simplex_table as st


class CanonicalTask:

    def __init__(self, A, b, c):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)

    def get_dual(self):
        A_buf = -self.A.transpose()
        b_buf = -self.c
        c_buf = -self.b

        A = np.concatenate((A_buf, -A_buf, -np.identity(A_buf.shape[0])), axis=1)
        b = b_buf
        c = np.concatenate((c_buf, -c_buf, np.zeros(A_buf.shape[0])))
        return CanonicalTask(A, b, c)


class SymmetricalTask:

    def __init__(self, A, b, c):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)

    def get_canon(self):
        A = np.concatenate((self.A, -np.identity(self.A.shape[0])), axis=1)
        b = self.b
        c = np.concatenate((self.c, np.zeros(self.b.size)))

        return CanonicalTask(A, b, c)


class GeneralTask:

    def __init__(self, A_meq, A_eq, b_meq, b_eq, c, N1):
        self.A_meq = np.array(A_meq)
        self.A_eq = np.array(A_eq)
        self.b_meq = np.array(b_meq)
        self.b_eq = np.array(b_eq)
        self.c = np.array(c)
        self.N1 = np.array(N1)
        self.N = np.arange(self.c.size)
        self.N2 = np.setdiff1d(self.N, self.N1)

    def get_sym(self):
        A_1 = np.concatenate((self.A_meq[:,self.N1], self.A_meq[:,self.N2], -self.A_meq[:,self.N2]), axis=1)
        A_2 = np.concatenate((self.A_eq[:, self.N1], self.A_eq[:, self.N2], -self.A_eq[:, self.N2]), axis=1)
        A_3 = np.concatenate((-self.A_eq[:, self.N1], -self.A_eq[:, self.N2], self.A_eq[:, self.N2]), axis=1)
        A = np.concatenate((A_1, A_2, A_3), axis=0)
        b = np.concatenate((self.b_meq, self.b_eq, -self.b_eq))
        c = np.concatenate((self.c[self.N1], self.c[self.N2], -self.c[self.N2]))

        return SymmetricalTask(A, b, c)

    def get_solve_from_sym(self, vec):
        x = vec[np.arange(self.N1.size)]
        u = vec[np.arange(self.N1.size, self.N1.size + self.N2.size)]
        v = vec[np.arange(self.N1.size + self.N2.size, self.N1.size + self.N2.size + self.N2.size)]
        res = []
        i_res_1 = 0
        i_res_23 = 0
        for i in self.N:
            if i in self.N1:
                res.append(x[i_res_1])
                i_res_1 += 1
            else:
                res.append(u[i_res_23] - v[i_res_23])
                i_res_23 += 1
        return np.array(res)

    def get_canon(self):
        A_1 = np.concatenate((self.A_meq[:,self.N1], self.A_meq[:,self.N2], -self.A_meq[:,self.N2], -np.identity(self.A_meq.shape[0])), axis=1)
        A_2 = np.concatenate((self.A_eq[:,self.N1], self.A_eq[:,self.N2], -self.A_eq[:,self.N2], -np.zeros((self.A_eq.shape[0], self.A_meq.shape[0]))), axis=1)
        A = np.concatenate((A_1, A_2), axis=0)
        b = np.concatenate((self.b_meq, self.b_eq))
        c = np.concatenate((self.c[self.N1], self.c[self.N2], -self.c[self.N2], np.zeros(self.A_meq.shape[0])))

        return CanonicalTask(A, b, c)

    def get_solve_from_canon(self, vec):
        x = vec[np.arange(self.N1.size)]
        u = vec[np.arange(self.N1.size, self.N1.size + self.N2.size)]
        v = vec[np.arange(self.N1.size + self.N2.size, self.N1.size + self.N2.size + self.N2.size)]
        res = []
        i_res_1 = 0
        i_res_23 = 0
        for i in self.N:
            if i in self.N1:
                res.append(x[i_res_1])
                i_res_1 += 1
            else:
                res.append(u[i_res_23] - v[i_res_23])
                i_res_23 += 1
        return np.array(res)

    def solve_vectors(self):
        can_task = self.get_canon()
        solve = sm.SimplexVectors(can_task.A, can_task.b, can_task.c)
        solve.solve()
        self.x = solve.solution()
        if isinstance(self.x, bool):
            return self.x
        else:
            self.x = self.get_solve_from_canon(self.x)
            return self.x


class SuperGeneralTask:

    def __init__(self, A_meq, A_eq, A_leq, b_meq, b_eq, b_leq, c, N1):
        self.A_meq = np.array(A_meq)
        self.A_eq = np.array(A_eq)
        self.A_leq = np.array(A_leq)
        self.b_meq = np.array(b_meq)
        self.b_eq = np.array(b_eq)
        self.b_leq = np.array(b_leq)
        self.c = np.array(c)
        self.N1 = np.array(N1)
        self.x = False

    def get_general(self):
        A_meq = np.concatenate((self.A_meq, -self.A_leq), axis=0)
        b_meq = np.concatenate((self.b_meq, -self.b_leq))
        return GeneralTask(A_meq, self.A_eq, b_meq, self.b_eq, self.c, self.N1)

    def get_dual_gen(self):
        A_buf = np.concatenate((self.A_meq, -self.A_leq, self.A_eq), axis=0).transpose()
        b_buf = np.concatenate((self.b_meq, -self.b_leq, self.b_eq))

        A_meq_d = -A_buf[self.N1, ]
        A_eq_d = A_buf[np.setdiff1d(np.arange(self.c.size), self.N1), ]
        b_meq_d = -self.c[self.N1]
        b_eq_d = self.c[np.setdiff1d(np.arange(self.c.size), self.N1)]
        c_d = -b_buf
        N1_d = np.arange(self.A_meq.shape[0] + self.A_leq.shape[0])

        return GeneralTask(A_meq_d, A_eq_d, b_meq_d, b_eq_d, c_d, N1_d)

    def solve_vectors(self):
        gen_task = self.get_general()
        can_task = gen_task.get_canon()
        solve = sm.SimplexVectors(can_task.A, can_task.b, can_task.c)
        solve.solve()
        self.x = solve.solution()
        if isinstance(self.x, bool):
            return self.x
        else:
            self.x = gen_task.get_solve_from_canon(self.x)
            return self.x

    def solve_table(self):
        gen_task = self.get_general()
        can_task = gen_task.get_canon()
        solve = st.SimplexTable(can_task.A, can_task.b, can_task.c)
        solve.solve()
        self.x = solve.solution()
        if isinstance(self.x, bool):
            return self.x
        else:
            self.x = gen_task.get_solve_from_canon(self.x)
            return self.x