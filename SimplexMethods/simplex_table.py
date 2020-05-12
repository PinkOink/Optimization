import numpy as np
import utils as u
import math

class SimplexTable:

    def __init__(self, A, b, c):
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.N = np.arange(self.c.size)
        self.M = np.arange(self.b.size)
        for i in self.M:
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i,] = -self.A[i,]
        return

    def solve(self):
        I_buf = np.identity(self.M.size)

        basis = np.zeros(self.M.size)

        art_basis_count = 0
        #art_basis = []

        coefs = np.array(self.A)
        plan_col = np.array(self.b, dtype=float)

        art_basis_row = np.zeros(self.N.size)
        art_basis_col = 0.0

        for j in self.M:
            get = False
            for i in self.N:
                if (I_buf[:,j] == coefs[:,i]).all():
                    basis[j] = i
                    get = True
                    break
            if get == False:
                new_col = np.reshape(I_buf[:,j], (self.M.size, 1))
                coefs = np.concatenate((coefs, new_col), axis=1)
                basis[j] = art_basis_count + self.N.size
                #art_basis.append(art_basis_count + self.N.size)
                art_basis_count += 1

                art_basis_row -= coefs[j,self.N]
                art_basis_col -= self.b[j]

        art_basis_row = np.concatenate((art_basis_row, np.zeros(art_basis_count)))

        func_row = np.concatenate((self.c, np.zeros(art_basis_count)))
        func_col = 0.0

        while not u._check_above_equals_zero(art_basis_row):
            to_basis = 0
            for i in self.N:
                if art_basis_row[i] < 0:
                    if art_basis_row[i] < art_basis_row[to_basis]:
                        to_basis = i


            min = math.inf
            out_basis = 0
            for i in self.M:
                if abs(coefs[i,to_basis]) > 1e-10:
                    new_min = plan_col[i] / coefs[i,to_basis]
                    if new_min > 1e-10:
                        if new_min < min:
                            min = new_min
                            out_basis = i

            if min == math.inf:
                return True

            lead_elem = coefs[out_basis, to_basis]

            for i in self.M:
                if i != out_basis:
                    plan_col[i] -= plan_col[out_basis] * coefs[i,to_basis] / lead_elem
            func_col -= plan_col[out_basis] * func_row[to_basis] / lead_elem
            art_basis_col -= plan_col[out_basis] * art_basis_row[to_basis] / lead_elem
            plan_col[out_basis] /= lead_elem

            for i in self.M:
                if i != out_basis:
                    coefs[i,] -= coefs[out_basis,] * (coefs[i,to_basis] / lead_elem)
            func_row -= func_row[to_basis] * coefs[out_basis,] / lead_elem
            func_row[to_basis] = 0
            art_basis_row -= art_basis_row[to_basis] * coefs[out_basis,] / lead_elem
            art_basis_row[to_basis] = 0

            basis[out_basis] = to_basis

            coefs[out_basis,] /= lead_elem
            coefs[:,to_basis] = 0
            coefs[out_basis,to_basis] = 1

            for i in self.M:
                if plan_col[i] < 0:
                    plan_col[i] = -plan_col[i]
                    coefs[i,] = -coefs[i,]

        coefs = coefs[:,self.N]
        func_row = func_row[self.N]


        while not u._check_above_equals_zero(func_row):
            to_basis = 0
            #for i in self.N:
            #    if func_row[i] < 0:
            #        if func_row[i] < func_row[to_basis]:
            #            to_basis = i

            # Blend's rule
            for i in self.N:
                if func_row[i] < 0:
                    to_basis = i
                    break


            min = math.inf
            out_basis = 0
            for i in self.M:
                if abs(coefs[i,to_basis]) > 1e-10:
                    new_min = plan_col[i] / coefs[i,to_basis]
                    if new_min > 1e-10:
                        if new_min < min:
                            min = new_min
                            out_basis = i

            if min == math.inf:
                return True

            lead_elem = coefs[out_basis, to_basis]

            for i in self.M:
                if i != out_basis:
                    plan_col[i] -= plan_col[out_basis] * coefs[i,to_basis] / lead_elem
            func_col -= plan_col[out_basis] * func_row[to_basis] / lead_elem
            plan_col[out_basis] /= lead_elem

            for i in self.M:
                if i != out_basis:
                    coefs[i,] -= coefs[out_basis,] * (coefs[i,to_basis] / lead_elem)
            func_row -= func_row[to_basis] * coefs[out_basis,] / lead_elem
            func_row[to_basis] = 0

            basis[out_basis] = to_basis

            coefs[out_basis,] /= lead_elem
            coefs[:,to_basis] = 0
            coefs[out_basis,to_basis] = 1

            for i in self.M:
                if plan_col[i] < 0:
                    plan_col[i] = -plan_col[i]
                    coefs[i,] = -coefs[i,]

        ans = np.zeros(self.N.size)
        for i in self.M:
            ans[int(basis[i])] = plan_col[i]

        self.x = ans


    def solution(self):
        return self.x