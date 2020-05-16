import numpy as np
import math

class MethodOfPotentials:

    def __init__(self, a, b, C):
        self.a = np.array(a)
        self.b = np.array(b)
        self.C = np.array(C)
        self.x = np.zeros(C.shape)

    def solve(self, method):
        s = self.Solver(self.a, self.b, self.C)
        if method == "NW":
            s.method_NW_corner()
        elif method == "min":
            s.method_min_elem()
        else:
            self.x = False
            return
        self.x = s.solve()

    def solution(self):
        return self.x

    class Solver:

        def __init__(self, a, b, C):
            self.a = np.array(a)
            self.b = np.array(b)
            self.C = np.array(C)
            self.x = np.zeros(C.shape)
            self.used = np.full(self.C.shape, False, bool)
            self.N = np.arange(self.a.size)
            self.M = np.arange(self.b.size)

        def method_NW_corner(self):
            i_pos = 0
            j_pos = 0

            a_buf = np.array(self.a)
            b_buf = np.array(self.b)

            for n in np.arange(self.a.size + self.b.size - 1):
                self.x[i_pos,j_pos] = min(a_buf[i_pos], b_buf[j_pos])
                self.used[i_pos, j_pos] = True
                a_buf[i_pos] -= self.x[i_pos,j_pos]
                b_buf[j_pos] -= self.x[i_pos,j_pos]
                if b_buf[j_pos] == 0:
                    j_pos += 1
                else:
                    i_pos += 1
            return

        def method_min_elem(self):
            used_buf = np.full(self.C.shape, False, bool)
            a_buf = np.array(self.a)
            b_buf = np.array(self.b)

            for n in np.arange(self.a.size + self.b.size - 1):
                min_pos = (0,0)
                min_el = math.inf
                for i_pos in self.N:
                    for j_pos in self.M:
                        if not used_buf[i_pos, j_pos]:
                            if self.C[i_pos, j_pos] < min_el:
                                min_pos = (i_pos, j_pos)
                                min_el = self.C[i_pos, j_pos]
                self.x[min_pos[0],min_pos[1]] = min(a_buf[min_pos[0]], b_buf[min_pos[1]])
                self.used[min_pos[0],min_pos[1]] = True
                a_buf[min_pos[0]] -= self.x[min_pos[0],min_pos[1]]
                b_buf[min_pos[1]] -= self.x[min_pos[0],min_pos[1]]
                if a_buf[min_pos[0]] == 0:
                    used_buf[min_pos[0],] = True
                else:
                    used_buf[:,min_pos[1]] = True
            return

        def _find_S(self):
            S_u = []
            S_v = []
            for i_pos in self.N:
                for j_pos in self.M:
                    if self.used[i_pos, j_pos]:
                        S_u.append(i_pos)
                        S_v.append(j_pos)
            return np.array((S_u, S_v))

        def _find_u_v(self, S):
            visited = np.zeros(self.a.size + self.b.size)
            visited[0] = 1
            barrier = self.a.size
            u = np.zeros(self.a.size)
            v = np.zeros(self.b.size)
            stack = [0]
            while stack:
                cur = stack.pop()
                if cur < barrier:
                    for el in S.T:
                        if el[0] == cur:
                            if visited[el[1] + barrier] == 0:
                                stack.append(el[1] + barrier)
                                visited[el[1] + barrier] = 1
                                v[el[1]] = self.C[el[0],el[1]] + u[cur]
                else:
                    for el in S.T:
                        if el[1] == (cur - barrier):
                            if visited[el[0]] == 0:
                                stack.append(el[0])
                                visited[el[0]] = 1
                                u[el[0]] = v[cur - barrier] - self.C[el[0],el[1]]
            return u,v

        def _get_delta_min(self, u, v):
            delta = np.zeros(self.C.shape)
            min_val = math.inf
            min_pos = (-1,-1)
            for i in self.N:
                for j in self.M:
                    delta[i,j] = self.C[i,j] - (v[j] - u[i])
                    if delta[i,j] < min_val:
                        min_pos = (i,j)
                        min_val = delta[i,j]
            return min_val, min_pos

        def _find_cycle(self, S, pos):

            def _dfs(cur, cur_ind):
                if cur < barrier:
                    for i in S_ind:
                        if i != cur_ind:
                            if S[0][i] == cur:
                                if used_S[i] == 1:
                                    path.append(cur_ind)
                                    return True
                                else:
                                    used_S[i] = 1
                                    if _dfs(barrier + S[1][i], i) == True:
                                        path.append(cur_ind)
                                        return True
                                    used_S[i] = 0
                else:
                    for i in S_ind:
                        if i != cur_ind:
                            if S[1][i] == (cur - barrier):
                                if used_S[i] == 1:
                                    path.append(cur_ind)
                                    return True
                                else:
                                    used_S[i] = 1
                                    if _dfs(S[0][i], i) == True:
                                        path.append(cur_ind)
                                        return True
                                    used_S[i] = 0
                return False

            barrier = self.a.size
            S_ind = np.arange(S.shape[1])
            used_S = np.zeros(S.shape[1])
            used_S[S.shape[1] - 1] = 1
            path = []
            _dfs(pos[0], S.shape[1] - 1)
            return np.array(path)

        def _recount(self, S, path, new_pos):
            path_ind = np.arange(path.size)

            #for i in path_ind:
            #    if i % 2:
            #        signs[i] = 1
            #    else:
            #        signs[i] = -1

            min_val = math.inf
            min_pos = (-1,-1)
            for i in path_ind:
                if (i % 2) == 0:
                    if self.x[S[0,path[i]],S[1,path[i]]] < min_val:
                        min_val = self.x[S[0,path[i]],S[1,path[i]]]
                        min_pos = (S[0,path[i]],S[1,path[i]])

            for i in path_ind:
                if i % 2:
                    self.x[S[0,path[i]],S[1,path[i]]] += min_val
                else:
                    self.x[S[0,path[i]],S[1,path[i]]] -= min_val

            self.used[min_pos[0],min_pos[1]] = False
            self.used[new_pos[0],new_pos[1]] = True

            return

        def step(self):
            S = self._find_S()
            u, v = self._find_u_v(S)

            min_val, min_pos = self._get_delta_min(u, v)
            if min_val >= 0:
                return False
            S = np.concatenate((S.T, [min_pos]), axis=0).T
            path = self._find_cycle(S, min_pos)
            self._recount(S, path, min_pos)
            return True

        def solve(self):
            flag = self.step()
            while flag:
                flag = self.step()
            return self.x