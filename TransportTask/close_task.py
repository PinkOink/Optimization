import numpy as np

class Close_Task:

    def __init__(self, a, b, C):
        self.a = np.array(a)
        self.b = np.array(b)
        self.C = np.array(C)

    def get_closed_task(self):
        if sum(self.a) < sum(self.b):
            a = np.append(self.a, sum(self.b) - sum(self.a))
            b = np.array(self.b)
            C = np.concatenate((self.C, np.zeros((1, self.b.size))), axis=0)
            return a, b, C
        elif sum(self.a) < sum(self.b):
            a = np.array(self.a)
            b = np.append(self.b, sum(self.a) - sum(self.b))
            C = np.concatenate((self.C, np.zeros((self.a.size, 1))), axis=1)
            return a, b, C
        else:
            return self.a, self.b, self.C