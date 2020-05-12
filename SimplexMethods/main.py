import task as t
import numpy as np
import simplex_table as st


A_meq = np.array([[12, 1, 4, 0, -7]])
A_leq = np.array([[3, 3, 10, -27, 9]])
A_eq = np.array([[8, 0, 0, 13, -4],
                 [0, 3, 1, -3, -8],
                 [9, 9, 3, 11, -4]])

b_meq = np.array([-296])
b_leq = np.array([1115])
b_eq = np.array([-261, -518, -92])

c = np.array([-1, 11, 3, 5, 7])

N1 = np.array([1, 2, 3, 4])

#x = np.array([-2, 1, 56, 3, 71])


task = t.SuperGeneralTask(A_meq, A_eq, A_leq, b_meq, b_eq, b_leq, c, N1)
ans = task.solve_vectors()
print("Vectors regular solve")
print(ans)
print(c.dot(ans))
print(A_meq.dot(ans), b_meq)
print(A_leq.dot(ans), b_leq)
print(A_eq.dot(ans), b_eq)
print()


dual = task.get_dual_gen()
ans = dual.solve_vectors()
print("Vectors dual solve")
print(ans)
print(-dual.c.dot(ans))
print(dual.A_meq.dot(ans), dual.b_meq)
print(dual.A_eq.dot(ans), dual.b_eq)
print()


ans = task.solve_vectors()
print("Table regular solve")
print(ans)
print(c.dot(ans))
print(A_meq.dot(ans), b_meq)
print(A_leq.dot(ans), b_leq)
print(A_eq.dot(ans), b_eq)
print()