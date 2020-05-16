import numpy as np
import potentials as p
import simplex_vectors as sv
import transport_task as tt

#a = np.array([100,150,200,100])
#b = np.array([120,200,100,30,100])
#C = np.array([[0.7,0.5,0.6,0.9,0.5],
#              [0.4,0.5,0.8,0.8,1.0],
#              [0.3,0.2,0.5,0.4,0.4],
#              [0.9,1.1,1.0,0.8,1.1]])

a = np.array([37, 15, 25, 27])
b = np.array([27, 35, 12, 14, 16])
C = np.array([[7, 9, 5, 12, 11],
              [9, 11, 21, 2, 4],
              [12, 7, 9, 19, 13],
              [3, 1, 2, 5, 15]])
transport_task = tt.TransportTask(a, b, C)


print("Potentials method:")
potential = p.MethodOfPotentials(a, b, C)

potential.solve("NW")
print("NW start")
print("x=", potential.solution())
print("Func val=", transport_task.check_solution(potential.solution()))
print()

potential.solve("min")
print("Min start")
print("x=", potential.solution())
print("Func val=", transport_task.check_solution(potential.solution()))
print()


print("Simplex method:")
A_eq, b_eq, c_func = transport_task.get_canon_task()
solve = sv.SimplexVectors(A_eq, b_eq, c_func)
solve.solve()
print("x=", solve.solution().reshape(C.shape))
print("Func val=", transport_task.check_solution(solve.solution().reshape(C.shape)))
print()


print("Dual task, simplex method:")
A_dual, b_dual, c_dual = transport_task.get_dual_task_canon()
solve = sv.SimplexVectors(A_dual, b_dual, c_dual)
solve.solve()
print("Func val=", -c_dual.dot(solve.solution()))
print()



print("With embargo")
C[0,0] = 10000
C[3,1] = 10000
C[1,4] = 10000
transport_task = tt.TransportTask(a, b, C)


print("Potential method:")
potential = p.MethodOfPotentials(a, b, C)

potential.solve("NW")
print("NW start")
print("x=", potential.solution())
print("Func val=", transport_task.check_solution(potential.solution()))
print()

potential.solve("min")
print("Min start")
print("x=", potential.solution())
print("Func val=", transport_task.check_solution(potential.solution()))
print()


print("Simplex method:")
A_eq, b_eq, c_func = transport_task.get_canon_task()
solve = sv.SimplexVectors(A_eq, b_eq, c_func)
solve.solve()
print("x=", solve.solution().reshape(C.shape))
print("Func val=", transport_task.check_solution(solve.solution().reshape(C.shape)))
print()


print("Dual task, simplex method:")
A_dual, b_dual, c_dual = transport_task.get_dual_task_canon()
solve = sv.SimplexVectors(A_dual, b_dual, c_dual)
solve.solve()
print("Func val=", -c_dual.dot(solve.solution()))
print()