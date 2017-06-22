import numpy as np
import gpusolver
from scipy.sparse import csr_matrix

m1 = np.array([[ 5.,  0.],
       [ 6., -1.]], dtype=np.float32)
b1 = np.ones(2, dtype=np.float32)
solver = gpusolver.DnSolver(np.int32(2), np.int32(2))
solver.from_dense(m1.flatten(order='F'), b1)
solver.solve()
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))

m1csr = csr_matrix(m1)

solver.from_csr(m1csr.indptr, m1csr.indices, m1csr.data, b1)
solver.solve()
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))

