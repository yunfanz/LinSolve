import numpy as np
import gpusolver
from scipy.sparse import csr_matrix

m1 = np.array([[ 5.,  0.],
       [ 6., -1.]], dtype=np.float32)
b1 = np.ones(2, dtype=np.float32)
solver = gpusolver.DnSolver(np.int32(2), np.int32(2))
solver.from_dense(m1.flatten(order='F'), b1)
solver.solve(0)
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))

solver.solve_Axb(0)
x = solver.retrieve()
print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))

m1csr = csr_matrix(m1)

solver.from_csr(m1csr.indptr, m1csr.indices, m1csr.data, b1)
for i in xrange(4):
	solver.solve(i)
	x = solver.retrieve()
	print 'Absolute Error', np.hypot(*(x-np.array([0.2,0.2])))


print "m2"
m2 = np.array([[ 5.,  0.,  1.],
       [ 6., -1.,  4.],
       [ 3.,  0.,  0.],
       [ 0.,  3.,  0.],
       [ 0., -2.,  0.],
       [ 0.,  6.,  0.],
       [ 5., -1.,  7.]]).astype(np.float32)
solver = gpusolver.DnSolver(np.int32(m2.shape[0]), np.int32(m2.shape[1]))

m1csr = csr_matrix(m2)
b2 = np.ones(m2.shape[0], dtype=np.float32)
#solver.from_dense(m2.flatten(order='F'), b2)
solver.from_csr(m1csr.indptr, m1csr.indices, m1csr.data, b2)
for i in xrange(4):
	solver.solve(i)
	x = solver.retrieve()
	print x
import IPython; IPython.embed()