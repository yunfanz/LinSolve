import numpy as np
cimport numpy as np

#assert sizeof(int) == sizeof(np.int32_t)
#assert sizeof(np.float32) == sizeof(np.float32_t)

cdef extern from "Solver_manager.hh":
    cdef cppclass C_DnSolver "DnSolver":
        C_DnSolver(np.float32_t*, np.float32_t*, np.int32_t, np.int32_t)
        void solve()
        void retrieve_to(np.float32_t*)

cdef class DnSolver:
    cdef C_DnSolver* g
    cdef int rows
    cdef int cols

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t] arr, np.ndarray[ndim=1,dtype=np.float32_t] rhs, np.int32_t lda):
        self.rows, self.cols = lda, len(arr)/lda
        self.g = new C_DnSolver(&arr[0], &rhs[0], self.rows, self.cols)

    def solve(self):
        self.g.solve()


    def retrieve(self):
        cdef np.ndarray[ndim=1, dtype=np.float32_t] x = np.zeros(self.cols, dtype=np.float32)

        self.g.retrieve_to(&x[0])

        return x