#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True


cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
    with gil:
        raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int k, d
    cdef int D = nd.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]
    cdef double ll = 0

    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)

    return ll