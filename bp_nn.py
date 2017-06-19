import numpy as np

records = np.random.randint(0, 2, (20, 6))
results = np.random.randint(0, 2, (20, 3))

# initialization
# configuration


def acfun(x):
    return 1.0 / (1.0 + np.exp(-x))


class mynn(object):
    """docstring for mynn"""

    def __init__(self, eta=0.3, max_iter=100, num_hidden_nodes=12):
        super(mynn, self).__init__()
        self.eta = eta
        self.max_iter = max_iter
        self.num_hidden_nodes = num_hidden_nodes
        self.vw_in = None
        self.th_in = None
        self.ww_h = None
        self.th_h = None

    def fit(self, records, results):
        num_input, self.num_input_nodes = records.shape
        num_res, self.num_res_nodes = results.shape

        # self.num_input_nodes  len=d
        # self.num_res_nodes    len=l
        # self.num_hidden_nodes len=q

        # init input weight and threshold
        self.vw_in = np.random.random(
            (self.num_input_nodes, self.num_hidden_nodes))  # d*q
        self.th_in = np.random.random((self.num_hidden_nodes, ))  # q

        # init hidden weight and threshold
        self.ww_h = np.random.random(
            (self.num_hidden_nodes, self.num_res_nodes))  # q*l
        self.th_h = np.random.random((self.num_res_nodes, ))  # l

        for it in xrange(self.max_iter):
            # iteration loop
            for i in xrange(num_input):
                # record loop
                vsigmoid = np.vectorize(acfun)
                b = vsigmoid(np.dot(records[i], self.vw_in) + self.th_in) #q
                y_cal = vsigmoid(np.dot(b, self.ww_h) + self.th_h) #l
                g = y_cal * (1 - y_cal) * (results[i] - y_cal) #l
                e = b*(1-b)*np.dot(self.ww_h, g) #q

                dww_h = self.eta*np.outer(b, g)
                dth_h = -self.eta*g
                dvw_in = self.eta*np.outer(records[i],e)
                dth_in = -self.eta*e

                self.ww_h += dww_h
                self.th_h += dth_h
                self.vw_in += dvw_in
                self.th_in += dth_in
                # e = cal_eh(g, ww_h)
                # update_nn()
        # print 'done!'

    def predict(self, records):
        if self.vw_in is not None and self.th_in is not None and self.ww_h is not None and self.th_h is not None and records is not None:
            # do predict
            vsigmoid = np.vectorize(acfun)
            b = vsigmoid(np.dot(records, self.vw_in) + self.th_in)
            y_cal = vsigmoid(np.dot(b, self.ww_h) + self.th_h)
            return (np.sign(y_cal-0.5)+1)/2
        else:
            raise("Not fit yet!")


def test():
    nn = mynn()
    nn.fit(records, results)
    k = nn.predict(records)
    print k
    print '==='
    print results


if __name__ == '__main__':
    test()
