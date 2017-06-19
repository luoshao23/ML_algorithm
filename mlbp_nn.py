import numpy as np

records = np.random.randint(0, 2, (20, 6))
results = np.random.randint(0, 2, (20, 3))
# records = np.array([[0, 1, 1, 1, 1, 1],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 1, 1, 1],
#                     [1, 0, 1, 1, 1, 0],
#                     [1, 1, 0, 1, 1, 1],
#                     [0, 0, 1, 1, 0, 1],
#                     [1, 0, 1, 0, 1, 0],
#                     [1, 0, 0, 1, 1, 0],
#                     [1, 0, 1, 1, 0, 1],
#                     [0, 0, 1, 1, 1, 0],
#                     [1, 1, 0, 0, 0, 0],
#                     [0, 1, 1, 1, 1, 1],
#                     [1, 0, 0, 0, 1, 1],
#                     [0, 1, 1, 1, 0, 0],
#                     [1, 0, 1, 0, 1, 0],
#                     [1, 0, 0, 1, 0, 0],
#                     [0, 1, 1, 0, 0, 0],
#                     [1, 1, 0, 1, 0, 1],
#                     [1, 1, 1, 0, 1, 0],
#                     [0, 1, 1, 0, 1, 1]])

# results = np.array([[1, 1, 0],
#                     [1, 0, 0],
#                     [1, 0, 0],
#                     [0, 1, 1],
#                     [0, 1, 1],
#                     [0, 0, 1],
#                     [1, 0, 1],
#                     [0, 1, 1],
#                     [0, 0, 1],
#                     [0, 1, 1],
#                     [1, 1, 1],
#                     [1, 0, 0],
#                     [0, 1, 1],
#                     [1, 1, 0],
#                     [1, 0, 0],
#                     [1, 1, 1],
#                     [0, 0, 0],
#                     [0, 0, 0],
#                     [1, 0, 0],
#                     [0, 1, 1]])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y - y * y


def rectify(x):
    return x * (x > 0)


def drectify(x):
    x = x.copy()
    x[x > 0] = 1
    x[x < 0] = 0
    return x


class mynn(object):
    """docstring for mynn"""

    def __init__(self, eta=0.1, lamb=0.0, max_iter=500, num_hidden_nodes=[8], num_hidden_layers=1, momentum=0.8):

        self.nonlinear = (sigmoid, dsigmoid)
        self.eta = eta
        self.lamb = lamb
        self.momentum = momentum
        self.max_iter = max_iter
        self.num_input_nodes = None
        self.num_res_nodes = None
        if not isinstance(num_hidden_nodes, list):
            raise TypeError('must be a list!')
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.layers_ = num_hidden_layers + 2

        self.ww = None
        self.th = None
        self.predict_ = None

    def cal_output(self, layer):
        pass

    def fit(self, records, results):
        self.input = records
        num_input, self.num_input_nodes = records.shape
        num_res, self.num_res_nodes = results.shape

        if num_input != num_res:
            raise('Data set error!')

        nodes_list = [self.num_input_nodes] + \
            self.num_hidden_nodes + [self.num_res_nodes]

        # init the weights, thresholds
        self.ww = [np.random.uniform(-.05, .05, (nodes_list[i], nodes_list[i + 1]))
                   for i in xrange(self.layers_ - 1)]
        # print self.ww[0].shape, self.ww
        self.th = [np.random.uniform(-.05, .05, (nodes_list[i + 1],))
                   for i in xrange(self.layers_ - 1)]

        self.dww = [np.zeros((nodes_list[i], nodes_list[i + 1]))
                    for i in xrange(self.layers_ - 1)]

        self.dww_last = [np.zeros((nodes_list[i], nodes_list[i + 1]))
                    for i in xrange(self.layers_ - 1)]

        self.dth = [np.zeros((nodes_list[i + 1],))
                    for i in xrange(self.layers_ - 1)]

        self.z = [np.zeros((nodes_list[i + 1],))
                  for i in xrange(self.layers_ - 1)]

        self.a = [np.zeros((nodes_list[i + 1],))
                  for i in xrange(self.layers_ - 1)]

        self.delta = [np.zeros((nodes_list[i + 1],))
                      for i in xrange(self.layers_ - 1)]

        for _ in xrange(self.max_iter):

            for i in xrange(num_input):
                for layer in xrange(self.layers_ - 1):
                    if layer == 0:
                        self.z[layer] = np.dot(self.input[i], self.ww[
                                               layer]) + self.th[layer]
                    else:
                        self.z[layer] = np.dot(
                            self.a[layer - 1], self.ww[layer]) + self.th[layer]

                    self.a[layer] = self.nonlinear[0](self.z[layer])

                for rlayer in xrange(self.layers_ - 2, -1, -1):
                    if rlayer == self.layers_ - 2:
                        self.delta[
                            rlayer] = -(results[i] - self.a[rlayer]) * self.nonlinear[1](self.z[rlayer])

                    else:
                        self.delta[rlayer] = np.dot(
                            self.ww[rlayer + 1], self.delta[rlayer + 1]) * self.nonlinear[1](self.z[rlayer])

                    if rlayer != 0:
                        self.dww[
                            rlayer] += np.outer(self.a[rlayer - 1], self.delta[rlayer])
                    else:
                        self.dww[
                            rlayer] += np.outer(self.input[i], self.delta[rlayer])
                    self.dth[rlayer] += self.delta[rlayer]

            # print self.ww[0].shape == self.dww[0].shape
            tempmax = -999
            for layer in xrange(self.layers_ - 1):
                self.ww[layer] = self.ww[layer] - self.eta * \
                    (self.dww[layer] / float(num_input) +
                     self.lamb * self.ww[layer] + self.momentum*self.dww_last[layer])

                self.dww_last[layer] = self.dww[layer] / float(num_input)

                tempmax = np.abs(self.dww[layer]).max() if np.abs(
                    self.dww[layer]).max() > tempmax else tempmax

                self.th[layer] = self.th[layer] - self.eta * \
                    (self.dth[layer] / float(num_input))
            # print tempmax
            if tempmax < 0.01:
                break

    def predict(self, records):
        for layer in xrange(self.layers_ - 1):
            if layer == 0:
                self.predict_ = self.nonlinear[0](
                    np.dot(records, self.ww[layer]) + self.th[layer])
            else:
                self.predict_ = self.nonlinear[0](
                    np.dot(self.predict_, self.ww[layer]) + self.th[layer])
        # return (np.sign(self.predict_ - 0.5) + 1) / 2
        return (self.predict_>0.5)+0.0


def test():
    nn = mynn()
    nn.fit(records, results)
    print nn.dww

    print results
    print nn.predict(records)

if __name__ == '__main__':
    test()
