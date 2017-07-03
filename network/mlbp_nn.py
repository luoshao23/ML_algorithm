import numpy as np
from sklearn.utils import check_random_state
from _base import ACTIVATIONS, DERIVATIVES


_SOLVER = ['sgd']


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
    learning_rules = {'const': lambda x, it: x,
                      'expo': lambda x, it: x * np.exp()}

    def __init__(self, activation='logistic', learning_rate_init=0.03, learning_rule='const', lamb=0.0, max_iter=200,
                 num_hidden_nodes=[100], num_hidden_layers=1, momentum=0.9, beta=0.0, ro0=0.05, shuffle=True, random_state=None):

        # self.nonlinear = (sigmoid, dsigmoid)
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.lamb = lamb
        self.momentum = momentum
        self.beta = beta
        self.ro0 = ro0
        self.max_iter = max_iter
        self.num_input_nodes = None
        self.num_res_nodes = None
        self.shuffle = shuffle
        if not isinstance(num_hidden_nodes, list):
            raise TypeError('must be a list!')
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.layers_ = num_hidden_layers + 2
        self._random_state = check_random_state(random_state)

        self.ww = None
        self.th = None
        self.predict_ = None

    def init_param(self, nodes_list):
        if self.activation == 'logistic':
            init_bound = lambda inb, outb: np.sqrt(2. / (inb + outb))
        else:
            init_bound = lambda inb, outb: np.sqrt(6. / (inb + outb))

        self.ww = [self._random_state.uniform(-init_bound(nodes_list[i], nodes_list[i + 1]), init_bound(nodes_list[i], nodes_list[i + 1]), (nodes_list[i], nodes_list[i + 1]))
                   for i in xrange(self.layers_ - 1)]

        self.th = [self._random_state.uniform(-init_bound(nodes_list[i], nodes_list[i + 1]), init_bound(nodes_list[i], nodes_list[i + 1]), (nodes_list[i + 1],))
                   for i in xrange(self.layers_ - 1)]

        self.dww = [np.empty_like(w) for w in self.ww]

        self.dww_last = [np.empty_like(w) for w in self.ww]

        self.dth = [np.empty_like(th) for th in self.th]

        self.z = [np.empty_like(th) for th in self.th]

        self.a = [np.empty_like(th) for th in self.th]

        self.ro = [np.empty_like(th) for th in self.th]

        self.delta = [np.empty_like(th) for th in self.th]

    def _forward_pass(self, activations):
        hidden_activation = ACTIVATIONS[self.activation]
        for layer in xrange(self.layers_ - 1):
            activations[
                layer + 1] = np.dot(activations[layer], self.ww[layer]) + self.th[layer]
            if (layer + 1) != (self.layers_ - 1):
                activations[
                    layer + 1] = hidden_activation(activations[layer + 1])
            else:
                out_activation = ACTIVATIONS[self.out_activation_]
                activations[layer + 1] = out_activation(activations[layer + 1])

        return activations

    def fit(self, records, results):
        hidden_activation = ACTIVATIONS[self.activation]
        inplace_derivative = DERIVATIVES[self.activation]

        num_input, self.num_input_nodes = records.shape
        num_res, self.num_res_nodes = results.shape
        self.input_ = records
        self.target_ = results

        if num_input != num_res:
            raise('Data set error!')

        nodes_list = [self.num_input_nodes] + \
            self.num_hidden_nodes + [self.num_res_nodes]

        # init the weights, thresholds
        self.init_param(nodes_list)

        for _ in xrange(self.max_iter):
            if self.shuffle:
                index = np.random.permutation(num_input)
                self.input_ = self.input_[index]
                self.target_ = self.target_[index]

            for i in xrange(num_input):
                for layer in xrange(self.layers_ - 1):
                    if layer == 0:
                        self.z[layer] = np.dot(self.input_[i], self.ww[
                                               layer]) + self.th[layer]
                        self.a[layer] = hidden_activation(self.z[layer])
                        self.ro[layer] = np.mean(
                            self.input_[i]) * self.a[layer]
                    else:
                        self.z[layer] = np.dot(
                            self.a[layer - 1], self.ww[layer]) + self.th[layer]
                        self.a[layer] = hidden_activation(self.z[layer])
                        self.ro[layer] = np.mean(
                            self.a[layer - 1]) * self.a[layer]
                    # print self.ro[layer]

                for rlayer in xrange(self.layers_ - 2, -1, -1):
                    if rlayer == self.layers_ - 2:
                        self.delta[
                            rlayer] = (-(self.target_[i] - self.a[rlayer]))

                    else:
                        self.delta[rlayer] = (
                            np.dot(self.ww[rlayer + 1], self.delta[rlayer + 1]))
                    inplace_derivative(self.z[rlayer], self.delta[rlayer])
                    # + self.beta * (-self.ro0 / self.ro[
                    # rlayer] + (1 - self.ro0) / (1 - self.ro[rlayer]))

                    if rlayer != 0:
                        self.dww[
                            rlayer] += np.outer(self.a[rlayer - 1], self.delta[rlayer])
                    else:
                        self.dww[
                            rlayer] += np.outer(self.input_[i], self.delta[rlayer])
                    self.dth[rlayer] += self.delta[rlayer]

            # print self.ww[0].shape == self.dww[0].shape
            tempmax = -999
            for layer in xrange(self.layers_ - 1):
                self.ww[layer] = self.ww[layer] - self.learning_rate_init * \
                    (self.dww[layer] / float(num_input) +
                     self.lamb * self.ww[layer] + self.momentum * self.dww_last[layer])

                self.dww_last[layer] = self.dww[layer] / float(num_input)

                tempmax = np.abs(self.dww[layer]).max() if np.abs(
                    self.dww[layer]).max() > tempmax else tempmax

                self.th[layer] = self.th[layer] - self.learning_rate_init * \
                    (self.dth[layer] / float(num_input))
            # print tempmax
            if tempmax < 0.01:
                break

    def predict(self, records):
        hidden_activation = ACTIVATIONS[self.activation]
        for layer in xrange(self.layers_ - 1):
            if layer == 0:
                self.predict_ = hidden_activation(
                    np.dot(records, self.ww[layer]) + self.th[layer])
            else:
                self.predict_ = hidden_activation(
                    np.dot(self.predict_, self.ww[layer]) + self.th[layer])
        # return (np.sign(self.predict_ - 0.5) + 1) / 2
        return self.predict_


def test():
    records = np.random.randint(0, 2, (20, 6))
    results = np.random.randint(0, 2, (20, 3))
    data = np.eye(6)
    nn = mynn()
    nn.fit(records, results)

    print results
    print nn.predict(records)

if __name__ == '__main__':
    test()
