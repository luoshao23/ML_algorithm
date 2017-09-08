import numpy as np
from sklearn.utils import check_random_state, gen_batches
from scipy.special import expit as logistic_sigmoid
import warnings
from _base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS


class mynn(object):
    """docstring for mynn"""

    def __init__(self, activation='relu', learning_rate_init=0.001, learning_rule='const', lamb=0.0, max_iter=200,
                 num_hidden_nodes=[8], num_hidden_layers=1, momentum=0.9, beta=0.0, ro0=0.05, shuffle=True, batch_size="auto", random_state=None):

        # self.nonlinear = (sigmoid, dsigmoid)
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.lamb = lamb
        self.max_iter = max_iter
        self.num_input_nodes = None
        self.num_res_nodes = None
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not isinstance(num_hidden_nodes, list):
            raise TypeError('must be a list!')
        self.num_hidden_nodes = num_hidden_nodes
        self.num_hidden_layers = num_hidden_layers
        self.n_layers_ = num_hidden_layers + 2
        self._random_state = check_random_state(random_state)

        self.ww = None
        self.th = None


    def init_param(self, nodes_list):
        self.out_activation_ = 'identity'
        if self.activation == 'logistic':
            init_bound = lambda inb, outb: np.sqrt(2. / (inb + outb))
        else:
            init_bound = lambda inb, outb: np.sqrt(6. / (inb + outb))

        self.ww = [self._random_state.uniform(-init_bound(nodes_list[i], nodes_list[i + 1]), init_bound(nodes_list[i], nodes_list[i + 1]), (nodes_list[i], nodes_list[i + 1]))
                   for i in xrange(self.n_layers_ - 1)]

        self.th = [self._random_state.uniform(-init_bound(nodes_list[i], nodes_list[i + 1]), init_bound(nodes_list[i], nodes_list[i + 1]), (nodes_list[i + 1],))
                   for i in xrange(self.n_layers_ - 1)]

        # self.dww = [np.empty_like(w) for w in self.ww]

        # self.dww_last = [np.empty_like(w) for w in self.ww]

        # self.dth = [np.empty_like(th) for th in self.th]

        # self.z = [np.empty_like(th) for th in self.th]

        # self.a = [np.empty_like(th) for th in self.th]

        # self.ro = [np.empty_like(th) for th in self.th]

        # self.delta = [np.empty_like(th) for th in self.th]

    def _forward_pass(self, activations):
        hidden_activation = ACTIVATIONS[self.activation]
        for layer in xrange(self.n_layers_ - 1):
            activations[layer + 1] = np.dot(activations[layer], self.ww[layer])
            activations[layer + 1] += self.th[layer]

            if (layer + 1) != (self.n_layers_ - 1):
                activations[
                    layer + 1] = hidden_activation(activations[layer + 1])

        out_activation = ACTIVATIONS[self.out_activation_]
        activations[layer + 1] = out_activation(activations[layer + 1])

        return activations

    def _backprog(self, X, y, dww, dth, delta, activations):
        n_inputs = X.shape[0]
        activations = self._forward_pass(activations)

        derivative = DERIVATIVES[self.activation]
        for rlayer in xrange(self.n_layers_ - 2, -1, -1):
            if rlayer == self.n_layers_ - 2:
                delta[rlayer] = activations[rlayer + 1] - y

            else:
                delta[rlayer] = np.dot(
                    delta[rlayer + 1], self.ww[rlayer + 1].T)
            derivative(activations[rlayer + 1], delta[rlayer])
            # + self.beta * (-self.ro0 / self.ro[
            # rlayer] + (1 - self.ro0) / (1 - self.ro[rlayer]))
            dww[rlayer] = np.dot(activations[rlayer].T, delta[rlayer])
            dww[rlayer] += (self.lamb * self.ww[rlayer])
            dww[rlayer] /= n_inputs

            dth[rlayer] = np.mean(delta[rlayer], 0)

        return dww, dth

    def _fit(self, X, y):
        num_input, self.num_input_nodes = X.shape
        num_res, self.num_res_nodes = y.shape

        if num_input != num_res:
            raise('Data set error!')

        nodes_list = [self.num_input_nodes] + \
            self.num_hidden_nodes + [self.num_res_nodes]

        self.init_param(nodes_list)
        if self.batch_size == 'auto':
            batch_size = min(200, num_input)
        else:
            batch_size = self.batch_size

        activations = [X]
        activations.extend( [np.empty((batch_size, n_out_node))
                             for n_out_node in nodes_list[1:]])

        activations = self._forward_pass(activations)

        delta = [np.empty_like(a_layer) for a_layer in activations]
        dww = [np.empty_like(w) for w in self.ww]
        dth = [np.empty_like(th) for th in self.th]

        for it in xrange(self.max_iter):
            if self.shuffle:
                index = np.random.permutation(num_input)
                X = X[index]
                y = y[index]
            for batch in gen_batches(num_input, batch_size):
                activations[0] = X[batch]
                dww, dth = self._backprog(
                    X[batch], y[batch], dww, dth, delta, activations)
                for layer in xrange(self.n_layers_ - 1):

                    self.ww[layer] -= self.learning_rate_init * dww[layer]
                    self.th[layer] -= self.learning_rate_init * dth[layer]

        return activations[-1]

    def _predict(self, X):
        num_input, num_input_nodes = X.shape
        nodes_list = [num_input_nodes] + \
            self.num_hidden_nodes + [self.num_res_nodes]
        activations = [X]
        activations.extend([np.empty((num_input, n_out_node))
                             for n_out_node in nodes_list[1:]])

        self._forward_pass(activations)

        return (np.sign(activations[-1]-0.5)+1)/2



def test():
    from sklearn.neural_network import MLPClassifier
    records = np.random.randint(0, 2, (10, 6))
    results = np.random.randint(0, 2, (10, 3))
    # records = np.eye(6)
    # results = records

    nn = mynn()
    nn2 = MLPClassifier()

    nn._fit(records, results)
    nn2.fit(records, results)
    print results
    print nn._predict(records)
    print nn2.predict(records)
    # print nn.ww

    # print results
    # print nn.predict(records)

if __name__ == '__main__':
    test()
