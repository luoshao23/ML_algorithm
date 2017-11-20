import numpy as np


X = [[2, 0, 0, 2, 0],
     [0, 0, 2, 2, 2],
     [1, 1, 1, 2, 1],
     [2, 2, 0, 1, 1],
     [0, 2, 2, 0, 2],
     [2, 0, 2, 0, 0],
     [0, 2, 2, 0, 1],
     [0, 1, 2, 1, 1],
     [0, 2, 0, 2, 2],
     [2, 2, 2, 2, 2],
     [0, 2, 0, 1, 2],
     [0, 0, 2, 2, 2],
     [1, 2, 0, 2, 0],
     [1, 0, 1, 1, 1],
     [2, 1, 0, 0, 1],
     [0, 1, 2, 0, 0],
     [1, 2, 1, 0, 0],
     [1, 2, 2, 1, 0],
     [0, 0, 1, 2, 0],
     [0, 0, 0, 0, 0],
     [0, 2, 1, 2, 2],
     [2, 0, 1, 2, 0],
     [0, 0, 2, 0, 1],
     [0, 1, 1, 2, 1],
     [1, 2, 1, 0, 2],
     [1, 2, 1, 1, 1],
     [0, 0, 2, 0, 2],
     [2, 2, 0, 0, 2],
     [2, 1, 0, 1, 1],
     [1, 1, 1, 1, 0]]

y = [1, 1, 0, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 0,
     0, 1, 1, 2, 1, 2, 2]


class Tree(object):
    """docstring for Tree"""

    def __init__(self, col=None):
        self.col = col
        self.cost_fun = self.get_entropy

    def get_entropy(self, values):
        """
        Parameters
        ----------
        values: a series of values

        Returns
        ----------
        entropy: entropy of the values
        """
        unique_val, counts = np.unique(values, return_counts=True)
        prob = counts / float(len(values))
        entropy = - sum(prob * np.log(prob))
        return entropy

    def choose_best_feature(self, X, y):
        """
        Parameters
        ----------
        X: data
        y: target

        Returns
        ----------
        best_index: index of the best feature
        best_split: sub dataset split by best feature
        """
        delta = 0
        best_index = -1  # np.random.choice(range(len(self.col)))
        best_split = None
        base_cost = self.cost_fun(y)
        for col in range(X.shape[1]):
            # try:
            sub_dict = self.split_data(X, y, col)
            # except:
            #     print X, y, col

            CONSTANT = 1
            temp_cost = sum([len(sub_target) / float(len(y)) *
                             self.cost_fun(sub_target) for _, sub_target in sub_dict.values()])
            iv_list = [(len(sub_target)) / float(len(y) + min(CONSTANT, len(y))) * np.log(
                (len(sub_target)) / float(len(y) + min(CONSTANT, len(y)))) for _, sub_target in sub_dict.values()]
            iv = -sum(iv_list)
            gain_ratio = (base_cost - temp_cost) / iv

            if delta < gain_ratio:
                delta = gain_ratio
                best_index = col
                best_split = sub_dict

        return best_index, best_split

    def split_data(self, X, y, i):
        sub_dict = {}

        unique_val = np.unique(X[:, i])

        c = range(i) + range(i + 1, X.shape[1])

        for val in unique_val:
            indice = np.where(X[:, i] == val)[0]
            # print indice.shape
            sub_dict[val] = (X[np.ix_(indice, c)], y[indice])

        return sub_dict  # sub_data, sub_target

    def creat_tree(self, X, y, col=list('abcde')):

        unique_val, counts = np.unique(y, return_counts=True)
        major_val = unique_val[counts.argmax()]

        if len(unique_val) == 1:
            return unique_val[0]

        if len(y) <= 3 or X.shape[1] == 1:
            return major_val

        tree = {}

        index, sub_dict = self.choose_best_feature(X, y)
        if index < 0:
            return major_val

        tree[col[index]] = {}
        sub_col = col[:index] + col[index + 1:]
        for val, (sub_data, sub_target) in sub_dict.items():
            # zz = sub_data[0, index]
            tree[col[index]][val] = self.creat_tree(sub_data, sub_target, sub_col)

        self.tree = tree

        return tree

    def display_node(self, values):
        if not isinstance(values, dict):
            print values
            return
        for val, item in values.items():
            print val
            self.display_node(item)

    def display(self):
        tree = self.tree
        if tree:
            self.display_node(tree)



def test(X, y):
    # X = np.array(X)
    # y = np.array(y)
    length = 200
    X = np.random.randint(3, size=(length, 5))
    y = np.random.randint(3, size=length)

    tree = Tree()
    res = tree.creat_tree(X, y)
    print res
    tree.display()

test(X, y)
