from math import log
from collections import Counter
import numpy as np
from pandas import DataFrame

# my_data = [['slashdot', 'USA', 'yes', 18, 213.2, 'None'],
#            ['google', 'France', 'yes', 23, 121.2, 'Premium'],
#            ['digg', 'USA', 'yes', 24, 21.32, 'Basic'],
#            ['kiwitobes', 'France', 'yes', 23, 1.2, 'Basic'],
#            ['google', 'UK', 'no', 21, .2, 'Premium'],
#            ['(direct)', 'New Zealand', 'no', 12, 71.2, 'None'],
#            ['(direct)', 'UK', 'no', 21, -21.2, 'Basic'],
#            ['google', 'USA', 'no', 24, 241.2, 'Premium'],
#            ['slashdot', 'France', 'yes', 19, 20, 'None'],
#            ['digg', 'USA', 'no', 18, 1.0, 'None'],
#            ['google', 'UK', 'no', 18, 2, 'None'],
#            ['kiwitobes', 'UK', 'no', 19, 44, 'None'],
#            ['digg', 'New Zealand', 'yes', 12, 27, 'Basic'],
#            ['slashdot', 'UK', 'no', 21, 86, 'None'],
#            ['google', 'UK', 'yes', 18, 2, 'Basic'],
#            ['kiwitobes', 'France', 'yes', 19, 0.0, 'Basic']]

my_data = [[213.2, 'None'],
           [121.2, 'Premium'],
           [21.32, 'Basic'],
           [1.2, 'Basic'],
           [.2, 'Premium'],
           [71.2, 'None'],
           [-21.2, 'Basic'],
           [241.2, 'Premium'],
           [20, 'None'],
           [1.0, 'None'],
           [2, 'None'],
           [44, 'None'],
           [27, 'Basic'],
           [86, 'None'],
           [2, 'Basic'],
           [0.0, 'Basic']]


class Tree(object):
  """docstring for Tree"""
  def __init__(self, arg):

    self.arg = arg
