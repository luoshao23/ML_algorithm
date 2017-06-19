#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-

' a test modelu'

__author__ = 'Shaoze.Luo'

from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

# class Student(object):
#     """docstring for Student"""

#     def __init__(self):
#         pass

#     @property
#     def score(self):
#         return self.__score

#     @score.setter
#     def score(self, value):
#         if not isinstance(value, int):
#             raise ValueError('score must be an integer!')
#         if value < 0 or value > 100:
#             raise ValueError('score must between 0 ~ 100!')
#         self.__score = value


# class Fib(object):

#     def __init__(self):
#         self.a, self.b = 0, 1  # 初始化两个计数器a，b

#     def __iter__(self):
#         return self  # 实例本身就是迭代对象，故返回自己

#     def next(self):
#         self.a, self.b = self.b, self.a + self.b  # 计算下一个值
#         if self.a > 100000:  # 退出循环的条件
#             raise StopIteration()
#         return self.a  # 返回下一个值

# class Chain(object):

#     def __init__(self, path=''):
#         self._path = path

#     def __getattr__(self, path):
#         return Chain('%s/%s' % (self._path, path))

#     def __call__(self,path):
#         return Chain('%s/%s' % (self._path, path))

#     def __str__(self):
#         return self._path

#     __repr__ = __str__


# class PostiveInt(int):
#     """docstring for PostiveInt"""
#     # def __init__(self, value):
#     #     super(PostiveInt, self).__init__(abs(value))

#     def __new__(cls, value):
#         return super(PostiveInt, cls).__new__(cls, abs(value))

