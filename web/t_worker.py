import time, sys, Queue

from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass

QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

server_addr = '127.0.0.1'
print 'Connect to server %s...' % server_addr

m = QueueManager(address = (server_addr, 5000), authkey=b'dqdcwqdqd')

m.connect()
task = m.get_task_queue()
result = m.get_result_queue()

# for i in xrange(10):
#     try:
#         n = task.get(timeout = 1)
#         # print 'run task %d * %d...' %(n, n)
#         # r = '%d * %d = %d' % (n, n, n*n)
#         time.sleep(0.1)
#         result.put(r)
#     except Queue.Empty:
#         print 'task queue is empty.'
flag = True
while flag:

    try:
        n = task.get(timeout = 60)
        if n == 'quit':
            print 'work exit.'
            r = 'quit'
            time.sleep(0.1)
            result.put(r)
            flag = False
        else:
            r = '='.join([s.upper() for s in n.split(' ')])
            print 'run task %s' % r
        # r = '%d * %d = %d' % (n, n, n*n)
            time.sleep(0.1)
            result.put(r)
    except Queue.Empty:
        print 'task queue is empty.'


