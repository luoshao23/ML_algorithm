import random
import time
import Queue
from multiprocessing.managers import BaseManager

task_queue = Queue.Queue()
result_queue = Queue.Queue()



class QueueManager(BaseManager):
    pass

QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)

manager = QueueManager(address=('', 5000), authkey=b'dqdcwqdqd')

manager.start()

task = manager.get_task_queue()
result = manager.get_result_queue()

while True:
    inputs = raw_input('Enter your sentence:\n').strip().strip('\n')
    print 'Sending %s...' % inputs
    task.put(inputs)

    r = result.get(timeout=60)
    if r == 'quit':
    	break
    else:
	    print 'Result: %s' % r

# for i in xrange(10):
#     n = random.randint(0, 10000)
#     print 'Put task %d...' % n
#     task.put(n)

# print 'Try get result...'
# for i in xrange(10):
#     r = result.get(timeout=10)
#     print 'Result: %s' % r

manager.shutdown()
print 'master exit.'
