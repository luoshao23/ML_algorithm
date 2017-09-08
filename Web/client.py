import socket

name = raw_input('Login as: ').strip()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 5555))
s.send(name)

print s.recv(1024).decode('utf-8')

while True:

    data = raw_input('Enter...: ')
    tousr, msg = data.split('|')
    if msg == 'exit':
        break
    handle_msg = '%s|%s|%s' % (name, tousr, msg)
    s.send(handle_msg)
    print 'sent: %s' % handle_msg

# s.send(b'exit')
s.close()
