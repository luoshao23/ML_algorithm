import socket
import threading
import time

class Users(object):
    """docstring for Users"""
    def __init__(self, name, skt):
        self.name = name
        self.skt = skt

    def sendmsg(self, msg):
        self.skt.send(msg)

    def logout(self):
        self.skt.close()




def tcplink(usr, addr):
    print 'Accept new connection from %s:%s...' % addr
    usr.sendmsg('Welcome new user %s!' % usr.name)
    while True:
        data = usr.skt.recv(1024)
        time.sleep(0.5)
        print 'recved: %s' % data
        name, tousr, msg = data.split('|')
        if not data or msg.decode('utf-8') == 'exit':
            break
        for user in userlist:
            if user.name == tousr:
                user.sendmsg('from[%s]:%s'%(name, msg))
        else:
            usr.sendmsg('No this user.')
    sock.close()
    print 'Connection from %s:%s closed.' % addr


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 5555))
s.listen(5)
userlist = []
print 'Waiting for connection...'

while True:
    sock, addr = s.accept()
    name = sock.recv(1024)
    if name not in userlist:
        usr = Users(name, sock)
        userlist.append(usr)

        print 'New user %s log in' % name

        t = threading.Thread(target=tcplink, args=(usr,addr))
        t.start()
