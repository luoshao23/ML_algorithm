def DH(p, g, randomKey):
    return g ** randomKey % p
p = 509
g = 5
myRandomKey = 123

A = DH(p, g, myRandomKey)
print(f'Send: p={p}, g={g}, A={A}')
otherRandomKey = 456
B = DH(p, g, otherRandomKey)
print(f'Send: B={B}')

s = DH(p, A, otherRandomKey)
print(f'Secretly s={s}')

print(DH(p, A, otherRandomKey) == DH(p, B, myRandomKey))
