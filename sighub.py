inputs = [(1,   2),
(2,   5),
(3,   1),
(4,   7),
(5,   5),
(6,   8),
(7,   6),
(8,   9),
(9,   9)]

def connect( input ):
    N=10 # Number of nodes
    rank=[0]*N
    parent=range(N)
    def Find(x):
        """Find representative of connected component"""
        if  parent[x] != x:
            parent[x] = Find(parent[x])
        return parent[x]

    def Union(x,y):
        """Merge sets containing elements x and y"""
        x = Find(x)
        y = Find(y)
        if x == y:
            pass
        if rank[x]<rank[y]:
            parent[x] = y
        elif rank[x]>rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1

    for a, b in input:
        # line = line.toDict()
        # XTRA_CARD_NBR   = line['XTRA_CARD_NBR']
        # XTRA_CARD_NBR_2 = line['XTRA_CARD_NBR_2']
        Union(a,b)

    value_list = {}
    for n in range(N):
        value_list['XTRA_CARD_NBR']  = str(n)
        value_list['EC_CARD_LINKED'] = str(Find(n))
        print value_list['XTRA_CARD_NBR'], value_list['EC_CARD_LINKED']

connect(inputs)