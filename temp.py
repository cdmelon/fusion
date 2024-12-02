q = ["Who", "Where", "What", "unknown"]
b = {}
for i in a.keys():
    print(i)
    print(a[i])
    o = input()
    b[i] = q[int(o) - 1]
