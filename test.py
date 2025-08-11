n, k = map(int, input().split())
good_items = set()
for i in range(k):
    l, r = map(int, input().split())
    for j in range(l, r+1):
        good_items.add(j)

defective_items = []
for i in range(1, n+1):
    if i not in good_items:
        defective_items.append(i)

print(len(defective_items))
print(" ".join(str(i) for i in defective_items))