buckets = 100
a = [(bucket*buckets,bucket*buckets+buckets) for bucket in range(10)]

def find(d):
    for index, item in enumerate(a):
        if item[0] < d and item[1] > d:
                return index
    return -1

print(find(1000))