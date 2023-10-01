# Source: https://stackoverflow.com/questions/57328273/algorithm-for-polygon-triangulation
# Generate all 14 unique permutations of vertices indices [vi, vj, vk] for each triangle
def trivial():
    permu = []
    cen = 0
    for i in range(1, 6):
        permu.append([cen, i, (i + 1)])
    permu.append([cen, 6, 1])
    return permu

def genTriangles(i, j):
    if j - i < 2:
        yield []
        return
    if j - i == 2:
        yield [[i, i+1, j]]
        return 
    for k in range(i + 1, j):
        for x in genTriangles(i, k):
            for y in genTriangles(k, j):
                yield x + y + [[i, k, j]]

def noncenter_permu(n = 6):
    permu = []
    for _, tr in enumerate(genTriangles(0, n - 1), 1):
        permu.append(tr)
    return permu

if __name__ == "__main__":
    print(noncenter_permu())
    print(trivial())