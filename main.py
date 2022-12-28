from util import load_data
from rank import *
import time

V, F, B, c, un = load_data(
    "movie", 10, 2, usecost=True)

t1 = time.perf_counter()
# unweighted greedy
r = greedy(V, F, B, c=c)
print(r, measure(r, V, F, B, c))
print("unweighted greedy: {:.3f}s".format(time.perf_counter() - t1))

t1 = time.perf_counter()
# weighted greedy
weight = [1/b for b in B]
r = greedy(V, F, B, c=c, weight=weight)
print(r, measure(r, V, F, B, c))
print("weighted greedy: {:.3f}s".format(time.perf_counter() - t1))

t1 = time.perf_counter()
# DP
r = DP(V, F, B, c, eps=0.5)
print(r, measure(r, V, F, B, c))
print("DP: {:.3f}s".format(time.perf_counter() - t1))
