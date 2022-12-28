import pytest
import numpy as np
from rank import DP, greedy


def test_DP():
    V = [0,1]
    c = [1,100]
    u = [0.1,0.9]
    F = [lambda S: sum([u[i] for i in set(S)])]
    B = [100]
    weight = np.ones(len(F), dtype='int')

    r = greedy(V, F, B, c, weight)
    assert r == [0]

    r = DP(V, F, B, c, eps=0.05)
    assert r == [1]

