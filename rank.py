import numpy as np
import itertools
import math


def measure(R, V, F, B=None, c=None):
    '''
    Measure the utility of a ranking.
    :param R: a list of ranked items indices
    :param V: a list of items to rank
    :param F: a list of submodular functions
    :param B: a list of budgets, one for each function
    :param c: a list of item costs
    '''

    if len(R) == 0:
        return 0

    if c is None:
        ms = [f([V[i] for i in R[:b]]) for f, b in zip(F, B)]
    else:
        accb = np.array(list(itertools.accumulate([c[i] for i in R])))
        # i as the first item beyond the budget b
        def b2i(b): return len(
            accb) if b > accb[-1] else np.where(b <= accb)[0][0]
        ms = [f([V[i] for i in R[:b2i(b)]]) for f, b in zip(F, B)]

    # print(ms)
    return sum(ms)


def DP(V, F, B, c, eps):
    '''
    DP algorithm for selecting large items given non-uniform costs
    '''
    eps = eps
    MAXc = 1e7

    # Sorted items with non-decreasing cost
    idxV = np.argsort([_ for _ in c])
    V_, c_ = V, c
    V = [V_[j] for j in idxV]
    c = [c_[j] for j in idxV]

    # Define DP objective
    def largeF(j, cost): return [k for k, b in enumerate(
        B) if cost+c[j] <= b and 2*c[j] > b]
    P = max([F[k]([V[j]]) for j in range(len(V)) for k in largeF(j, 0)] + [-1])
    if P == -1 or P == 0:
        # z=0 for all ranking, and return arbitrary ranking
        return idxV

    def scale_round(x): return math.floor(x / P * len(F) / eps)
    def z(j, cost): return sum(
        [scale_round(F[k]([V[j]])) for k in largeF(j, cost)])

    # DP table initialization
    nrow, ncol = scale_round(len(F)*P) + 1, 2
    T = dict([(t, []) for t in itertools.product(range(nrow), range(ncol))])
    cT = dict([(t, MAXc) for t in itertools.product(range(nrow), range(ncol))])
    for j in range(ncol):
        cT[(0, j)] = 0
    j = 0
    zj = z(j, 0)
    for i in range(1, nrow):
        if i <= zj:
            T[(i, j)] = [j]
            cT[(i, j)] = c[j]

    # DP updates: col by col
    for j in range(1, len(V)):
        j_ = j
        j = 1
        # pre-computing
        Bj = [(b, k) for k, b in enumerate(B) if 2*c[j_] > b]
        if len(Bj) > 0:
            Bj = reversed(sorted(Bj, key=lambda tup: tup[0]))
            Bj, Bjidx = zip(*Bj)
        zj, cur = 0, 0
        for i in reversed(range(0, nrow)):
            if len(Bj) > 0:
                while cur < len(Bj) and cT[(i, j-1)]+c[j_] <= Bj[cur]:
                    zj = zj + scale_round(F[Bjidx[cur]]([V[j_]]))
                    cur = cur + 1
            newi = i + zj
            if cT[(i, j-1)] + c[j_] < cT[(newi, j)]:
                T[(newi, j)] = T[(i, j-1)] + [j_]
                cT[(newi, j)] = cT[(i, j-1)] + c[j_]

        # final updates
        for i in reversed(range(1, nrow)):
            cs = [cT[(i, j-1)], cT[(i, j)], cT[(i+1, j)]
                  if i < nrow-1 else MAXc]
            Ts = [T[(i, j-1)], T[(i, j)], T[(i+1, j)] if i < nrow-1 else []]
            idx = np.argmin(cs)
            T[(i, j)] = Ts[idx]
            cT[(i, j)] = cs[idx]

        # move col 1 to col 0
        for i in range(1, nrow):
            T[(i, 0)] = T[(i, 1)]
            cT[(i, 0)] = cT[(i, 1)]

    j = 1
    istar = max([i for i in range(nrow) if cT[(i, j)] < MAXc])
    ranking = [idxV[j_] for j_ in T[(istar, j)]]
    return ranking


def greedy(V, F, B, c=None, weight=None):
    '''
    Cost-efficient greedy algorithm
    '''
    bmax = max(B)
    weight = np.ones(len(F), dtype='int') if weight is None else weight
    if c is None:
        c = np.ones(len(V), dtype='int')  # uniform cost if None
    ranking, ridx, sidx = [], [], set()
    scores = [1e8] * len(V)
    while sum([c[j_] for j_ in ridx]) < bmax:
        s_max = -1
        cr = sum([c[j_] for j_ in ridx])
        frs = [f(ranking) if b > cr else None for f, b in zip(F, B)]
        for j, v in enumerate(V):
            if j in sidx:
                scores[j] = 0
                continue
            # score[j] is non-increasing, update only when necessary,
            # i.e., when lower bound s_max is weak.
            if scores[j] < s_max:
                continue
            S = ranking + [v]
            cS = cr + c[j]
            score = sum([w * (f(S)-fr)
                        for w, f, b, fr in zip(weight, F, B, frs) if b >= cS])
            score = score / c[j]
            scores[j] = score
            if score > s_max:
                s_max = score
        nth = np.argmax(scores)
        if np.abs(scores[nth]) < 1e-9:
            break
        ranking.append(V[nth])
        ridx.append(nth)
        sidx.add(nth)

    return ridx


def greedy_sr(V, F, B, c=None):
    '''
    A greedy ranking by Submodular Ranking.
    Azar, Yossi, and Iftah Gamzu. "Ranking with submodular valuations." SODA, 2011.
    '''
    bmax = max(B)
    # uniform cost if None
    c = np.ones(len(V), dtype='int') if c is None else c
    ranking, ridx, sidx = [], [], set()
    denoms, scores = [[1e8]*len(F)] * len(V), [-1] * len(V)
    while sum([c[j_] for j_ in ridx]) < bmax:
        s_max = -1
        frs = [f(ranking) for f in F]
        for j, v in enumerate(V):
            if j in sidx:
                scores[j] = 0
                continue
            # the denominator (fS-fr) of each f is non-increasing,
            # obtain an upper bound by being divided by the new (1-fr).
            # update only when lower bound s_max is weak.
            ub = sum([d/(1-fr)/c[j] for d, fr in zip(denoms[j], frs)
                      if np.abs(1-fr) > 1e-5])  # f(V)=1
            if ub < s_max:
                # important, (1-fr) may be too small and be discarded
                scores[j] = ub
                continue

            S = ranking + [v]
            denom = [f(S)-fr for f, fr in zip(F, frs)]
            denoms[j] = denom
            score = sum([d/(1-fr)/c[j] for d, fr in zip(denoms[j], frs)
                         if np.abs(1-fr) > 1e-5])  # f(V)=1
            scores[j] = score
            if score > s_max:
                s_max = score

        nth = np.argmax(scores)
        if np.abs(scores[nth]) < 1e-9:
            break
        ranking.append(V[nth])
        ridx.append(nth)
        sidx.add(nth)

    return ridx


def rank_by_quality(V, F):
    def score(v): return sum([f([v]) for f in F])
    scores = np.array([score(v) for v in V])
    idx = np.argsort(-scores)
    return idx
