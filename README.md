# max-submodular-ranking-code

## Repo structure

* `rank.py`: algorithms and baselines.
* `data_scripts`: precess datasets
* `results.ipynb`: compare performance and plotting

## Examples

```
>>> import numpy as np
>>> from rank import measure, greedy, DP
>>> 
>>> # prepare toy dataset
>>> nitem, nfunc = 50, 5
>>> V = np.arange(nitem)
>>> sets = [{24, 34}, {34}, {35, 41}, {43}, {24}]
>>> F = [lambda S, subset=st: int(np.any([v in subset for v in S])) for st in sets]
>>> 
>>> maxbudget, maxcost = 10, 5
>>> B = np.array([5, 7, 1, 4, 1])
>>> c = np.array([1, 2, 4, 4, 4, 1, 5, 5, 4, 2, 3, 4, 4, 4, 3, 3, 4, 1, 2, 2, 5, 3,
...         1, 1, 4, 4, 5, 5, 3, 3, 2, 1, 5, 5, 4, 2, 5, 3, 5, 3, 1, 1, 3, 5,
...         5, 3, 2, 3, 5, 1])
>>> 
>>> # unweighted greedy
>>> r = greedy(V, F, B, c=c)
>>> r, measure(r, V, F, B, c)
([41, 34], 1)
>>> 
>>> # weighted greedy
>>> weight = [1/b for b in B]
>>> r = greedy(V, F, B, c=c, weight=weight)
>>> r, measure(r, V, F, B, c)
([41, 34], 1)
>>> 
>>> # DP
>>> r = DP(V, F, B, c, eps=0.5)
>>> r, measure(r, V, F, B, c)
([41, 34], 1)
>>> 
```

## Tests

Run 

```
>>> import doctest
>>> doctest.testfile('README.md')
```

and

```
$ pytest test_rank.py 
```
