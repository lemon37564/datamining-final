import numpy as np
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=4)

from experiment import new_exp


if __name__ == '__main__':
    import sys
    ver = int(sys.argv[1])
    it = int(sys.argv[2])
    rn = int(sys.argv[3])
    dataset = sys.argv[4]
    maxbudget = int(sys.argv[5])
    usecost = True if sys.argv[6] == 'True' else False

    methods = ['greedy','greedy_w','greedy_sr','random','quality']
    methods = methods + ['DP'] if usecost else methods
    #methods = ['DP']

    nsamp = [3,10,3]
    maxcosts = [10,5,10]
    epss = [0.9,0.05,0.05]
    idx = {'music': 0,
           'movie': 0,
           'book': 0,
           'news': 1,
           'mnist': 2,
           'nist': 2,
           'runtime': 0,
          }
    nths = [i for i in range(nsamp[idx[dataset]])]
    if dataset == 'runtime':
        if maxbudget == 99:
            nths = [10, 100, 1000, 10000]
            if methods[0] == 'DP' and len(methods)==0:
                nths = [10, 100, 1000]
        else:
            nths = [99]

    ex = new_exp(interactive=False)
    kv = [
        ('ver', [ver]),
        ('iter', [it]),
        ('rn', [rn]),
        ('dataset', [dataset]),
        ('nth_sample', nths),
        ('maxbudget', [maxbudget]),
        ('maxcost', [maxcosts[idx[dataset]]]),
        ('usecost', [usecost]),
        ('eps', [epss[idx[dataset]]]),
        ('method', methods),
    ]

    ks, vs = zip(*kv)
    for alls in itertools.product(*list(vs)):
        conf = dict([(k, v) for k,v in zip(list(ks),alls)])
        pp.pprint(conf)
        r = ex.run(config_updates=conf)
