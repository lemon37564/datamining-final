import numpy as np
import pandas as pd
import pickle
from time import process_time

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from util import mongo_uri, db_name, load_data
from rank import measure, greedy, greedy_sr, rank_by_quality, DP


def new_exp(interactive=True):
    ex = Experiment('jupyter_ex', interactive=interactive)
    # ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
    if mongo_uri is not None and db_name is not None:
        ex.observers.append(MongoObserver(url=mongo_uri, db_name=db_name))

    def get_logger(_run):
        def log_run(key, val):
            _run.info[key] = val
        from logger import log
        if mongo_uri is not None and db_name is not None:
            # log = _run.log_scalar # for numerical series
            log = log_run

        return log

    @ex.config
    def my_config():
        pass

    @ex.main
    def my_main(_run, ver, iter, dataset, rn,
                nth_sample, method, maxbudget, usecost, maxcost, eps
                ):
        log = get_logger(_run)
        np.random.seed(rn)

        # Load data
        V,F,B,c,_ = load_data(dataset, maxbudget=maxbudget, nth=nth_sample, usecost=usecost, maxcost=maxcost)

        # ... START training
        t1_start = process_time()

        if method == 'DP':
            r = DP(V, F, B, c, eps)
        if method == 'greedy':
            r = greedy(V, F, B, c=c)
        if method == 'greedy_w':
            weight = [1/b for b in B]
            r = greedy(V, F, B, c=c, weight=weight)
        if method == 'greedy_sr':
            r = greedy_sr(V, F, B, c=c)
        if method == 'random':
            r = [i for i in np.random.permutation(len(V))]
        if method == 'quality':
            r = rank_by_quality(V, F)

        t1_stop = process_time()
        log('runtime', t1_stop - t1_start)
        # ... END training

        log('obj', measure(r, V, F, B, c))
        if dataset.endswith('nist'):
            accs = _
            log('acc', measure(r, range(len(V)), accs, B, c) / len(accs))

    return ex
