########## Incense ##########
import itertools
import pickle
import numpy as np
from incense import ExperimentLoader

# Try to locate config file for Mongo DB
import importlib
spec = importlib.util.find_spec('mongodburi')
if spec is not None:
    from mongodburi import mongo_uri, db_name
else:
    mongo_uri, db_name = None, None


def get_loader(uri=mongo_uri, db=db_name):
    loader = ExperimentLoader(
        mongo_uri=uri,
        db_name=db
    )
    return loader


########## Util ##########


def load_data(dataset, maxbudget, nth=None, n_top_kw=20, usecost=False, maxcost=1):
    if dataset == 'runtime':
        # use nth for #func
        V = np.arange(maxbudget)

        maxsz = min(50, len(V)//10)
        F_ = [set(np.random.choice(len(V), np.random.randint(
            0, maxsz), replace=False)) for _ in range(nth)]
        F = [lambda S, subset=sgs: int(
            np.any([v in subset for v in S])) for sgs in F_]

        B = np.random.randint(1, maxbudget + 1, size=len(F))
        c = np.random.randint(1, maxcost + 1, size=len(V)) if usecost else None

        return V, F, B, c, None

    with open(f'datasets/{dataset}.pkl', 'rb') as fin:
        samp = pickle.load(fin)

        if dataset in ['music', 'movie', 'book', 'ratings_Grocery_and_Gourmet_Food']:
            if nth is not None:
                samp = samp[nth]

            V = itertools.chain(*[sgs for sgs in samp.values()])
            V = list(set(V))

            F_ = [set(sgs) for sgs in samp.values()]
            F = [lambda S, subset=sgs: int(
                np.any([v in subset for v in S])) for sgs in F_]

            B = np.random.randint(1, maxbudget + 1, size=len(F))
            c = np.random.randint(
                1, maxcost + 1, size=len(V)) if usecost else None

        if dataset == 'news':
            if nth is not None:
                samp = samp[nth]
            cat, components, doc_word_list, doc_lens, words = samp

            V = list(range(len(doc_word_list)))
            Vset = [set(doc_word_list[i]) for i in V]

            F_ = []
            for topic_idx, topic in enumerate(components):
                top_features_ind = topic.argsort()[:-n_top_kw - 1:-1]
                F_.append(set(top_features_ind))

            def union(S): return itertools.chain(*[Vset[i] for i in S])
            F = [lambda S, subset=kws: len(subset.intersection(
                union(S))) / len(subset) for kws in F_]

            if usecost:
                B = np.random.randint(1, np.mean(
                    doc_lens) * maxbudget, size=len(F))
            else:
                B = np.random.randint(1, maxbudget + 1, size=len(F))
            c = np.array(doc_lens) if usecost else None

        if dataset.endswith('nist'):
            with open(f'datasets/{dataset}-{nth}.pkl', 'rb') as fin_:
                samp = pickle.load(fin_)
            (X_train, y_train), (X_test, y_test), dims, Ds = samp

            V = list(range(X_train.shape[0]))

            # rc originally means selected row/col of an image
            R = [max([max(Ds[i][u]) for u in V] + [1e-5])
                 for i, rc in enumerate(dims)]  # avoid 0
            # radius originally minimax facility function
            #radius_ = lambda S, rcidx: max([min([Ds[rcidx][min(v,s)][np.abs(v-s)] for s in S]) for v in V])
            def radius_(S, rcidx): return np.mean(
                [min([Ds[rcidx][min(v, s)][np.abs(v-s)] for s in S]) for v in V])
            F = [lambda S, rcidx=i, maxr=maxr_:
                 (maxr - radius_(S, rcidx))/maxr if len(S) > 0 else 0
                 for i, (rc_, maxr_) in enumerate(zip(dims, R))]

            B = np.random.randint(1, maxbudget + 1, size=len(F))
            c = np.random.randint(
                1, maxcost + 1, size=len(V)) if usecost else None

            def norm(vec): return np.linalg.norm(vec)
            def d(u, v, rc): return norm(u[rc] - v[rc])

            def pred(S, rc): return np.array([y_train[S[np.argmin([d(X_train[s], x, rc)
                                                                   for s in S])]]
                                              for x in X_test])
            accs = [lambda S, rc=rc_: sum(pred(S, rc) == y_test) / len(y_test)
                    if len(S) > 0 else 0
                    for rc_ in dims]

            return V, F, B, c, accs

    return V, F, B, c, None
