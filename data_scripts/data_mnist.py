import pickle
import numpy as np

#dataset = 'mnist'
dataset = 'nist'
with open(f'datasets/{dataset}.pkl', 'rb') as fin:
    samps = pickle.load(fin)
    for nth, samp in enumerate(samps):
        (X_train, y_train), (X_test, y_test), dims = samp

        V = list(range(X_train.shape[0]))
        Xs = [[X_train[v][dim] for v in V] for dim in dims]

        norm = lambda vec: np.linalg.norm(vec)
        d = lambda u,v,idim: norm(Xs[idim][u] - Xs[idim][v])
        Ds = [[[d(u,v,i) for v in V if u<=v] for u in V] for i,dim in enumerate(dims)] # pre-computing
        print(f'sample {nth} is done...')

        with open(f'datasets/{dataset}-{nth}.pkl', 'wb') as fout:
            samp = [(X_train, y_train), (X_test, y_test), dims, Ds]
            pickle.dump(samp, fout)
