import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


rng = np.random.default_rng(2525)
fname = 'datasets/ratings_Grocery_and_Gourmet_Food.csv'
foname = 'datasets/ratings_Grocery_and_Gourmet_Food.pkl'


def sample_users(nsamp, nsz, like_thr=2.9):
    users = set()
    with open(fname, 'r') as fin:
        line = fin.readline()  # skip 1st line
        while True:
            line = fin.readline()
            if not line:
                break
            # e.g.: 1,296,5.0,1147880044
            song, user, count, _ = line.split(',')
            users.add(user)

    users = list(users)
    print('#user: ', len(users))

    samp = []
    for i in range(nsamp):
        idx = rng.choice(range(len(users)), size=nsz, replace=False)
        usamp = [users[j] for j in idx]
        samp.append(dict([(u, []) for u in usamp]))

    with open(fname, 'r') as fin:
        line = fin.readline()  # skip 1st line
        while True:
            line = fin.readline()
            if not line:
                break
            song, user, count, _ = line.split(',')
            count = float(count)
            for s in samp:
                if user in s and count > like_thr:
                    s[user].append(song)

    with open(foname, 'wb') as fout:
        pickle.dump(samp, fout)


def test_song_overlap():
    with open(foname, 'rb') as fin:
        samp = pickle.load(fin)

        for s in samp:
            counter = defaultdict(int)
            for u, sgs in s.items():
                for song in sgs:
                    counter[song] += 1

            hist = defaultdict(int)
            for sg, c in counter.items():
                hist[c] += 1
            print(hist)


if __name__ == '__main__':
    sample_users(nsamp=10, nsz=1000)
    test_song_overlap()
