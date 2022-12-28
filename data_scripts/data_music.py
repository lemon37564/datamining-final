import numpy as np
import pandas as pd
import pickle
from collections import defaultdict


rng = np.random.default_rng(12345)


def sample_users(nsamp, nsz, like_thr=1):
    users = set()
    with open('datasets/train_triplets.txt', 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            user, song, count = line.split()
            users.add(user)

    users = list(users)
    print('#user: ', len(users))

    samp = []
    for i in range(nsamp):
        idx = rng.choice(range(len(users)), size=nsz, replace=False)
        usamp = [users[j] for j in idx]
        samp.append(dict([(u, []) for u in usamp]))

    with open('datasets/train_triplets.txt', 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            user, song, count = line.split()
            count = int(count)
            for s in samp:
                if user in s and count > like_thr:
                    s[user].append(song)

    with open('datasets/music.pkl', 'wb') as fout:
        pickle.dump(samp, fout)


def test_song_overlap():
    with open('datasets/music.pkl', 'rb') as fin:
        samp = pickle.load(fin)

        for s in samp:
            counter = defaultdict(int)
            for u,sgs in s.items():
                for song in sgs:
                    counter[song] += 1

            hist = defaultdict(int)
            for sg, c in counter.items():
                hist[c] += 1
            print(hist)


if __name__ == '__main__':
    sample_users(nsamp=10, nsz=100)
    #test_song_overlap()
