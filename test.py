from util import get_loader
import pprint
from functools import partial
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import ScaledTranslation
from matplotlib.transforms import Affine2D
import tikzplotlib
# plt.style.use(['science','bright'])
# plt.style.use(['fivethirtyeight'])
# plt.style.use(['seaborn-paper'])
# plt.style.use(['ggplot'])
plt.style.use(['default'])
plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 15
plt.rcParams["font.family"] = "Times New Roman"

pp = pprint.PrettyPrinter(indent=4)


loader = get_loader()

pd.set_option('display.max_columns', 500)  # or 1000
pd.set_option('display.max_rows', 500)  # or 1000

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# change color
c_pu = colors[4]
c_gr = colors[2]
c_or = colors[1]
c_red = colors[3]
colors[2] = c_pu
colors[4] = c_gr

# dataset = 'music'
# dataset = 'movie'
# dataset = 'book'
# dataset = 'news'
dataset = 'nist'

usecost = True
# usecost = False

algs = ['greedy', 'greedy_w', 'greedy_sr', 'quality', 'random']
ALGS = dict([('DP', 'DP'),
             ('greedy', 'Greedy-U'),
             ('greedy_sr', 'AG'),
             ('greedy_w', 'Greedy-W'),
             ('quality', 'Quality'),
             ('random', 'Random')])


version = 11

query = {"$and": [
    {"status": 'COMPLETED'},
    {"config.ver": version},
    {"config.usecost": usecost},
    {"config.dataset": dataset},
    #     {"config.nth_sample": 0},
    #     {"_id": 9},
    #     {"config.dataset": "dota"}
]}
qs = loader.find(query)
df = qs.project(on=[
    'config.iter',
    'config.rn',
    'config.dataset',
    'config.nth_sample',
    'config.maxbudget',
    'config.method',

    'info.obj',
    'info.runtime',
])

df = df.sort_values(by=['rn', 'dataset', 'nth_sample', 'maxbudget'])
# df


df2 = df.pivot_table(index=['dataset', 'nth_sample', 'rn'], columns=[
                     'maxbudget', 'method'], values=['obj'])
groups = pd.MultiIndex.from_product([df2.columns.get_level_values(0).unique(),
                                     df2.columns.get_level_values(1).unique()])
df2

fig, ax = plt.subplots()
trans = {'greedy_sr': Affine2D().translate(-0.5, 0.0),
         'greedy_w': Affine2D().translate(0.5, 0.0),
         'DP': Affine2D().translate(-0.5, 0.0),
         'quality': Affine2D().translate(0.5, 0.0)
         }

idx = pd.IndexSlice
ratios = []
for i in df2.index.get_level_values(1).unique().values:
    samp = df2.loc[idx[:, i, :], ]
    ratio = samp.std() / (samp.mean() + 1e-9)
    ratios.append(ratio)
ratio = sum([r for r in ratios]) / len(ratios)

df3 = df2.mean()
df3_err = df2.mean() * ratio
maxbudget = df3.index.get_level_values(1).unique()
# algs = df3.index.get_level_values(2).unique()
idx = pd.IndexSlice

# for alg in algs[1:].tolist() + [algs[0]]:
for i, alg in enumerate(algs + ['DP'] if dataset != 'book' else algs):
    y = df3.loc[idx[:, :, alg]].values
    err = df3_err.loc[idx[:, :, alg]].values
    bar = plt.errorbar(maxbudget, y, yerr=err, fmt='-o', label=ALGS[alg],
                       color=colors[i],
                       transform=trans[alg] +
                       ax.transData if alg in trans else None
                       )
    bar[0].set_label(ALGS[alg])

plt.xlabel('Maximum budget')
plt.ylabel('\#activated\\\\users')
plt.legend()

code = tikzplotlib.get_tikz_code()

fig, ax = plt.subplots()
trans = {'greedy_sr': Affine2D().translate(-1, 0.0),
         'greedy_w': Affine2D().translate(1, 0.0),
         }

idx = pd.IndexSlice
ratios = []
for i in df2.index.get_level_values(1).unique().values:
    samp = df2.loc[idx[:, i, :], ]
    ratio = samp.std() / (samp.mean() + 1e-9)
    ratios.append(ratio)
ratio = sum([r for r in ratios]) / len(ratios)

df3 = df2.mean()
df3_err = df2.mean() * ratio
maxbudget = df3.index.get_level_values(1).unique()
# algs = df3.index.get_level_values(2).unique()
idx = pd.IndexSlice

for i, alg in enumerate(algs):
    y = df3.loc[idx[:, :, alg]].values
    err = df3_err.loc[idx[:, :, alg]].values
    bar = plt.errorbar(maxbudget, y, yerr=err, fmt='-o', label=ALGS[alg],
                       c=colors[i],
                       #                        transform=trans[alg]+ax.transData if alg in trans else None # only movie
                       )
    bar[0].set_label(ALGS[alg])

plt.xlabel('Maximum budget')
plt.ylabel('\#activated\\\\users')
plt.legend()

code = tikzplotlib.get_tikz_code()
