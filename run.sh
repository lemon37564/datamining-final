 
datasets=(music)
datasets=(movie)
datasets=(book)
maxbudget=(5 10 15 20)

#datasets=(news)
#maxbudget=(2 4 6 8)

#datasets=(nist)
#maxbudget=(25 50 75 100)

#datasets=(runtime)
#maxbudget=(99)
#maxbudget=(10 100 1000 10000)

#ver=1
#ver=2 # n_top_kw from 10 to 20
#ver=3 # +DP and bug fixes
#ver=4 # random cost scheme; memory-efficient DP; bug in random ranking
#ver=5 # DP speedup: usecost=True only
#ver=6 # Only for 20news: query-based data
#ver=7 # Only for 20news: doc length
#ver=8 # further DP speedup; +NIST; lazy greedy
#ver=9 # kNN only; minimax to avg min distance; feature selections
#ver=10 # kNN only; ndim=20; runtime
ver=11 # DP bug; +error bars

#its=(1 2 3)
its=(1)
rn=(nan 42 250 999 2500 21826)

#ver=99 # for testing
#datasets=(news)
#maxbudget=(2)

for d in ${datasets[@]}; do
for it in ${its[@]}; do
for b in ${maxbudget[@]}; do
    r=${rn[$it]}
    nohup python run.py $ver $it ${r} ${d} ${b} True &> logdata-$d-$ver-$it-${r}-${b}-cTrue &
    #nohup python run.py $ver $it ${r} ${d} ${b} False &> logdata-$d-$ver-$it-${r}-${b}-cFalse &
done
done
done
