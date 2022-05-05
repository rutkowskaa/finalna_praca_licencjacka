import Get_Data
import features
import pandas as pd
import numpy as np

data = Get_Data.Get_Data("^IXIC", start="2015-01-01", interval="1d").make_diff()


okno = 300

result = []

for i in range(okno, len(data)):
    df = data.iloc[i - okno: i]
    stats = 0
    stats = features.get_statistics(df, max_lag=3)
    print(i - okno, i)
    if stats[0][0] > 0.05 and \
       stats[0][1] > 0.05 and \
       stats[0][2] > 0.05 and \
       stats[1][0] > 0.05 and \
       stats[1][1] > 0.05 and \
       stats[1][2] > 0.05 and \
       stats[2][0] < 0.05 and \
       stats[2][1] < 0.05 and \
       stats[2][2] < 0.05:
        result.append([df.index[0], df.index[-1]])
        print(stats[2])

print(result)