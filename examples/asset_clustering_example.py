import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from asset_cluster.mpPDF import *
from asset_cluster.data import load_data
from asset_cluster.cluster import asset_clustering
import FinanceDataReader as fdr
import os
from datetime import datetime

snp500 = fdr.StockListing('S&P500')
idx = np.random.permutation(500)[:100]
tickers = list(snp500.Symbol.iloc[idx])
stocks_info = snp500.set_index('Symbol')

if not os.path.exists('Datasets/close.csv'):
    load_data(tickers, close=True)
df = pd.read_csv('Datasets/close.csv', index_col='Date')
returns = df.pct_change().dropna()

dist_matrix, clusters = asset_clustering(returns, metric='corr', denosing=False, max_num_clusters=25)

for i in range(len(clusters)):
    today = datetime.now().date()
    path = 'Results/' + today + str(i) + '.csv'
    stocks_info.loc[clusters[i]].to_csv(path)

sns.heatmap(dist_matrix)
plt.show()