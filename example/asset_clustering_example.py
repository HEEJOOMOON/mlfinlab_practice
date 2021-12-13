import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from asset_cluster.mpPDF import *
from asset_cluster.data import load_data
from asset_cluster.cluster import asset_clustering
import FinanceDataReader as fdr
import os

snp500 = fdr.StockListing('S&P500')
idx = np.random.permutation(500)[:50]
tickers = list(snp500.Symbol.iloc[idx])

if not os.path.exists('Datasets/close.csv'):
    load_data(tickers, close=True)
df = pd.read_csv('Datasets/close.csv', index_col='Date')
returns = df.pct_change().dropna()

clusters = asset_clustering(returns, metric='corr')
sns.heatmap(clusters)
plt.show()