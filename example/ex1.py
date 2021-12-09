from asset_cluster.mpPDF import *
import seaborn as sns
import matplotlib.pyplot as plt
from asset_cluster.data import load_data



load_data()
df = pd.read_csv('data/snp_close.csv', index_col='Date')
returns = df.pct_change().dropna()
q = len(returns.index) / len(returns.columns)
corr = empirical_denoising(returns, q=q, metrics='corr')
corr_, cluster_, _ = clusterKMeansTop(corr, maxNumClusters=10)
sns.heatmap(corr_)
plt.show()