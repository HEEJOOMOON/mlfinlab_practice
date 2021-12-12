import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from asset_cluster.metrics import *
from asset_cluster.mpPDF import *
from asset_cluster.utils import *
from typing import Optional

# © 2020 Machine Learning for Asset Managers, Marcos Lopez de Prado


def clusterKMeansBase(matrix,
                      maxNumClusters: int=10,
                      n_init: int=10):

    x, silh = ((1-matrix.fillna(0))/2.0)**0.5, pd.Series(dtype='float64')
    x.fillna(0, inplace=True)
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh, kmeans = silh_, kmeans_

    newIdx = np.argsort(kmeans.labels_)
    corr1 = matrix.iloc[newIdx]  # reorder rows

    corr1 = matrix.iloc[:, newIdx]   # reorder columns
    clstrs = {i:matrix.columns[np.where(kmeans.labels_==i)[0]].tolist() \
              for i in np.unique(kmeans.labels_)}   # cluster members: keys-clusters' labels, values-list(elements)
    silh = pd.Series(silh, index=x.index)
    return corr1, clstrs, silh


def makeNewOutputs(matrix, clstrs, clstrs2):
    clstrsNew = {}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.key():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])

    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = matrix.loc[newIdx, newIdx]
    x = ((1-matrix.fillna(0))/2.)**0.5
    kmeans_labels = np.zeros(len(x.columns))

    for i in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i

    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index = x.index)
    return corrNew, clstrsNew, silhNew


def clusterKMeansTop(matrix, maxNumClusters=None, n_init=10):
    if maxNumClusters==None: maxNumClusters=matrix.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase(matrix, maxNumClusters=min(maxNumClusters, matrix.shape[1]-1), \
                                            n_init=n_init)
    clusterTstats = {i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters = [i for i in clusterTstats.keys() if \
                    clusterTstats[i]<tStatMean]
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]] # redoCluster에 있는 element들의 모음
        corrTmp = matrix.loc[keysRedo, keysRedo] # 새로 클러스터 만들 corr 새로 만들기
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp, \
                                                 maxNumClusters=min(maxNumClusters, \
                                                 corrTmp.shape[1]-1), n_init=n_init)    # 기준 미달 클러스터끼리 다시 클러스터 만들기
                                                                                        # 함수 안에 함수 계속 반복: K가 1일 때까지
        # Make new outputs, if necessary
        corrNew, clstrsNew, silhNew = makeNewOutputs(matrix,\
                                                     {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, \
                                                     clstrs2)       # 이미 선택된 것과 새로 선택된 것으로 다시 corr 등 만들기
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/ \
                                np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew


def asset_clustering(df: pd.DataFrame,
                     metric: str,
                     denosing: bool,
                     max_num_clusters: Optional[int],
                     n_init: int=10,
                     ):

    '''

    :param df: (pd.DataFrame)
    :param metric: (str) 'corr', 'mutual' or 'variation'
    :return:
    '''

    if denosing:
        matrix = empirical_denoising(df, q=len(df.columns)/len(df.index), metrics=metric)
    else:
        matrix = empirical_info(df, norm=True, metrics=metric)

    dist_new, clstrs_new, _ = clusterKMeansTop(matrix, maxNumClusters=max_num_clusters, n_init=n_init)

    return clstrs_new