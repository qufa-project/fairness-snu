#!/usr/bin/env python
# coding: utf-8

# # 필요 Library 설치

# In[ ]:


get_ipython().system('pip install pandas ')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scikit-learn ')
get_ipython().system('pip install umap-learn ')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap.umap_ as umap


# # 데이터 Description

# - 'generated_imputed.csv` : 생성된 자료 중 일부 결측값을 해당 컬럼의 평균값으로 대치하여 만든 데이터

# In[ ]:


generated_imputed = pd.read_csv("generated_imputed.csv", engine="python")


# In[ ]:


# 데이터 확인
generated_imputed.head()


# # Visualization

# ## 1) Sex 변수를 기준으로 dimension reduction

# In[ ]:


features_sex = generated_imputed.drop("sex", axis=1) # independent variable 
target_sex = generated_imputed["sex"].values # target variable


# In[ ]:


features_sex.head()


# In[ ]:


# t-SNE와 UMAP 알고리즘을 사용하여 2차원으로 차원 축소 
tsne_sex = TSNE(n_components=2).fit_transform(features_sex.values)
umap_sex = umap.UMAP().fit_transform(features_sex.values)


# In[ ]:


tsne_sex.shape


# In[ ]:


umap_sex.shape


# In[ ]:


sex_reduced = pd.DataFrame({"tsne_1" : tsne_sex[:, 0], "tsne_2" : tsne_sex[:, 1],
                            "umap_1" : umap_sex[:, 0], "umap_2" : umap_sex[:, 1],
                            "sex" : target_sex})


# In[ ]:


sex_reduced.head()


# In[ ]:


sex_reduced.to_csv("sex_reduced.csv")


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "orange"})
g = sns.scatterplot(x="tsne_1", y="tsne_2", hue="sex", palette=color_dict, s=100, data=sex_reduced, legend=True)
plt.title("t-SNE")
plt.legend(title="Sex", labels=["Male", "Female"])
plt.show(g)


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "orange"})
g = sns.scatterplot(x="umap_1", y="umap_2", hue="sex", palette=color_dict, s=100, data=sex_reduced, legend=True)
plt.title("UMAP")
plt.legend(title="Sex", labels=["Male", "Female"])
plt.show(g)


# ## 2) DM (당뇨병 진단 여부) 변수를 기준으로 dimension reduction

# In[ ]:


features_dm = generated_imputed.drop("dm", axis=1) # independent variable 
target_dm = generated_imputed["dm"].values # target variable


# In[ ]:


# t-SNE와 UMAP 알고리즘을 사용하여 2차원으로 차원 축소 
tsne_dm = TSNE(n_components=2).fit_transform(features_dm.values)
umap_dm = umap.UMAP().fit_transform(features_dm.values)


# In[ ]:


dm_reduced = pd.DataFrame({"tsne_1" : tsne_dm[:, 0], "tsne_2" : tsne_dm[:, 1],
                            "umap_1" : umap_dm[:, 0], "umap_2" : umap_dm[:, 1],
                            "dm" : target_dm})


# In[ ]:


dm_reduced.to_csv("dm_reduced.csv")


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "red"})
g = sns.scatterplot(x="tsne_1", y="tsne_2", hue="dm", palette=color_dict, s=100, data=dm_reduced, legend=True)
plt.title("t-SNE")
plt.legend(title="Diabetus Mellitus") # labels=["Negative", "Positive"]
plt.show(g)


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "red"})
g = sns.scatterplot(x="umap_1", y="umap_2", hue="dm", palette=color_dict, s=100, data=dm_reduced, legend=True)
plt.title("UMAP")
plt.legend(title="Diabetus Mellitus") # labels=["Negative", "Positive"]
plt.show(g)


# # Single Cluster Analysis 

# - UMAP 결과에서 단독으로 큰 하나의 클러스터가 생성되어있어, 해당 클러스터를 자세히 보기 위함

# ## 1) Sex 기준

# - 위의 UMAP 결과에서 Main cluster 만 따로 확인해보기 위해 DBSCAN 클러스터링 알고리즘 사용

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


sex_reduced.head()


# In[ ]:


cluster = DBSCAN(n_jobs = -1)


# In[ ]:


model_sex = cluster.fit(sex_reduced[["umap_1", "umap_2"]].values)
sex_reduced["DBSCAN_cluster"] = model_sex.labels_


# In[ ]:


# index of cluster with maximum points (main cluster)
sex_umap_main_cluster_idx = np.argmax(np.unique(model_sex.labels_, return_counts = True)[1])


# In[ ]:


sex_umap_main_cluster = sex_reduced.loc[sex_reduced["DBSCAN_cluster"] == sex_umap_main_cluster_idx]


# In[ ]:


sex_umap_main_cluster.head()


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "orange"})
g = sns.scatterplot(x="umap_1", y="umap_2", hue="sex", palette=color_dict, s=100, data=sex_umap_main_cluster, legend=True)
plt.title("UMAP - Main Cluster")
plt.legend(title="Sex", labels=["Male", "Female"])
plt.show(g)


# ## 2) DM 기준

# In[ ]:


model_dm = cluster.fit(dm_reduced[["umap_1", "umap_2"]].values)
dm_reduced["DBSCAN_cluster"] = model_dm.labels_


# In[ ]:


dm_reduced.head()


# In[ ]:


# index of cluster with maximum points (main cluster)
dm_umap_main_cluster_idx = np.argmax(np.unique(model_dm.labels_, return_counts = True)[1])


# In[ ]:


dm_umap_main_cluster = dm_reduced.loc[sex_reduced["DBSCAN_cluster"] == dm_umap_main_cluster_idx]


# In[ ]:


dm_umap_main_cluster.head()


# In[ ]:


plt.figure(figsize=(15,10))
color_dict = dict({1 : "dodgerblue", 2 : "red"})
g = sns.scatterplot(x="umap_1", y="umap_2", hue="dm", palette=color_dict, s=100, data=dm_umap_main_cluster, legend=True)
plt.title("UMAP - Main Cluster")
plt.legend(title="Diabetus Mellitus") # labels=["Negative", "Positive"]
plt.show(g)


# In[ ]:




