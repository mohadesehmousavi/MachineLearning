#!/usr/bin/env python
# coding: utf-8

# # clustering customers of a company based on their information.

# In[1]:


import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt   
get_ipython().run_line_magic('matplotlib', 'inline')


# In[239]:


df=pd.read_csv(r"C:\Users\mohad\Downloads\1632560262896716.csv")
df.head()


# ### removing Gender feature because k_means algorithm is not directly applicable on categorical features.

# In[240]:


from sklearn import preprocessing
df=df.drop('Gender',1)
df.head()


# ### scaling data on normal distribution chart

# In[90]:


from sklearn.preprocessing import StandardScaler
x = df.values[:,1:]
x = np.nan_to_num(x)
df = StandardScaler().fit_transform(x)
df


# ### Evaluating the optimal k

# In[94]:


from sklearn.cluster import KMeans
clusterNum = [2,3,4,5,6]
sse = []
meanSSE=[]
for k in clusterNum:
    k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
    k_means.fit(df)
    centroids = k_means.cluster_centers_
    for i in range (len(df)):
        sse.append((df[i, 0] - centroids[0]) ** 2 + (df[i, 2] - centroids[1]) ** 2)
    meanSSE.append(np.mean(sse))
plt.plot(clusterNum,meanSSE,color='blue')
plt.xlabel('k')
plt.ylabel('mean of distances data points from centroid')
plt.show()


# ### As shown above the plot looks like an arm with a clear elbow at k = 3.

# ### creating k_means model

# In[95]:


df=pd.DataFrame(df,columns=['Age','Annual Income','Spending Score'])
df.head()


# In[96]:


k=5
k_means = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means.fit(df)


# In[97]:


labels=k_means.labels_
print(labels)
df['cluster'] = labels
df['cluster'].value_counts()
df.head()


# In[98]:


df=df.dropna()


# ### visualization
# 

# In[127]:


area = np.pi * ( df.iloc[:,2])**2  
plt.scatter(df.iloc[:,0], df.iloc[:,1], s=area*3, c=labels.astype(np.float), alpha=0.9)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Annual income', fontsize=16)
plt.show()


# In[128]:


area = np.pi * ( df.iloc[:,1])**2  
plt.scatter(df.iloc[:,0], df.iloc[:,2], s=area*3, c=labels.astype(np.float), alpha=0.9)
plt.xlabel('Age', fontsize=18)
plt.ylabel('spending score', fontsize=16)
plt.show()


# #### it seems that purple cluster differ from dark green cluster in terms of height(z axis). so we'll plot it in a 3D area. also in this case, our dataset is clustered almost with no overlap and the amount of noises has been significantly reduced.

# In[102]:


area = np.pi * ( df.iloc[:,0])**2  
plt.scatter(df.iloc[:,1], df.iloc[:,2], s=area*3, c=labels.astype(np.float), alpha=0.9)
plt.xlabel('Annual income', fontsize=18)
plt.ylabel('spending score', fontsize=16)
plt.show()


# In[123]:


fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
plot_geeks = ax.scatter(df.iloc[:, 1], df.iloc[:, 0], df.iloc[:, 2], c= labels.astype(np.float))
#plt.cla()
ax.set_title("3D plot")
ax.set_xlabel('Annual income')
ax.set_ylabel('Spending score')
ax.set_zlabel('Age')
plt.show()


# ## hierarchy

# ### calculating distance matrix

# In[195]:


df=pd.read_csv(r"C:\Users\mohad\Downloads\1632560262896716.csv")
df.head()


# In[196]:


from sklearn import preprocessing
df=df.drop('Gender',1)
df.head()


# In[197]:


from sklearn.preprocessing import StandardScaler
x = df.values[:,1:]
x = np.nan_to_num(x)
dfScaled = StandardScaler().fit_transform(x)
dfScaled


# In[198]:


from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(dfScaled,dfScaled) 
print(distance_matrix)


# ### creating agglomerative model

# In[199]:


from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters = 5, linkage = 'complete')
agglom.fit(distance_matrix)

agglom.labels_


# In[200]:


df['cluster'] = agglom.labels_
df.head()


# In[201]:


df.rename(columns = {'Annual Income (k$)':'AnnualIncome','Spending Score (1-100)':'SpendingScore'}, inplace = True)
df.head()


# ### visualization

# In[217]:


import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster == label]
    for i in subset.index:
            plt.text(subset.AnnualIncome[i], subset.SpendingScore[i],str(subset['Age'][i]), rotation=25) 
    plt.scatter(subset.AnnualIncome, subset.SpendingScore, s= subset.Age*5, c=color, label='cluster'+str(label),alpha=0.6)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters',fontsize=22)
plt.xlabel('Annual income',fontsize=22)
plt.ylabel('Spending Score',fontsize=22)
plt.show()


# ### it has more noise than the previous clustering.

# In[218]:


import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster == label]
    for i in subset.index:
            plt.text(subset.Age[i], subset.AnnualIncome[i],str(subset['SpendingScore'][i]), rotation=25) 
    plt.scatter(subset.Age, subset.AnnualIncome, s= subset.SpendingScore*5, c=color, label='cluster'+str(label),alpha=0.6)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters',fontsize=22)
plt.xlabel('Age',fontsize=22)
plt.ylabel('Annual Income',fontsize=22)
plt.show()


# ### it has a lot of noise and is not clustered very accurately.
# ### the first clustering which we considered the relation between annual income and spending score is more accurate and logical. 

# In[219]:


import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster == label]
    for i in subset.index:
            plt.text(subset.Age[i], subset.SpendingScore[i],str(subset['AnnualIncome'][i]), rotation=25) 
    plt.scatter(subset.Age, subset.SpendingScore, s= subset.AnnualIncome*5, c=color, label='cluster'+str(label),alpha=0.6)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters',fontsize=22)
plt.xlabel('Age',fontsize=22)
plt.ylabel('Spending Score',fontsize=22)
plt.show()


# In[242]:


df.dropna()
df.head()


# In[249]:


from sklearn.preprocessing import StandardScaler
x = df.values[:,1:]
x = np.nan_to_num(x)
dfScaled = StandardScaler().fit_transform(x)
dfScaled


# ## DBSCAN

# ### creating DBSCAN model

# In[290]:


from sklearn.cluster import DBSCAN 
epsilon = 0.4
minimumSamples = 7
dbscan = DBSCAN(eps=epsilon, min_samples=minimumSamples)
dbscan.fit(dfScaled)
labels = dbscan.labels_
labels


# ### number of clusters without noise

# In[291]:


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_


# ### getting labels

# In[292]:


unique_labels = set(labels)
unique_labels


# ### detecting core data points

# In[293]:


core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
core_samples_mask


# ### visualization

# In[294]:


colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = df[class_member_mask & core_samples_mask]
    #print(xy)
    #print('\n')
    plt.scatter(xy.iloc[:, 1], xy.iloc[:, 2],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = df[class_member_mask & ~core_samples_mask]
    plt.scatter(xy.iloc[:, 1], xy.iloc[:, 2],s=50, c=[col], marker=u'o', alpha=0.5)


# In[ ]:





# In[ ]:




