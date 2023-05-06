#!/usr/bin/env python
# coding: utf-8

# # creating a reccomendation system for users of IMDB website.
# ## (collaborative filtering)

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movies=pd.read_csv(r"C:\Users\mohad\Downloads\1636213079611301\movies.csv")
ratings=pd.read_csv(r"C:\Users\mohad\Downloads\1636213079611301\ratings.csv")


# In[3]:


movies.head()


# In[4]:


movies['year']=movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies['year']=movies.year.str.extract('(\d\d\d\d)',expand=False)
movies['title']=movies.title.str.replace('(\(\d\d\d\d\))','')
movies['title'] = movies['title'].apply(lambda x: x.strip())
movies.head(10)


# #### we removed genres column because we wont decide based on contents and features of an item.

# In[5]:


movies = movies.drop('genres', 1)
movies.head()


# #### removing timestamp column because we dont need it in recommending process.

# In[6]:


ratings = ratings.drop('timestamp', 1)
ratings.head()


# In[7]:


userInput=[
    {'title':'Heat','rating':8},
    {'title':'Jumanji','rating':5},
    {'title':'GoldenEye','rating':6.7},
    {'title':'Waiting to Exhale','rating':4.3},
    {'title':'Toy Story','rating':7.5}
]
inputMovies=pd.DataFrame(userInput)
inputMovies


# #### to add the movie id to inputMovies df, we search in movies df and take the ones which their name is in the list of the names of inputMovies df. and then we merge two dfs to add the movie id into inputMovies. the result is as kind of movies df so we have to drop  year column.

# In[8]:


inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year', 1)
inputMovies


# #### for collaborative fitering recommendation system we need to consider other users and their ratings.
# #### so to find the subset of users who have watched and rated the input movies, we search into ratings df by id of the movies that our user have rated.

# In[16]:


userSubset = ratings[ratings['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()


# #### we grouped the users by their ids. so the rows that have the same value on userID column will be in one group.

# In[19]:


userSubsetGroup = userSubset.groupby(['userId'])
userSubsetGroup.get_group(6)


# #### then we sorted the user subset by the number of movies they have rated.
# #### lambda function get the userSubsetGroup as parameter and sort the df by the length of the second column 'movieid'. it means the users that share the most movies in common with the input user will get on the top of df.

# In[20]:


userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[:3]


# #### we selected the top 100 users to work with. and we dont need the users that have less movies in common with input user.

# In[21]:


userSubsetGroup = userSubsetGroup[:100]


# #### our system is user-based so we have to find the similarity of selected users to input user.
# #### we used pearson correlation which is an invariant to scaling.
# #### pearson correlation coefficient 1 means that user is similar to input and -1 means opposite.
# #### first we created an empty dictionary where key is user id and its value is similarity coefficient.
# #### then we iterated through userSubsetGroup and sorted every group by movie id and also did the same for inputMovies. 
# #### then we keep the length of each group which is the number of movies that current user has rated. and then we created a new df named temp_df  which is the movies both input and current user have in common. and then we converted the ratings of input and current user into a list.
# #### now it's time to calculate the pearson correlation.
# #### x is considered as input user and y as current user.
# #### sxy is the sum of the product of the deviation of each rating from the mean of ratings.
# #### sxx is the sum of squeres of deviation from mean for input user.
# #### syy is the sum of squeres of deviation from mean for current user.
# #### if denominators are not equal to zero we define the similarity coefficient as Sxy/sqrt(Sxx*Syy) and add it to pearsonCorrelationDict as the value otherwise we add 0.
# #### the whole process will be repeat for each group.
# #### and by the end we'll know how much each user is similar to input user.

# In[23]:


#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0


# In[24]:


pearsonCorrelationDict.items()


# #### we converted the pearsonCorrelationDict to pandas data frame and by defining 'index' for orient, the keys in passed dictionary will become the rows of data frame.
# #### we added the 'userId' and 'similarityIndex' as the columns.
# #### the values of 'similarityIndex' are the values in dictionary and the values of 'userId' are the keys in dictionary.

# In[25]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()


# #### we selected the top 50 of users that have the most similarityIndex. 

# In[26]:


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


# #### to create the weighted average of the ratings of the movies using pearson correlation as weight, we merged the topUsers with ratings to get the movie id and its rating
# #### in fact we want to get the movies that the similar users have watched and rated to recommend to input user.

# In[27]:


topUsersRating=topUsers.merge(ratings, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# #### we multiply similarityIndex as weight to rating and add the result in a new column named 'weightedRating'.

# In[28]:


topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()


# #### we created a temporary df to keep the sum of similarity and sum of the weighted ratings of each movie. in this way we can clearly see that how much each movie has been rated in total and how much is similar to the movies watched by the input user.

# In[29]:


tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()


# #### we created the recommendation data frame and divided the sum of weighted ratings by sum of similarity index to get the wighted average recom score.

# In[34]:


recommendation = pd.DataFrame()
#Now we take the weighted average
recommendation['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation['movieId'] = tempTopUsersRating.index
recommendation.head()


# #### in the final step we searched into movies df by the movie ids of recommended movies to find the information such as title and the release yaer of them.
# #### and here are the top 10 movies we recommend to user.

# In[32]:


movies.loc[movies['movieId'].isin(recommendation.head(10)['movieId'].tolist())]


# In[ ]:




