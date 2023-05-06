#!/usr/bin/env python
# coding: utf-8

# # creating a reccomendation system for users of IMDB website.
# ## (content-based)

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


movies=pd.read_csv(r"C:\Users\mohad\Downloads\1636213079611301\movies.csv")
ratings=pd.read_csv(r"C:\Users\mohad\Downloads\1636213079611301\ratings.csv")


# In[21]:


movies.head()


# #### extracting the release year of the movie in title column and adding it into a new column named year. and then we remove the year in title column by replacing it with an empty string('').
# #### in the last step we removed any whitespaces at the end and the begining of each title.

# In[68]:


movies['year']=movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies['year']=movies.year.str.extract('(\d\d\d\d)',expand=False)
movies['title']=movies.title.str.replace('(\(\d\d\d\d\))','')
movies['title'] = movies['title'].apply(lambda x: x.strip())
movies.head(10)


# #### we seperated genres of each movie in genres column by '|' with the help of split() function and converted them into a list of genres.

# In[69]:


movies['genres']=movies.genres.str.split('|')
movies.head()


# #### first, we created a new data frame 'moviesWithGenres' in order to have the genre matrix.
# #### and then copied the movies df into it.
# #### iterrows() function iterate through each row of a df and returns the index and date of a row as series.
# #### in the first for loop we iterate through each row and in the nested loop we iterate through the different genres of a movie in the 'genre' column and for each genre we add it as a new column in moviesWithGenres and assign 1 to that location.
# #### and then we fill the empty cells by 0 which means that specific genre does not belong to the corresponding movie.

# In[70]:


moviesWithGenres=movies.copy()
for index,row in movies.iterrows():
    for genre in row['genres']:
        moviesWithGenres.at[index,genre]=1
moviesWithGenres=moviesWithGenres.fillna(0)
moviesWithGenres.head()


# #### we delete 'genres' and 'year' columns because the genres are already present as columns. and in our recommendation system which is content-based we only decides by the genres not the year.

# In[71]:


moviesWithGenres=moviesWithGenres.drop('genres',1)
moviesWithGenres=moviesWithGenres.drop('year',1)
moviesWithGenres.head()


# In[7]:


ratings.head()


# #### removing timestamp column because we dont need it in recommending process.

# In[72]:


ratings=ratings.drop("timestamp",1)
ratings.head()


# #### creating a list of dictionaries which shows the the movies with its ratings that have been watched by a particular user. then converting it into a pandas data frame.

# In[73]:


userInput=[
    {'title':'Heat','rating':8},
    {'title':'Jumanji','rating':5},
    {'title':'GoldenEye','rating':6.7},
    {'title':'Waiting to Exhale','rating':4.3},
    {'title':'Toy Story','rating':7.5}
]
inputMovies=pd.DataFrame(userInput)
inputMovies


# #### to add the movie id to inputMovies df, we search in movies df and take the ones which their name is in the list of the names of inputMovies df. and then we merge two dfs to add the movie id into inputMovies. the result is as kind of movies df so we have to drop genres and year columns.

# In[74]:


inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]
inputMovies=pd.merge(inputId,inputMovies)
inputMovies=inputMovies.drop('genres',1).drop('year',1)
inputMovies


# #### now we have to find the genres of the user's movies. so we search into moviesWithGenres df by id and take the ones which their id is present in the list of id's of user's watched movies.

# In[75]:


userMoviesWithGenre = moviesWithGenres[moviesWithGenres['movieId'].isin(inputMovies['movieId'].tolist())]
userMoviesWithGenre


# #### we do some cleaning, reset the index of df and keep only genres columns.

# In[76]:


userMoviesWithGenre = userMoviesWithGenre.reset_index(drop=True)
userGenreTable = userMoviesWithGenre.drop('movieId', 1).drop('title', 1)
userGenreTable


# #### now we have to multiply the user's ratings to the genres of corresponding movie to obtain weighted genre matrix and finally user profile.

# In[116]:


userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile


# #### now we get the genres of every movie we have. and the index of row based on movie id column.
# #### and then remove the unnecessary columns.

# In[117]:


genreTable = moviesWithGenres.set_index(moviesWithGenres['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1)
genreTable.head()


# #### now we have to multiply user profile which indicates user prefrences to genre matrix to figure out how much each genre user likes and which movies to recommend.

# In[120]:


recommendationTable = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable.head()


# #### we sorted the recommendation table by the score that is assigned to movies that user has not watched yet. then we can use n top rows to recommend to the user.

# In[121]:


recommendationTable = recommendationTable.sort_values(ascending=False)
recommendationTable.head()


# #### to find the place of top 20 movies, we search into movies df by id and now we can see the whole information about the recommended movies.

# In[122]:


movies.loc[movies['movieId'].isin(recommendationTable.head(20).keys())]


# In[ ]:




