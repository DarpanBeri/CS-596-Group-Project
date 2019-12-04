#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('kanye_lyrics.csv', encoding = 'latin-1')


# In[ ]:


pd.options.display.max_colwidth = 5000


# In[ ]:


df.head()


# In[ ]:


songs = df


# In[ ]:


len(songs)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


import nltk
from nltk.corpus import stopwords


# In[ ]:


nltk.download('stopwords')


# In[ ]:


stop_words = stopwords.words('english')


# In[ ]:


stop_words.extend(['could', 'really', 'many', 'go', 'let', 'know', 'say', 'get', 'gon','things', 'thing', 'shit', 'ever', 'got', 'man', 'hey', 'uh', 'it', "it's", "like",'oh', 'gonna', 'yo', 'thing', 'huh', 'yeah', 'ya'])


# In[ ]:


vectorizer = TfidfVectorizer(stop_words =  stop_words, min_df = 0.1)


# In[ ]:


tfidf = vectorizer.fit_transform(songs['lyric'])


# In[ ]:


from sklearn.decomposition import NMF


# In[ ]:


nmf = NMF(n_components = 6)


# In[ ]:


topic_values = nmf.fit_transform(tfidf)


# In[ ]:


for topic_num, topic in enumerate(nmf.components_):
    message = "Topic #{}: ".format(topic_num + 1)
    message += " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-11 :-1]])
    print(message)


# In[ ]:


topic_labels = ['religion', 'love/women', 'heartbreak', 'trap lyfe', 'struggle', 'fame'] 


# In[ ]:


df_topics = pd.DataFrame(topic_values, columns = topic_labels)
df_topics


# In[ ]:


songs = songs.join(df_topics)


# In[ ]:


songs.head()


# In[ ]:


songs.loc[songs['trap lyfe'] >= 0.1, 'trap lyfe'] = 1
songs.loc[songs['love/women'] >= 0.1, 'love/women'] = 1
songs.loc[songs['religion'] >= 0.1, 'religion'] = 1
songs.loc[songs['struggle'] >= 0.1, 'struggle'] = 1
songs.loc[songs['fame'] >= 0.1, 'fame'] = 1
songs.loc[songs['heartbreak'] >= 0.1, 'heartbreak'] = 1


# In[ ]:


songs.loc[songs['trap lyfe'] <= 0.1, 'trap lyfe'] = 0
songs.loc[songs['love/women'] <= 0.1, 'love/women'] = 0
songs.loc[songs['religion'] <= 0.1, 'religion'] = 0
songs.loc[songs['struggle'] <= 0.1, 'struggle'] = 0
songs.loc[songs['fame'] <= 0.1, 'fame'] = 0
songs.loc[songs['heartbreak'] <= 0.1, 'heartbreak'] = 0


# In[ ]:


songs[1:len(songs)]


# In[ ]:


year_topics = songs.groupby('year').sum().reset_index()


# In[ ]:


year_topics


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize = (20,10))
plt.plot(year_topics['year'], year_topics['trap lyfe'], label = 'trap lyfe' )
plt.plot(year_topics['year'], year_topics['love/women'], label = 'love/women' )
plt.plot(year_topics['year'], year_topics['religion'], label = 'religion' )
plt.plot(year_topics['year'], year_topics['struggle'], label = 'struggle' )
plt.plot(year_topics['year'], year_topics['fame'], label = 'fame' )
plt.plot(year_topics['year'], year_topics['heartbreak'], label = 'heartbreak' )
plt.grid()
plt.legend()

