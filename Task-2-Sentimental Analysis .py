#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Sentiment analysis, also known as opinion mining, is the process of using␣
↪natural language processing (NLP) techniques
to determine the sentiment or emotional tone expressed in text data.
When applied to social media data, sentiment analysis can provide valuable␣
↪insights into public opinion, customer feedback,
brand perception, and more. Here's a brief overview of sentiment analysis using␣
↪social media data:
'''


# In[ ]:


'''
panda,numpy,matplotlib,seaborn,sklearn are the basic libraries used in the
email spam filtering
natural language tool kit used to study the data which means a mail

and visualized the data in the different graphical form(pictorial representation
'''


# Packages required for the Analysis
#   - nltk: natural language tool kit used for text analysis
#   - pandas: used for analyse dataframe
#   - matplotlib and seaborn: used for plotting

# In[3]:


import nltk


# In[4]:


nltk.download('stopwords')


# In[5]:


import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
import string
import re
import matplotlib.pyplot as plt 
import seaborn as sns    
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')


# $step-1$:

# **1.1 Reading the Data**

# In[37]:


df=pd.read_csv('C:\\Users\\gouthami\\Downloads\\Techno Hacks\\Sentimental_Analysis\\Tweets.csv')
df.head(10)


# In[38]:


df.columns


# In[39]:


df.dtypes


# In[40]:


df.shape


# In[41]:


df.isnull().sum()
# understanding any null/missing values are present
# Our main Column here is Review Title which having reviews information
# We can see below it has 7 missing values
# out of 284 we have 7 missing values


# In[42]:


df=df.dropna()
# dropping missing values


# In[43]:


df.isnull().sum()
# after dropping we are checking still any missing value are there
# There is no missing values


# In[44]:


display(df.shape)
display(df.info())


# In[45]:


# redefining dataset for analysis
df=df[['airline_sentiment','text']]
df


# In[15]:


# By above we can see the relationship between the airline_sentiment and the text of reviews


# In[46]:


# airline_sentiment distribution
sns.countplot(data=df,x='airline_sentiment')
plt.title('Graph-1-Airline Sentiment Distribution')
plt.show()


# In[17]:


# creating a new column counting the number of word in each tweets
df['count_word'] = df['text'].apply(lambda x : len(x.split(' ')))
sns.histplot(data = df , x='count_word',kde=True)
plt.title('Graph-2-Number de Word Distribution without any Cleaning Task')
plt.show()


# In[18]:


# word distribution  without cleaning the data
sns.histplot(data = df , x='count_word',hue='airline_sentiment',alpha=0.6,kde=True)
plt.title('Graph-3-Number de Word Distribution without any Cleaning Task')
plt.show()


# In[19]:


# Using the box plots to visualaize the words at tweets more better
sns.boxplot(data = df , y='count_word',x='airline_sentiment')
plt.title('Graph-3(1)=Boxplot Number of Word Across Tweets Categories')
plt.show()


# In[20]:


# Lets see the Negative Tweets
df.loc[np.logical_or(df['count_word']>35,df['count_word']<=5),:]


# **Preprocessing the Data**:

# In[ ]:


- Punctuation Removal
- Stopword Removal
- Numeric Values Removal
- Stemming
- Tokenization


# In[23]:


# import Preprocessing libraries
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[24]:


# Punctuation Removal
def remove_punctuation(text):
    return re.sub(r'[^\w\s]','',text)

#stopword removal
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filter_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filter_tokens)
 
#remove numeric
def remove_numeric(text):
    return re.sub(r'\d+','',text)

#Stemming
def apply_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)
 
def remove_mentions(text):
    return re.sub(r'@\w+','',text)


# In[25]:


import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def apply_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)

input_text = "walking throw the street, a passenger walked toward me, talking about a walked chicken on the streets"
stemmed_text = apply_stemming(input_text)
print(stemmed_text)    


# In[26]:


# General Preprocessing Function
def text_preprocessing(text):
    sentence = remove_mentions(text)
    sentence = remove_punctuation(sentence)
    sentence = remove_stopwords(sentence)
    sentence = remove_numeric(sentence)
    sentence = apply_stemming(sentence)
    return sentence


# In[31]:


text_preprocessing('walking throw the street , a passenger walked toward me, talking about a walked chicken on the streets')


# In[32]:


df.loc[:,'new_text'] = df['text'].apply(lambda x : text_preprocessing(x))


# In[33]:


df.loc[:,'new_count_word'] = df['new_text'].apply(lambda x : len(x.split(' ')))
sns.histplot(data = df , x='new_count_word',kde=True)
plt.title('Graph-5-Number of Word Distribution after Cleaning Task')
plt.show()


# In[34]:


# airline_Sentiment distribution
sns.countplot(data=df,x='airline_sentiment')
plt.title('Graph-1(a)-Airline Sentiment Distribution-after cleaning the data')
plt.show()


# In[35]:


sns.histplot(data = df , x='count_word',hue='airline_sentiment',alpha=0.9,kde=True)
plt.title('Graph-3-Number de Word Distribution after Cleaning Task')
plt.show()


# In[ ]:




