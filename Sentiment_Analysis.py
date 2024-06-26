#!/usr/bin/env python
# coding: utf-8

# # Import All Libraries & Comments

# In[237]:


#import libraires
import pandas as pd
import numpy as np
import re
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[238]:

print('\nImporting All Comments')

#load data
df = pd.read_csv("all_comments.csv")
df_original = df

print('\nAll Comments Imported')

# # Data Cleaning

# In[239]:

print('\nDoing Data Cleaning')
def clean_text(text):
    text = text.lower()
    text = re.sub('rt[\s]+', '', text)
    text = re.sub('@[A-Za-z0-9]+', '', text)
    text = re.sub('#', '', text)
    text = re.sub('&amp;', '', text)
    text = re.sub(r'[\r|\n|\r\n]+', '', text)
    text = re.sub('https:?\/\/\S+', '', text)
    text = re.sub(r'[:_!?,;-]', '', text)
    return text
df['comment'] = df['comment'].apply(str)
df['source'] = df['source'].apply(str)
df['comment'] = df['comment'].apply(clean_text)
df['source'] = df['source'].apply(clean_text)
df.head() 


# In[240]:


#make a separate dataframe for amazon
amazon = df.loc[df.source.str.contains("amazon", na=False)]


# In[241]:


#Delete rows of amazon, all empty comments and duplicates
df = df[~df.source.str.contains("Amazon", na=False)]
df = df[~df.source.str.contains("amazon", na=False)]
df = df[~df.source.str.contains("source", na=False)]
df = df[~df.source.str.contains("Source", na=False)]
df = df[df.comment != '']
df = df.dropna(subset=['comment'])
df.drop_duplicates(subset=['comment'])
amazon.drop_duplicates(subset=['comment'])
print('\nData Cleaning Complete')

# In[242]:

# # Perform Sentiment Analysis

# In[243]:

print('\nPerforming Sentiment Analysis')
#import sentiment library textblob
from textblob import TextBlob

#get polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

#create subj and polarity columns in data frame
df['textblob_polarity'] = df['comment'].apply(get_polarity)


# In[244]:


#import sentiment library vader
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

#get vader polarity
def get_vader_polarity(text):
    return sentiment.polarity_scores(text)
df['scores'] = df['comment'].apply(get_vader_polarity)
df['vader_polarity']  = df['scores'].apply(lambda score_dict: score_dict['compound'])


# In[245]:


df['polarity'] = round((df['textblob_polarity'] + df['vader_polarity'])/2,2)
df['score'] = (df['polarity']+1)*2+1

print('\nSentiments Calculated.. Exporting Results')
# In[246]:


amazon['score'] = amazon['rating']


# In[247]:


final_df = pd.concat([df, amazon], axis=0)


# # Categorize according to Polarity
# 
# All comments have been scored on a scale of 1 to 5
# 
# Categorisation criteria:
# 
# If < 3 then Negative, If =3 then Neutral, If >3 then Positive

# In[248]:


#categorize score into pos, neg, neutral

def category(polarity):
    if polarity < 3:
        return 'Negative'
    elif polarity >3:
        return 'Positive'
    else:
        return 'Neutral'
    
final_df['category'] = final_df['score'].apply(category)


# In[249]:


final_df = final_df[['comment','date','source','category', 'score']]
final_df = final_df[final_df.source != '']
final_df = final_df.dropna(subset=['source'])
final_df = final_df[final_df.source != 'nan']
final_df = final_df[final_df.comment != 'nan']


# In[250]:


#Count the number of pos, neg & neutral tweets
print('\nResults snapshot')
print('\n--------------------------------')
print(final_df['category'].value_counts())
print('')
print('--------------------------------')




# # Generate Output File

# In[252]:


#export file
#import time
from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M")
filename = 'output/final_sentiment_analysis_' + timestampStr 
final_df.to_csv("{}.csv".format(filename),index=False)
print('\nResults file exported.. View the latest file in output folder')

# In[ ]:




