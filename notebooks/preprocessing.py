
# coding: utf-8

# # Prcessed Data

# This notebook guides through the steps to creating the processed dataset

# In[3]:


import os
dir_path = os.path.dirname(os.getcwd())


# In[4]:


import numpy as np
import pandas as pd
import sys  
sys.path.append(os.path.join(dir_path, 'src'))
from clean_comments import clean
import processing
from processing import process_txt


# In[5]:


train_path = os.path.join(dir_path, 'data', 'raw', 'train.csv')


# In[7]:


## Read the raw data
df = pd.read_csv(train_path)


# In[8]:


### Output of process_txt with stemming
input_str =  df.comment_text[2]
process_txt(input_str, stemm= True)


# In[5]:


### Output of process_txt without stemming
input_str =  df.comment_text[2]
process_txt(input_str)


# ## Create processed_dataset

# In[6]:


def create_processed_df(stamming = False):
        df = pd.read_csv('../data/raw/train.csv')
        df = df.head()

        ### processing each comment and appending it to the dataset
        df['clean_comment'] = df['comment_text'].apply(lambda x:process_txt(x, stemm = stamming))

        ### dropping the original unclean comment coloum from dataset
        df = df.drop('comment_text', axis = 1)

        ### Renaming the clean comment colum to comment_text for ease
        df = df.rename({'clean_comment': 'comment_text'}, axis=1)

        # print("dataset after dropping old comment {}".format(df.comment_text))

        ### Save processed data to 
        df.to_csv(r'F:\AI\Toxic-comment-classifier\data\processed\file1.csv')


# In[7]:


### Uncomment to get processed dataset with stemming
# create_processed_df(stamming=True)


# In[8]:


### uncomment to get processed dataset without stemming
# create_processed_df(stamming=False)


# #### Preview of processed data

# In[9]:


df = pd.read_csv('../data/processed/file1.csv')


# In[10]:


df.head()

