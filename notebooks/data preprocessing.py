
# coding: utf-8

# ## Data Preprocessing

# In[3]:


import numpy as np
import pandas as pd
import re
import string


# In[4]:


df = pd.read_csv('../Input/train.csv')


# In[5]:


input_str = df.comment_text[0]
input_str


# In[6]:


## Convert in lower case
input_str = input_str.lower()
print(input_str)


# In[7]:


### Remove numbers
result = re.sub(r'\d+', '', input_str)
print(result)


# In[8]:


### Remove punctuation
result = input_str.translate(str.maketrans('','', string.punctuation))
print(result)


# In[9]:


### Remove white spaces (removes tabs)
input_str = input_str.strip()
input_str


# In[10]:


### data cleaner function
def clean(input_str):
    input_str = input_str.lower()
    input_str = re.sub(r'\d+', '', input_str)
    input_str = re.sub(r"n't", " not ", input_str)
    input_str = re.sub(r"can't", "cannot ", input_str)
    input_str = re.sub(r"what's", "what is ", input_str)
    input_str = re.sub(r"\'s", " ", input_str)
    input_str = re.sub(r"\'ve", " have ", input_str)
    input_str = re.sub(r"\'re", " are ", input_str)
    input_str = re.sub(r"\'d", " would ", input_str)
    input_str = re.sub(r"\'ll", " will ", input_str)
    input_str = re.sub(r"\'scuse", " excuse ", input_str)
    input_str = re.sub(r"I'm", "I am", input_str)
    input_str = re.sub(r" m ", " am ", input_str)
    input_str = re.sub('\s+', ' ', input_str)
    input_str = re.sub('\W', ' ', input_str)
    input_str = input_str.translate(str.maketrans('','', string.punctuation))
    input_str = input_str.strip()
    return input_str


# In[11]:


st = df.comment_text[1]
clean(st)

