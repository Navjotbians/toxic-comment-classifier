
# coding: utf-8

# ## Data exploration after pre-processing

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import operator


# In[107]:


import matplotlib.pyplot as plt


# In[108]:


# nltk.download('punkt')
# 
nltk.download('stopwords')


# In[109]:


df = pd.read_csv('../input/processed_stem_data.csv')


# In[112]:


df.head()


# In[113]:


df = df.drop('comment_text', axis = 1)


# In[114]:


df.head()


# In[117]:


df = df.rename({'clean_comment': 'comment_text'}, axis=1)


# In[118]:


df.head()


# #### Dataset with toxic comments

# In[119]:


#extract dataset with toxic label
df_toxic = df[df['toxic'] == 1]
#Reseting the index
df_toxic.set_index(['id'], inplace = True)
df_toxic.reset_index(level =['id'], inplace = True)


# In[121]:


#df_toxic.head()


# #### Dataset of severe toxic comments

# In[122]:


#extract dataset with Severe toxic label
df_severe_toxic = df[df['severe_toxic'] == 1]
#Reseting the index
df_severe_toxic.set_index(['id'], inplace = True)
df_severe_toxic.reset_index(level =['id'], inplace = True)
# df_severe_toxic =df_severe_toxic.drop('comment_text', axis=1)


# In[123]:


#df_severe_toxic.head()


# #### Dataset with obscene comment 

# In[124]:


#extract dataset with obscens label
df_obscene = df[df['obscene'] == 1]
#Reseting the index
df_obscene.set_index(['id'], inplace = True)
df_obscene.reset_index(level =['id'], inplace = True)
#df_obscene =df_obscene.drop('comment_text', axis=1)


# In[125]:


#df_obscene.head()


# #### Dataset with comments labeled as "identity_hate" 

# In[126]:


df_identity_hate = df[df['identity_hate'] == 1]
#Reseting the index
df_identity_hate.set_index(['id'], inplace = True)
df_identity_hate.reset_index(level =['id'], inplace = True)


# In[127]:


#df_identity_hate.head()


# #### Dataset with all the threat comments

# In[128]:


df_threat = df[df['threat'] == 1]
#Reseting the index
df_threat.set_index(['id'], inplace = True)
df_threat.reset_index(level =['id'], inplace = True)


# In[129]:


# df_threat.head()


# #### Dataset of comments with "Insult" label

# In[130]:


df_insult = df[df['insult'] == 1]
#Reseting the index
df_insult.set_index(['id'], inplace = True)
df_insult.reset_index(level =['id'], inplace = True)


# In[131]:


# df_insult.head()


# #### Dataset with comments which have all six labels

# In[132]:


df_6 = df[(df['toxic']==1) & (df['severe_toxic']==1) & (df['obscene']==1) & (df['threat']==1)& (df['insult']==1)& (df['identity_hate']==1)]


# In[133]:


df_6.set_index(['id'], inplace = True)
df_6.reset_index(level =['id'], inplace = True) 
# df6 = df_6.drop('comment_text', axis=1)


# In[135]:


# df_6.head()


# In[136]:


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
    input_str = re.sub(r"i'm", "i am", input_str)
    input_str = re.sub(r" m ", " am ", input_str)
    input_str = re.sub('\s+', ' ', input_str)
    input_str = re.sub('\W', ' ', input_str)
    input_str = input_str.translate(str.maketrans('','', string.punctuation))
    input_str = input_str.strip()
    return input_str


# In[137]:


### Frequently used words in the obscene comments
def make_dict(data):
    all_word = []
    counts = dict()
    for i in range (0,len(data)):

        ### Load input
        input_str = data.comment_text[i]

        ### Clean input data
        processed_text = clean(input_str)

        ### perform tokenization
        tokened_text = word_tokenize(processed_text)

        ### remove stop words
        comment_word = []
        for word in tokened_text:
            if word not in stopwords.words('english'):
                comment_word.append(word)
        #print(len(comment_word))
        all_word.extend(comment_word)
      
    for word in all_word:
      if word in counts:
          counts[word] += 1
      else:
          counts[word] = 1
    
    return all_word, counts


# In[138]:


all_words, word_count = make_dict(df_obscene)


# In[139]:


# Arrange the words in descening order and pick the words with minimum count
def descend_odr(data, min_count):
  all_words, word_count = make_dict(data)
  sorted_d = dict( sorted(word_count.items(), key=operator.itemgetter(1),reverse=True))
  for m,n in sorted_d.items():
    if n > min_count:
      print (m,n)


# In[140]:


descend_odr(df_toxic, 1000)


# <br>These are the words most frequently used in toxic comments

# In[141]:


descend_odr(df_severe_toxic, 500)


# <br>These are the words most frequently used in severe toxic comments

# In[142]:


descend_odr(df_obscene, 1000)


# <br>These are the words most frequently used in obscene comments

# In[144]:


descend_odr(df_threat, 50)


# <br>These are the words most frequently used in severe threat comments

# In[145]:


descend_odr(df_insult,1000)


# <br>These are the words most frequently used in comments labeled as an insult

# In[146]:


descend_odr(df_identity_hate,150)


# <br>These are the most frequently used words in the comments labeled as identity_hate

# In[147]:


descend_odr(df_6,5)


# <br>These are the most frequently used words in the comments labeled as identity_hate
