
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import operator
from sklearn import feature_extraction,model_selection,naive_bayes,pipeline,manifold,preprocessing


# In[24]:


# import sys
# print(sys.getrecursionlimit()))


# In[25]:


# sys.setrecursionlimit(5000)


# In[26]:


df = pd.read_csv('../data/raw/train.csv')


# In[27]:


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


# In[29]:


### Frequently used words in the obscene comments
def make_dic(dd, stemm = False,lemm = True):
    #all_word = []
        ### Clean input data
    processed_text = cldan(d)
        ### Tokenization
    processed_text = word_tokenize(processed_text)
     ### remove stop words
    processed_text = [word for word in processed_text if word not in stopwords.words('english')]
    #all_word.append(processed_text)

    ### Stemming
    if stemm == True:
      ps = nltk.stem.porter.PorterStemmer()
      processed_text = [ps.stem(word) for word in processed_text]

    ### Lemmatization
    if lemm == True:
      lem = nltk.stem.wordnet.WordNetLemmatizer()
      processed_text = [lem.lemmatize(word) for word in processed_text]

    text = " ".join(processed_text)
    
    return text

    


# In[31]:


input_str = df.comment_text[2]

make_dict(input_str, stemm= True)


# ## Append the clean comments in the dataset

# In[32]:


# df['clean_comment'] = df['comment_text'].apply(lambda x:make_dict(x, stemm= True))


# In[33]:


#df.head(20)


# #### Reading the processed data

# In[34]:


df = pd.read_csv('../data/processed/processed_data.csv')


# In[35]:


### dropping the original unclean comment coloum from dataset
df = df.drop('comment_text', axis = 1)


# In[36]:


### Renaming the clean comment colum to comment_text for ease
df = df.rename({'clean_comment': 'comment_text'}, axis=1)


# In[37]:


df.head()


# ### Bag of Words

# In[60]:


df_threat = df[df['threat'] == 1]
#Reseting the index
df_threat.set_index(['id'], inplace = True)
df_threat.reset_index(level =['id'], inplace = True)


# In[69]:


corpus = df_threat['comment_text']


# In[70]:


bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)


# In[85]:


X = bw_vectorizer.fit_transform(corpus).toarray()
X.shape


# In[83]:


X[0]


# #### TF -IDF

# In[92]:


tf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=100)


# In[93]:


X1 = tf_vectorizer.fit_transform(corpus).toarray()


# In[94]:


X1.shape


# In[96]:


X1[0]

