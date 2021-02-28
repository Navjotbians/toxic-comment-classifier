
# coding: utf-8

# ### Model training

# #### Import all the required packages

# In[113]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing
from sklearn.metrics import accuracy_score


# #### load processed dataset

# In[ ]:


df = pd.read_csv('../data/processed/processed_data.csv')


# In[ ]:


### dropping the original unclean comment coloum from dataset
df = df.drop('comment_text', axis = 1)


# In[ ]:


### Renaming the clean comment colum to comment_text for ease
df = df.rename({'clean_comment': 'comment_text'}, axis=1)


# In[ ]:


df.head()


# In[ ]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[ ]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ### Split the date into train test datasets

# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],test_size=0.25,random_state=42)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


# Stats of X_train labels
counts = []
for i in labels:
    counts.append((i, y_train[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# In[ ]:


#stats of X_test labels
counts = []
for i in labels:
    counts.append((i, y_test[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# #### Converting text comments into vectors using bag of words or TF-IDF 

# In[ ]:


def word_embeddings(X_train, X_test, embedding_type = "tfidf"):
    if embedding_type == "bow":
        bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
        X_train = bw_vectorizer.fit_transform(X_train).toarray()
        X_test = bw_vectorizer.fit_transform(X_test).toarray()
    if embedding_type == "tfidf":
        tf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=100)
        X_train = tf_vectorizer.fit_transform(X_train).toarray()
        X_test = tf_vectorizer.fit_transform(X_test).toarray()
    return X_train, X_test


# In[ ]:


Xv_train, Xv_test = word_embeddings(X_train, X_test, "tfidf")


# ### Training

# In[ ]:


### Linear regression 
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))
    logreg.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = logreg.predict(Xv_test)
    print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))


# <br> Checked the impact of use of Bag of words and TF-IDF on the accuracy of Linear regression. 
# <br>In this use case accuracy:
# <br>Remained same - Identity hate, threat
# <br>Almost same - Severe_toxic
# <br>Little bit improved with the use of TF-IDF but not very significant change - Toxic, Obscene, insult

# In[ ]:



    
        

