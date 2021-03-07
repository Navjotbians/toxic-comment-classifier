
# coding: utf-8

# ### Model training

# #### Import all the required packages

# In[39]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.metrics import accuracy_score, classification_report


# #### load processed dataset

# In[54]:


df = pd.read_csv('../data/processed/processed_stem_data.csv')


# In[55]:


df.head()


# In[56]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[57]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ### Split the date into train test datasets

# In[58]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],test_size=0.25,random_state=42)


# In[59]:


X_train.shape, X_test.shape


# In[60]:


# Stats of X_train labels
counts = []
for i in labels:
    counts.append((i, y_train[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# In[61]:


#stats of X_test labels
counts = []
for i in labels:
    counts.append((i, y_test[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# #### Converting text comments into vectors using bag of words or TF-IDF 

# In[62]:


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


# In[63]:


Xv_train, Xv_test = word_embeddings(X_train, X_test, "tfidf")


# ### Training

# In[52]:


### Linear regression 
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))
    logreg.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = logreg.predict(Xv_test)
    print('Validation accuracy is \n {}'.format(classification_report(y_test[label], prediction)))


# <br> Checked the impact of use of Bag of words and TF-IDF on the accuracy of Linear regression. 
# <br>In this use case accuracy:
# <br>Remained same - Identity hate, threat
# <br>Almost same - Severe_toxic
# <br>Little bit improved with the use of TF-IDF but not very significant change - Toxic, Obscene, insult

# In[64]:


### Naive bayes
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
    nbayes.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = nbayes.predict(Xv_test)
    print('Validation accuracy is \n {}'.format(classification_report(y_test[label], prediction)))


# In[ ]:



    
        

