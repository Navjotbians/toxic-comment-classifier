
# coding: utf-8

# ### Model training

# #### Import all the required packages

# In[2]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.metrics import accuracy_score, classification_report
import sys  
sys.path.append('F:/AI/Toxic-comment-classifier/src')
from word_embeddings import w_embeddings


# #### load processed dataset

# In[3]:


df = pd.read_csv('../data/processed/processed_stem_data.csv')


# In[4]:


df.head()


# In[5]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[6]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ### Split the date into train test datasets

# In[18]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],test_size=0.25,random_state=42)


# In[19]:


X_train.shape, X_test.shape


# In[20]:


# Stats of X_train labels
counts = []
for i in labels:
    counts.append((i, y_train[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# In[21]:


#stats of X_test labels
counts = []
for i in labels:
    counts.append((i, y_test[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# ### Converting text comments into vectors using bag of words or TF-IDF 

# We know that machine learning models doesn't accept input in the text format. So we need to convert the text data into Vector form, it is also called **Word Embeddings**. Word Embeddings can be broadly classified as:
# 1. Frequency based - Most popular techniques are **Bag-of-Words**, **TF-IDF**
# 2. Pridiction based - Most popular techniques are **Word2vec** and **Glove**
# 
# 

# Here we will be using **Bag-of-Words** and **TF-IDF**<br>
# <br>**Bag-of-Words(BOW)** - To get the embeddings from BOW we will firstly make a dictionary of words is from the test data along with the count of each word occurance in the data, then these words from the dictionary are sorted in descending order of their occurance,put these words into the columns and used as an independent features and here rows are the sentences or samples. These features will have values 0 or 1 based on if the word exists in the sentence.
# <br>
# **Disadvantage** of BOW - Word Embedding we get from BOW have either 0's and 1's as a values, no weights are given to the words according to their importance in the sentence. That means we can not get the sementics of the sentence.

# **TF-IDF** - It stands for Term Frequency - Inverse Document Frequency
# <br> To get embedding with TF-IDF, we calculate Term frequency and Inverse Document Frequency seperate and then multiply them together to get TF-IDF.
# Formulas to calculate TF-IDF: <br>
# **TF** : $$\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}$$ 
# **IDF**:$$log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>
# **TF-IDF**: $$\Bigg[\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}\Bigg]*log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>In **TF-IDF** also, we need dictionary of words with their count of occurance to do the calculation. **TF** assign more weightage to the word which repeat multiple times in the sentance where as **IDF** decreases the weightage to word as number of sentences containing the increases. Here, feature vectors not only contains 0's and 1' but does contain other other values depending on the word importance in the sentence. This is retaining the sementics of the sentence to some extent so it should perform better than BOW.
# <br>Here **TF-IDG** can have zero value for the word which existed in every sentence and give more weightage to less often occured words that means it could cause over-fitting problem but that is yet to discove. 

# In[11]:


Xv_train, Xv_test = w_embeddings(X_train, X_test, "tfidf")


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
# <br>Accuracy remained same - `Identity hate`, `threat`
# <br>Accuracy remained same - `Severe_toxic`
# <br>Accuracy improved little bit  with the use of TF-IDF but not very significant change for `Toxic`, `Obscene`, `insult`

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



    
        

