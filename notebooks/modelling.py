
# coding: utf-8

# ### Model training

# #### Import all the required packages

# In[ ]:


import os
dir_path = os.path.dirname(os.getcwd())


# In[59]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pickle
import sys  
sys.path.append(os.path.join(dir_path, "src"))
from word_embeddings import get_embeddings
from clean_comments import clean
from processing import process_txt


# #### load processed dataset

# In[3]:


processed_data = os.path.join(dir_path, 'data', 'processed', 'processed_stem_data.csv')


# In[4]:


df = pd.read_csv(processed_data)


# In[5]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[6]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ### Split the date into train test datasets

# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],
                                                                    test_size=0.25,random_state=42)


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
# Formulas to calculate TF-IDF: 
# <br>
# <br>**TF :**  $$\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}$$ 
# **IDF :**$$log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>
# **TF-IDF :** $TF * IDF$ $$\Bigg[\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}\Bigg]*log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>In **TF-IDF** also, we need dictionary of words with their count of occurance to do the calculation. **TF** assign more weightage to the word which repeat multiple times in the sentance where as **IDF** decreases the weightage to word as number of sentences containing the increases. Here, feature vectors not only contains 0's and 1' but does contain other other values depending on the word importance in the sentence. This is retaining the sementics of the sentence to some extent so it should perform better than BOW.
# <br>Here **TF-IDG** can have zero value for the word which existed in every sentence and give more weightage to less often occured words that means it could cause over-fitting problem but that is yet to discove. 

# We will try Bag-of-Words and TF-IDFto get our features for `X_train`  and `X_test` data. The resultant embeddings are in numpy array format, if we have a look at the the embeddings we will know it is high dimensional sparse data.

# In[ ]:


pkl_file = os.path.join(dir_path, 'model','bw_vectorizer1000.pkl' )


# In[ ]:


# Save vectorizer
file = open(pkl_file,"wb")
pickle.dump(bw_vectorizer,file)
file.close()


# In[ ]:


# ### Open the saved vectorizer
# open_file = open(pkl_file, "rb")
# bw_vectorizer = pickle.load(open_file)
# open_file.close()


# In[ ]:


Xv_train = train_feat


# In[ ]:


Xv_test = test_feat


# ### Matrix used to evaluate the models

# In[32]:


def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
    return jaccard.mean()*100


# In[33]:


def print_score(y_pred, y_test, clf):
    print("Clf: ",clf.__class__.__name__)
    print("Jaccard score: {}".format(j_score(pd.DataFrame(y_test), pd.DataFrame(y_pred))))
    print("F1 Score : {}".format(f1_score(y_test, y_pred,average='macro')))
    


# ### Training

# In[34]:


### OneVsRestClassifier
def train_model(classifier,X, y, max_feature = 1000, embedding= 'bow' ):

    #Train-test split
    print("... Performing train test split")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,
                                                                    test_size=0.25,random_state=42)
    
    ## Features extraction with word embedding
    print("... Extracting features")
    Xv_train, Xv_test, vectorizer = get_embeddings(X_train, X_test,
                                                          max_feature = max_feature , embedding_type= embedding)
    
    # train the model 
    print('... Training {} model'.format(classifier.__class__.__name__))
    clf = OneVsRestClassifier(classifier)
    clf.fit(Xv_train, y_train)

    # compute the test accuracy
    print("...Computing accuracy")
    prediction = clf.predict(Xv_test)

    ## Accuracy score
    score = (accuracy_score(y_test, prediction))
    type2_score = j_score(y_test, prediction)
    f1_s = f1_score(y_test, prediction,average='macro')
    roc_auc = roc_auc_score(y_test, prediction)
    confusion_matrix = multilabel_confusion_matrix(y_test, prediction)
    score_sumry = [score, type2_score, f1_s, roc_auc]
    
    
    ## Save model
    print("...Saving model in model directory")
    pkl_file = os.path.join(dir_path,'model', classifier.__class__.__name__)
    file = open(pkl_file,"wb")
    pickle.dump(clf,file)
    file.close()
    
     #### Prediction on comment 

    input_str = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]
    input_str = clean(input_str[0])
    input_str = process_txt(input_str, stemm= True)
    input_str = vectorizer.transform([input_str])
    

    print('\n')
    print("Model evaluation")
    print("------")
    print(print_score(prediction,y_test, classifier))
    print('Accuracy is {}'.format(score))
    print("ROC_AUC - {}".format(roc_auc))
    print(print("check model accuracy on input_string {}".format(clf.predict(input_str))))
    print("------")
    print("Multilabel confusion matrix \n {}".format(confusion_matrix))
    
    return clf, vectorizer, score_sumry
    


# In[35]:


n_bayes = naive_bayes.MultinomialNB()
logreg = LogisticRegression(solver='sag')


# #### With simple Train-Test split

# Which model to use?
# <br>
# Which is the fastest model for high dimensional sparse data?  - **Logistic regression**
# We will use Logistic regression for this dataset to start with and the solver we are using is 'sag' as it is faster for large datsets.

# In[56]:


lr_clf, lr_vectorizer, lr_sumry = train_model(logreg, corpus, df[labels])


# In[58]:


summary_lr = pd.DataFrame(lr_sumry, index = ['accuracy', 'jaccard score', 'F1_score', 'roc_score'], columns= [lr_clf.estimators_[0]])
summary_lr


# <br>**Naive Bayes** is quite populer with text data problems. It learns the parameters by looking at each feature individually and collect simple per-class stats from each feature.
# We are going to use MultinomialNB because it assumes count data, that means, each feature represents an integer count of some-thing, in our problem- how often a word appears in a sentence.

# In[36]:


nb_clf, nb_vectorizer, nb_sumry = train_model(n_bayes, corpus, df[labels] )


# In[51]:


summary_nb = pd.DataFrame(nb_sumry, index = ['accuracy', 'jaccard score', 'F1_score', 'roc_score'], columns= [nb_clf.estimators_[0]])
summary_nb


# <br>
# Lets do testing on unseen data

# In[52]:


input_str = ["that is so good, i am so happy bitch"]
input_str = clean(input_str[0])
input_str = process_txt(input_str, stemm= True)
input_str = nb_vectorizer.transform([input_str])
input_str


# In[53]:


### Open the saved mode.pkl
pkl_file = os.path.join(dir_path, 'model', 'MultinomialNB')
open_file = open(pkl_file, "rb")
model = pickle.load(open_file)
open_file.close()


# In[54]:


model


# In[55]:


model.predict(input_str)


# # Check if it needs to stay here
# **`Bag-of-Words` and `TF-IDF`**
# <br>Accuracies for `Bag-of-Words` and `TF-IDF` are not same but their difference is also not very significant for both Linear regression and Naive Bayes 
# <br>Accuracy remained same - `Identity hate`, `threat`
# <br>Accuracy remained almost same - `Severe_toxic`
# <br>Accuracy improved little bit  with the use of TF-IDF but not very significant change for `Toxic`, `Obscene`, `insult`.
# One possible reason for not seeing the expected significant improvement with the use of `TF-IDF` could be - the human raters didn't care for the semantics of the sentances and rated the comment based on the presence of toxic words.
# In this case, we will choose `Bag-of-Words` because the performance is almost same as `TF-IDF` but less chance of overfitting.
# 

# **Naive Bayes or Logistic regression !!**
# <br>Compairing the confusion matrixs and Jaccard score, Naive Bayes clearly out performed Linear regression and Naive Bayes even tends to get trained faster.
# Just yet we can't decide, we need to try different model evaluation techniqes first.

# ### K FOLD CROSS VALIDATION

# #### K Fold cross validation with Gridsearch for Logistic Regression 

# We will use GridSearchCV to evaluate the model using different values of **C**

# In[9]:


param_grid = {"estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]}


# In[10]:


#Train-test split
print("... Performing train test split")
X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],
                                                                    test_size=0.25,random_state=42)


# In[11]:


## Features extraction with word embedding
print("... Extracting features")
Xv_train, Xv_test, vectorizer = get_embeddings(X_train, X_test,
                                                          max_feature = 1000 , embedding_type= 'bow')


# In[14]:


# clf_lr = OneVsRestClassifier(logreg)


# In[15]:


# gs_lr = GridSearchCV(clf_lr, param_grid ,scoring = 'f1_micro', cv=3)


# In[16]:


# gs_lr.fit(Xv_train, y_train)


# In[26]:


print("Accuracy score of the best estimator : {}".format(gs_lr.score(Xv_test, y_test )))


# In[28]:


print("Best estimator is : {}".format(gs_lr.best_estimator_))


# <br>
# As we can see that *Logistic Regression* performed well with **C = 0.1**. on 3 Folds of cross validation. We will use this value with the various combination of *maximum_features* and *embedding type* in next part to check the performance of *Logistic Regression*

# In[66]:


#### Naive Bayes with K-fold cross validation


# In[68]:


### Naive bayes
nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
nbayes.fit(Xv_train, y_train)
cv_score= cross_val_score(nbayes,Xv_train,y_train,cv=3)
print('Naive Bayes Cross Validation score = {}'.format(cv_score))


# <br>
# We can see even with the K-fold cross validation *Naive Bayes* is giving consistant results for the accuracy

# ## Best Model Selection

# In[72]:


logreg = LogisticRegression(solver='sag', C= 0.1)
classifier = [logreg, n_bayes]
max_feature = [1200, 1500, 1700, 2000]
embedding= ['bow', 'tfidf']


# In[73]:


summary_report =  []
combinations =[]
best_jscore = 1
for estimator in classifier:
  print(".... Processing {}".format(estimator))
  for m in max_feature:
    for n in embedding:
      print(".... Combination {}, {} ".format(m,n))
      combinations.append((estimator.__class__.__name__, m, n))
      clf, vectorizer, score_sumry = train_model(estimator, corpus, df[labels], max_feature=m, embedding= n )
      summary_report.append(score_sumry)
      if score_sumry[1] > best_jscore:
        best_jscore =score_sumry[1]
        best_param = {"classifier": clf.__class__.__name__, "max_features": m, "embedding_type":n}
      
      
print(summary_report)
print(best_param)      


# In[74]:


summary_r = pd.DataFrame(summary_report, index = [combinations], columns= ['accuracy', 'jaccard score', 'F1_score', 'roc_score'],)
summary_r


# In[75]:


summary_all_combinations = summary_r.sort_values(by=['jaccard score'],ascending=False)
summary_all_combinations


# with **C =0.1** logistic regression is doing much worst that **C= 1**

# ### Stratified K Fold Cross Validation

# In[ ]:


Xc = corpus


# **Logistic Regression**

# In[ ]:


skf_lr_accuracy = []
for label in labels:
    print('\n... PROCESSING {}'.format(label.upper()))
    # train the model
    clf = logreg

    
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(Xc,df[label])
    
    score_lr_acc = []
    score_lr_jaccard = []
    score_lr_roc = []
    score_lr_f1 = []
    i = 1
    for train_index, test_index in skf.split(Xc,df[label]):
        Y = df[label]
        #print("Train:", train_index, "validation:", test_index)
        X1_train, x1_test = Xc.iloc[train_index], Xc.iloc[test_index]
        y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]

        ### Convert into word embeddings
        train_feat, test_feat, vectorizer = get_embeddings(X1_train,x1_test)
#       bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
#       X1_train = bw_vectorizer.fit_transform(X1_train).toarray()
#       x1_test = bw_vectorizer.fit_transform(x1_test).toarray()

        clf.fit(train_feat, y1_train)
        prediction_lr = clf.predict(test_feat)
        
        acc_lr_score = accuracy_score(y1_test,prediction_lr)
        score_lr_acc.append(acc_lr_score)
        
        jaccard_lr = j_score(pd.DataFrame(y1_test),pd.DataFrame(prediction_lr))
        score_lr_jaccard.append(jaccard_lr)
        
        roc_lr = roc_auc_score(y1_test,prediction_lr)
        score_lr_roc.append(roc_lr)
        
        print("----- Processed {} fold".format(i))
        print(print_score(prediction_lr, y1_test, clf))
        
        i = i+1
        
    print("Model evaluation")
    print("------")
    print("ROC_AUC - {}".format(score_lr_roc))
    print('Average accuracy for 5 Kfolds in {} category {}'.format(label, np.array(score_lr_acc).mean()))
    print("------")
    #print(type(score))
    skf_lr_accuracy.append(np.array(score_lr_acc).mean())
print("\n Mean accuracies of all six labeles: {}".format(skf_lr_accuracy))
    


# <br>
# **Naive Bayes**

# In[ ]:


classifier = n_bayes
skf_nb_accuracy = []
for label in labels:
    print('\n... PROCESSING {}'.format(label.upper()))
    # train the model
    classifier = n_bayes

    
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(Xc,df[label])
    
    score_label = []
    score_nb_jaccard = []
    score_nb_roc = []
    score_nb_f1 = []
    i = 1
    for train_index, test_index in skf.split(Xc,df[label]):
        Y = df[label]
        #print("Train:", train_index, "validation:", test_index)
        X1_train, x1_test = Xc.iloc[train_index], Xc.iloc[test_index]
        y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]

        ### Convert into word embeddings
        train_feat, test_feat, vectorizer = get_embeddings(X1_train,x1_test)
#       bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
#       X1_train = bw_vectorizer.fit_transform(X1_train).toarray()
#       x1_test = bw_vectorizer.fit_transform(x1_test).toarray()

        classifier.fit(train_feat, y1_train)
        prediction = classifier.predict(test_feat)
        
        acc_score = accuracy_score(y1_test,prediction)
        score_label.append(acc_score)
        
        jaccard_nb = j_score(pd.DataFrame(y1_test),pd.DataFrame(prediction))
        score_nb_jaccard.append(jaccard_nb)
        
        roc_nb = roc_auc_score(y1_test,prediction)
        score_nb_roc.append(roc_nb)
        
        print("----- Processed {} fold".format(i))
        print(print_score(prediction, y1_test, classifier))
        
        i = i+1
        
    print("Model evaluation")
    print("------")
    print("ROC_AUC - {}".format(score_nb_roc))
    print('Average accuracy for 5 Kfolds in {} category {}'.format(label, np.array(score_label).mean()))
    print("------")
    #print(type(score))
    skf_nb_accuracy.append(np.array(score_label).mean())
print("\n Mean accuracies of all six labeles: {}".format(skf_nb_accuracy))
    


# ### Trying multiple values of C in Linear regression model using simple Train-Test split method

# In[ ]:


best_accuracy = {'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 0}
best_c_value = {}
for c in [0.001, 0.01, 0.1, 1, 10, 100]:
   ### Naive bayes
  print('\n... Processing  C= {}'.format(c)) 
  accuracy_lr = []
  for label in labels:
      print('... Processing {}'.format(label))
      # train the model 
      logreg = OneVsRestClassifier(LogisticRegression(C=c ,solver='sag'))
      logreg.fit(Xv_train, y_train[label])
      # compute the testing accuracy
      prediction = logreg.predict(Xv_test)
      score = (accuracy_score(y_test[label], prediction))
      accuracy_lr.append(score)
      if score > best_accuracy[label]:
        best_accuracy[label] = score
        best_c_value[label] = c
      print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
  print("Accuracy_lr: {}".format(accuracy_lr))
print("\n Best accuracy : {}".format(best_accuracy))
print("Best parameter : {}".format(best_c_value))


# In[ ]:


# Output of above cell
# Best accuracy : {'toxic': 0.9107612864412303, 'severe_toxic': 0.9900233123605645, 'obscene': 0.9515955180106785, 'threat': 0.9973679592911037, 'insult': 0.9509688416514176, 'identity_hate': 0.9910259945353821}
# Best parameter : {'toxic': 0.1, 'severe_toxic': 0.001, 'obscene': 0.1, 'threat': 0.001, 'insult': 0.1, 'identity_hate': 0.001}


# In[21]:


##### Old version to try Kfolf CV

# ### OneVsRestClassifier
# for classifier in [n_bayes,logreg]:
    
#     print('... Processing {}'.format(classifier))
    
#     #Train-test split
#     print("... Performing train test split")
#     X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],
#                                                                     test_size=0.25,random_state=42)
    
#     ## Features extraction with word embedding
#     print("... Extracting features")
#     Xv_train, Xv_test, vectorizer = get_embeddings(X_train, X_test,
#                                                           max_feature = 1000 , embedding_type= 'bow')
    
#     # train the model 
#     print('... Training {} model'.format(classifier.__class__.__name__))
    
#     # train the model 
#     clf = OneVsRestClassifier(classifier)
#     clf.fit(Xv_train, y_train)

#     # compute the testing accuracy
#     prediction = clf.predict(Xv_test)
    
#     score = (accuracy_score(y_test, prediction))
#     cv_score= cross_val_score(classifier,corpus,df[labels],cv=5)
#     roc_lr = roc_auc_score(y1_test,prediction_lr)
    
#     ## Save model
#     pkl_file = os.path.join(dir_path,'model', classifier.__class__.__name__)
#     file = open(pkl_file,"wb")
#     pickle.dump(clf,file)
#     file.close()
    
#     #### Prediction on comment 
        
#     input_str = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]
#     input_str = clean(input_str[0])
#     input_str = process_txt(input_str, stemm= True)
#     input_str = vectorizer.transform([input_str])
    
#     print("KFold score {}".format(cv_score))
#     print(print_score(prediction, y_test, classifier))
#     print(print("check model accuracy on input_string {}".format(clf.predict(input_str))))
#     print('Validation accuracy is {}'.format(score))
#     print("------")
    


# In[22]:


# ### Linear regression for kfold CV
# cv_accuracy_lr = []
# accuracy_lr = []
# for label in labels:
#     print('... Processing {}'.format(label))
#     # train the model 
#     clf = OneVsRestClassifier(logreg)
#     clf.fit(Xv_train, y_train[label])
#     # compute the testing accuracy
#     prediction = logreg.predict(Xv_test)
#     score = (accuracy_score(y_test[label], prediction))
#     cv_score= cross_val_score(logreg,X,df[labels],cv=10)
#     accuracy_lr.append(score)
    
#     print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
#     print('\n cv_score = {}'.format(cv_score))
# print("\n Accuracy_lr: {}".format(accuracy_lr))


# In[ ]:


# ### Naive bayes
# accuracy_nb = []
# for label in labels:
#     print('\n... Processing {}'.format(label))
#     # train the model 
#     nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
#     nbayes.fit(Xv_train, y_train[label])
#     # compute the testing accuracy
#     prediction = nbayes.predict(Xv_test)
#     score = (accuracy_score(y_test[label], prediction))
#     cv_score= cross_val_score(nbayes,X,df[labels],cv=10)
#     accuracy_nb.append(score)
#     print('Validation accuracy is {}'.format(score))
#     print('cv_score = {}'.format(cv_score))
# print("\n Accuracy_nb: {}".format(accuracy_nb))


# In[61]:


#Train-test split
print("... Performing train test split")
X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],
                                                                    test_size=0.25,random_state=42)


# In[62]:


## Features extraction with word embedding
print("... Extracting features")
Xv_train, Xv_test, vectorizer = get_embeddings(X_train, X_test,
                                                          max_feature = 1000 , embedding_type= 'bow')

