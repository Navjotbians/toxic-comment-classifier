
# coding: utf-8

# # Model training

# ## Import all the required packages

# In[1]:


import os
dir_path = os.path.dirname(os.getcwd())


# In[2]:


from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sys  
sys.path.append(os.path.join(dir_path, "src"))
from word_embeddings import get_embeddings
from clean_comments import clean
from processing import process_txt
from train import train_model
from sklearn import feature_extraction, preprocessing, naive_bayes
import pickle


# ## Load Processed Dataset

# In[3]:


processed_data = os.path.join(dir_path, 'data', 'processed', 'processed_stem_data.csv')


# In[4]:


df = pd.read_csv(processed_data)


# In[5]:


df.head()


# In[6]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[7]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ## Convert Text Comments into Vectors using Bag-of-Words or TF-IDF 

# We know that machine learning models doesn't accept input in the text format. So we need to convert the text data into vector form, it is also called **Word Embeddings**. Word Embeddings can be broadly classified as:
# 1. Frequency based - Most popular techniques are **Bag-of-Words**, **TF-IDF**
# 2. Prediction based - Most popular techniques are **Word2vec** and **Glove**
# 
# 

# Here we will be using **Bag-of-Words** and **TF-IDF**<br>
# <br>**Bag-of-Words(BOW)** - To get the embeddings with BOW technique, firstly we will have to make a dictionary with keys as a words from the test data and count of each word occurance as a value, then sort the dictionary in descending order of its values.Then these words from the dictionary are the names of our independed features and we can also choose how many features we want to use by selecting the top n words from the dictionary. Now these features will have values 0 or 1 based on if the word exists in the sentence.
# <br>
# **Disadvantage** of BOW - Word Embeddings have either 0's and 1's as a values, no weights are given to the words according to their importance in the sentence. That means we can not get the sementics of the sentence.

# **TF-IDF** - It stands for Term Frequency - Inverse Document Frequency
# <br> To get embedding with TF-IDF, we calculate Term frequency and Inverse Document Frequency seperate and then multiply them together to get TF-IDF.
# Formulas to calculate TF-IDF: 
# <br>
# <br>**TF :**  $$\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}$$ 
# **IDF :**$$log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>
# **TF-IDF :** $TF * IDF$ $$\Bigg[\frac{Number\, of\, repetition\, of\, word\, in\, a\, sentence}{Number\, of\, words\, in\, a\, sentence}\Bigg]*log\Bigg[\frac{Total\, Number\, of\,sentences}{Number\, of\, sentences\, containing\, the \, word}\Bigg]$$ 
# <br>In **TF-IDF** also, we create dictionary of words with their count of occurance. **TF** assign more weightage to the word which repeat multiple times in the sentance where as **IDF** decreases the weightage to word as number of sentences containing the increases. Here, feature vectors not only contains 0's and 1' but does contain other values depending on the significance of the word in the sentence. This technique retains the sementics of the sentence to some extent so it should perform better than BOW.
# <br>Here **TF-IDG** can have zero value for the word which existed in every sentence and give more weightage to least occured words that means it could cause over-fitting problem but that is yet to discove. 

# We will try Bag-of-Words and TF-IDFto get our features for `X_train`  and `X_test` data. The resultant embeddings are in numpy array format, if we have a look at the the embeddings we will know it is high dimensional sparse data.

# ## Matrix used to evaluate the models

# Multi-label classification problems must be assessed using different performance measures than single-label classification problems.
# <br>
# <br>
# Jaccard similarity, or the Jaccard index, is the size of the intersection of the predicted labels and the true labels divided by the size of the union of the predicted and true labels. It ranges from 0 to 1, and 1 is the perfect score.Here we are taking mean so 100 is perfect score. This function can be imported from *Sklearn.metrics* but just to have better usderstanding we are defining the `j_score` function.
# <br>
# <br>
# We will also look at *F1-score* and *ROC score*

# ## Model Training

# ### Models we are using for text classification
# 1. Logistics Regression
# 2. Naive Bayes (NB)
# 

# Which model to use?
# <br>
# Which is the fastest model for high dimensional sparse data?  - **Logistic regression**
# <br>We will use Logistic regression for this dataset to start with and the solver we are using is 'sag' as it is faster for large datsets.
# 

# ### Logistic Regression
# 
# Logistic regression is similar to `Linear Regression` except it predicts whether something is **True** or **False**, instead of predicting a continuous value. Also instead of fitting a line to the data it fits an **"S"** shaped "logistic function" which is called `sigmoid function`.
# <br> In `Logistic Regression ` we don't need to do much to classify an instance. All we have to do is calculate the sigmoid of the vectorunder test multiplied by the weights optimized. If sigmoid gives a value greater than 0.5 the class is 1 and it is 0 otherwise.
# 

# ### Initiate models

# In[12]:


logreg = LogisticRegression(solver='sag')


# In[56]:


lr_clf, lr_vectorizer, lr_sumry = train_model(logreg, corpus, df[labels])


# ### Logistic Regression Results

# In[58]:


summary_lr = pd.DataFrame(lr_sumry, index = ['accuracy', 'jaccard score', 'F1_score', 'roc_score'], 
                          columns= [lr_clf.estimators_[0]])
summary_lr


# <br>
# ### Naive Bayes

# It is a classifier under supervised ML group based on probalistic logic. Probabilistic logic it uses is  `Bayes Theorem`  which gives probability of an event based on prior knowledge of condition that might be related to event
# <br>
# <br>
# **P(Class|Features) :**  $$\frac{P(Features|Class) * P(Class)}{P(Features)}$$ 
# It assumes that the probability of a one word doesn't depends on any other word in the document. We know this is unrealistic. That's why it is known as `naive Bayes`. Despite of its incorrect assumptions, `naive Bayes` is effective at classification. 
# <br>Moreover, assuming conditional independence among the features in the dataset, it reduces the need of large training data.
# #### Multinomial NB
# In our use case we will be using this `Multinomial NB` because it assumes count data which means each feature represents an integer count of some-thing, in our problem it is that how often a word appears in a sentence.

# ### Initiate models

# In[8]:


n_bayes = naive_bayes.MultinomialNB()


# In[43]:


nb_clf, nb_vectorizer, nb_sumry = train_model(n_bayes, corpus, df[labels] )


# ### Naive Bayes Results

# In[44]:


summary_nb = pd.DataFrame(nb_sumry, index = ['accuracy', 'jaccard score', 'F1_score', 'roc_score'],
                          columns= [nb_clf.estimators_[0]])
summary_nb


# <br>
# **Naive Bayes or Logistic regression !!**
# <br>Compairing the confusion matrixs and Jaccard score, Naive Bayes clearly out performed Linear regression and Naive Bayes even tends to get trained faster.
# Just yet we can't decide, we need to try different model evaluation techniqes first.

# ### K FOLD Cross Validation

# #### Logistic Regression - K Fold cross validation with Gridsearch

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


# Uncomment to reproduce results
clf_lr = OneVsRestClassifier(logreg)


# In[15]:


# Uncomment to reproduce results
gs_lr = GridSearchCV(clf_lr, param_grid ,scoring = 'f1_micro', cv=3)


# In[25]:


# Uncomment to run GridSerch with K-fold cross validation
gs_lr.fit(Xv_train, y_train)


# In[26]:


print("Accuracy score of the best estimator : {}".format(gs_lr.score(Xv_test, y_test )))


# In[28]:


print("Best estimator is : {}".format(gs_lr.best_estimator_))


# <br>
# As we can see that *Logistic Regression* performed well with **C = 0.1** on 3 Folds of cross validation. We will use this value of **C** with the various combination of *maximum_features* and *embedding types* in next part to check the performance of *Logistic Regression* during the best model selection process

# <br>
# #### Naive Bayes - K-fold cross validation

# In[68]:


### Naive bayes
nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
nbayes.fit(Xv_train, y_train)
cv_score= cross_val_score(nbayes,Xv_train,y_train,cv=3)
print('Naive Bayes Cross Validation score = {}'.format(cv_score))


# <br>
# We can see even with the K-fold cross validation *Naive Bayes* is giving consistant results for the accuracy

# ## Best Model Selection

# In[33]:


logreg = LogisticRegression(solver='sag', C= 0.1)
classifier = [logreg, n_bayes]
max_feature = [1200, 1500, 1700, 2000]
embedding= ['bow', 'tfidf']


# #### Training models with all different combinations of the parameters stated in above cell

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


# ### Performance Summary of all the Models Trained on different sets of Parameters

# In[74]:


summary_r = pd.DataFrame(summary_report, index = [combinations], columns= ['accuracy', 'jaccard score', 'F1_score', 'roc_score'],)
summary_r


# In[75]:


summary_all_combinations = summary_r.sort_values(by=['jaccard score'],ascending=False)
summary_all_combinations


# with **C =0.1** logistic regression is doing much worst that **C= 1**

# Its evident from above results that *Logistic Regression* perform well with continuous data (when features were computed with "TF-IDF") and *Naive Bayes* do good when data is in discrete form (When features were computer with "Bag-of Words"). Considering *Jaccard score*, *F1-score*, and *ROC_AUC score* **Naive Bayes** perform well, though when we train *Logistic Regression* with **C=1**, it does perform better than *Naive Bayes* only in terms of *Jaccard score*. I am sure with further fine tuning of the hyper parameters, we could get good *Logistic Regression* Model but its is quite expensive to get there in terms of computations and time. So we will go with **Naive Bayes** using *"Bag-of-Words"* embedding technique with *max_features* count as 2000

# ## Training the best model

# In[9]:


final_clf, final_vectorizer , model_summary = train_model(n_bayes, corpus, df[labels], max_feature=2000, embedding= "bow")


# ### Best Model Results

# In[46]:


summary_best_model = pd.DataFrame(model_summary, index = ['accuracy', 'jaccard score', 'F1_score', 'roc_score'],
                          columns= [final_clf.estimators_[0]])
summary_best_model


# ### Save Model

# In[48]:


## Save model
pkl_file = os.path.join(dir_path,'model', 'final_model.pkl')
file = open(pkl_file,"wb")
pickle.dump(final_clf,file)
file.close()
print("...Model saved in model directory")
    


# In[25]:


## Save vectorizer
print("...Saving vectorizer in model directory")
pkl_file = os.path.join(dir_path,'model', 'final_vectorizer.pkl')
file = open(pkl_file,"wb")
pickle.dump(final_vectorizer,file)
file.close()
print("...Saved vectorizer in model directory")


# ### Ignore below part

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

