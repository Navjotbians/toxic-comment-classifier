
# coding: utf-8

# ### Model training

# #### Import all the required packages

# In[41]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import sys  
sys.path.append('F:/AI/Toxic-comment-classifier/src')
from word_embeddings import w_embeddings


# #### load processed dataset

# In[42]:


df = pd.read_csv('../data/processed/processed_stem_data.csv')


# In[43]:


df.head()


# In[44]:


### fill NA for any missing data 
df['comment_text'].fillna("missing", inplace=True)


# In[45]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']


# ### Split the date into train test datasets

# In[46]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],
                                                                    test_size=0.25,random_state=42)


# In[47]:


X_train.shape, X_test.shape


# In[48]:


# Stats of X_train labels
counts = []
for i in labels:
    counts.append((i, y_train[i].sum()))
df_stats = pd.DataFrame(counts, columns=['Labels', 'number_of_comments'])
df_stats


# In[49]:


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

# In[50]:


### pass "bow" to use Bag-of-Words instead of "tfidf"
Xv_train, Xv_test = w_embeddings(X_train, X_test, "bow") 


# In[51]:


Xv_train


# Now word embeddings are ready for `X_train`  and `X_test` data. These embeddings are in numpy array format, if we have a look at the the embeddings we will know this is high dimensional sparse data.

# ### Training

# Which model to use?
# <br>
# Which is the fastest model for hight dimensional sparse data?  - **Logistic regression**
# We will use Logistic regression for this dataset to start with and the solver we are using is 'sag' as it is faster for large datsets.

# #### With simple Train-Test split

# In[ ]:


### Linear Regression
accuracy_lr = [] ### list of accuracies of all the labels predicted by Linear regression
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))
    logreg.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = logreg.predict(Xv_test)
    score = (accuracy_score(y_test[label], prediction))
    accuracy_lr.append(score)
    print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
print("\n accuracy_lr: {}".format(accuracy_lr))


# <br>**Naive Bayes** is quite populer with text data problems. It learns the parameters by looking at each feature individually and collect simple per-class stats from each feature.
# We are going to use MultinomialNB because it assumes count data, that means, each feature represents an integer count of some-thing, in our problem- how often a word appears in a sentence.

# In[74]:


### Naive bayes
accuracy_nb = []  ## list of accuracies of all the labels predicted by Naive Bayes
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
    nbayes.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = nbayes.predict(Xv_test)
    score = (accuracy_score(y_test[label], prediction))
    accuracy_nb.append(score)
    print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
print("\n Accuracy_nb: {}".format(accuracy_nb))


# In[54]:


accuracy_lr_tfidf = [0.9260271225528288, 0.9898478429799714, 0.9656832025668663, 0.9973679592911037, 0.958213220364475, 0.9910510615897525]
accuracy_nb_tfidf = [0.9245983004537137, 0.989822775925601, 0.9633519665104153, 0.9973679592911037, 0.9569348005915825, 0.9910510615897525]     


# In[59]:


accuracy_lr_bow = [0.9107612864412303, 0.9899982453061941, 0.9515704509563081, 0.9973679592911037, 0.950993908705788, 0.9910259945353821]
accuracy_nb_bow = [0.9192590178728097, 0.987040332890482, 0.965608001403755, 0.9946857844734666, 0.9576868122226957, 0.9866893941292959]    


# In[60]:


## Linear regression accuracies compaired
accuracy_lr_tfidf == accuracy_lr_bow


# In[61]:


## Naive Bayes accuracies compaired
accuracy_nb_tfidf == accuracy_nb_bow


# **`Bag-of-Words` and `TF-IDF`**
# <br>Accuracies for `Bag-of-Words` and `TF-IDF` are not same but their difference is also not very significant for both Linear regression and Naive Bayes 
# <br>Accuracy remained same - `Identity hate`, `threat`
# <br>Accuracy remained almost same - `Severe_toxic`
# <br>Accuracy improved little bit  with the use of TF-IDF but not very significant change for `Toxic`, `Obscene`, `insult`.
# One possible reason for not seeing the expected significant improvement with the use of `TF-IDF` could be - the human raters didn't care for the semantics of the sentances and rated the comment based on the presence of toxic words.
# In this case, we will choose `Bag-of-Words` because the performance is almost same as `TF-IDF` but less chance of overfitting.
# 

# **Naive Bayes or Logistic regression !!**
# <br>Naive Bayes is tend to even faster in training but it provide generalized performance that is slightly worse than that of logistic regression model.
# Just yet we can't decide, we need to try different model evaluation techniqes first

# ### K FOLD CROSS VALIDATION

# Here, we need to convert complete dataset comments to word embeddings before doing cross-validation and word_embedding funtion works if dataset is already divided into Train-Test.  but for now we are converting dataset comments in word embeddings without using the function. later we will make a better way to do this.

# In[86]:


bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
X = bw_vectorizer.fit_transform(corpus).toarray()


# In[65]:


### Linear regression 
cv_accuracy_lr = []
accuracy_lr = []
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))
    logreg.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = logreg.predict(Xv_test)
    score = (accuracy_score(y_test[label], prediction))
    cv_score= cross_val_score(logreg,X,df[labels],cv=10)
    accuracy_lr.append(score)
    
    print('Validation accuracy is {}'.format(accuracy_score(y_test[label], prediction)))
    print('\n cv_score = {}'.format(cv_score))
print("\n Accuracy_lr: {}".format(accuracy_lr))


# ### Linear regeression accuracy summery
# <br>... Processing toxic
# <br>Validation accuracy is 0.9107612864412303
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]
#  
# <br>... Processing severe_toxic
# <br>Validation accuracy is 0.9900233123605645
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]
#  
# <br>... Processing obscene
# <br>Validation accuracy is 0.9515955180106785
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]
# 
# <br>... Processing threat
# <br>Validation accuracy is 0.9973679592911037
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]
#  
# <br>... Processing insult
# <br>Validation accuracy is 0.9509688416514176
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]
#  
# <br>... Processing identity_hate
# <br>Validation accuracy is 0.9910259945353821
# <br>cv_score = [0.90092743, 0.89509306, 0.9011719,  0.90148524, 0.89816382, 0.90273861,
#  0.90029454, 0.90004387, 0.90161058, 0.89822648]

# In[88]:


### Naive bayes
accuracy_nb = []
for label in labels:
    print('\n... Processing {}'.format(label))
    # train the model 
    nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())
    nbayes.fit(Xv_train, y_train[label])
    # compute the testing accuracy
    prediction = nbayes.predict(Xv_test)
    score = (accuracy_score(y_test[label], prediction))
    cv_score= cross_val_score(nbayes,X,df[labels],cv=10)
    accuracy_nb.append(score)
    print('Validation accuracy is {}'.format(score))
    print('cv_score = {}'.format(cv_score))
print("\n Accuracy_nb: {}".format(accuracy_nb))


# ### Naive Bayes Accuracy Summary
# <br>
# ... Processing toxic
# <br>Validation accuracy is 0.9192590178728097
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
# <br>... Processing severe_toxic
# <br>Validation accuracy is 0.987040332890482
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
# <br>... Processing obscene
# <br>Validation accuracy is 0.965608001403755
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
# <br>... Processing threat
# <br>Validation accuracy is 0.9946857844734666
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
# <br>... Processing insult
# <br>Validation accuracy is 0.9576868122226957
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
# <br>... Processing identity_hate
# <br>Validation accuracy is 0.9866893941292959
# <br>cv_score = [0.89478631, 0.89108228, 0.89315034, 0.89503039, 0.89227298, 0.8975998,
#  0.89578242, 0.89496773, 0.8965971,  0.8916463 ]
# 
#  <br>Accuracy_nb: [0.9192590178728097, 0.987040332890482, 0.965608001403755, 0.9946857844734666, 0.9576868122226957, 0.9866893941292959]

# ### Stratified K Fold Cross Validation

# In[ ]:


from sklearn.model_selection import StratifiedKFold


# In[79]:


Xc =corpus


# In[ ]:


skf_lr_accuracy = []
for label in labels:
    print('... Processing {}'.format(label))
    # train the model 
    logreg = OneVsRestClassifier(LogisticRegression(solver='sag'))

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X,df[label])
    lr_label_score = []
    for train_index, test_index in skf.split(Xc,df[label]):
      Y = df[label]
      #print("Train:", train_index, "validation:", test_index)
      X1_train, x1_test = Xc.iloc[train_index], Xc.iloc[test_index]
      y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]

      ### Convert into word embeddings
      bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
      X1_train = bw_vectorizer.fit_transform(X1_train).toarray()
      x1_test = bw_vectorizer.fit_transform(x1_test).toarray()


      logreg.fit(X1_train, y1_train)
      prediction = logreg.predict(x1_test)
      score = (accuracy_score(y1_test,prediction))
      lr_label_score.append(score)
    print('Validation accuracy of 10 Kfold is {}'.format(lr_label_score))
    skf_lr_accuracy.append(lr_label_score)

print("\n skf_lr_accuracy: {}".format(skf_lr_accuracy))


# In[95]:


# ## Resultant accuracies of Linear regression model using Stratified K Fold Cross Validation

# skf_lr_accuracy: [[0.9048752976563479, 0.913141567963903,
#                    0.9244218838127467, 0.9134549100708154,
#                    0.9128908942783731, 0.9118881995362537,
#                    0.9115748574293413, 0.8985398257817886,
#                    0.9031146205427085, 0.8975997994610516], 
#                   [0.9899110164180975, 0.9899730525788055,
#                    0.9904117315284828, 0.9904743999498653,
#                    0.9902237262643354, 0.9897223788932756,
#                    0.9899730525788055, 0.9899730525788055,
#                    0.9899730525788055, 0.990035721000188], 
#                   [0.947487153778669, 0.9534373629128282,
#                    0.9636523155981701, 0.9537507050197406, 
#                    0.9533120260700633, 0.9524346681707088, 
#                    0.9528733471203861, 0.9437237575985461,
#                    0.9461678260324622, 0.9414050260073948],
#                   [0.9969921042737185, 0.9970545841950241, 
#                    0.9970545841950241, 0.9969919157736417,
#                    0.9969919157736417, 0.9969919157736417,
#                    0.9969919157736417, 0.9969919157736417, 
#                    0.9969919157736417, 0.9969919157736417], 
#                   [0.9510590299536282, 0.9517453155355017, 
#                    0.9538760418625055, 0.9540013787052705, 
#                    0.9525600050134737, 0.9519333207996491, 
#                    0.9513066365858244, 0.9505546155292348, 
#                    0.9508679576361472, 0.9503666102650874], 
#                   [0.9911016418097506, 0.9912264210064549,
#                    0.9912264210064549, 0.9911637525850724,
#                    0.9912264210064549, 0.9912264210064549,
#                    0.9911637525850724, 0.9911637525850724,
#                    0.9911637525850724, 0.9911637525850724]]


# In[ ]:


skf_nb_accuracy = []
for label in labels:
    print('\n... Processing {}'.format(label))
    # train the model 
    nbayes = OneVsRestClassifier(naive_bayes.MultinomialNB())

    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X,df[label])
    
    score_label = []
    for train_index, test_index in skf.split(Xc,df[label]):
      Y = df[label]
      #print("Train:", train_index, "validation:", test_index)
      X1_train, x1_test = Xc.iloc[train_index], Xc.iloc[test_index]
      y1_train, y1_test = Y.iloc[train_index], Y.iloc[test_index]

      ### Convert into word embeddings
      bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
      X1_train = bw_vectorizer.fit_transform(X1_train).toarray()
      x1_test = bw_vectorizer.fit_transform(x1_test).toarray()


      nbayes.fit(X1_train, y1_train)
      prediction = nbayes.predict(x1_test)
      score = accuracy_score(y1_test,prediction)
      score_label.append(score)
      
    print('Validation accuracy for 10 Kfolds {}'.format(score_label))
    #print(type(score))
    skf_nb_accuracy.append(score_label)
    print("\n skf_nb_accuracy: {}".format(skf_nb_accuracy))


# In[89]:


# ## Resultant accuracies of Naive Bayes model using Stratified K Fold Cross Validation
# skf_nb_accuracy: [[0.9036846722646948, 0.923105846963715, 0.9217898101146832,
#                    0.9155856363978191, 0.9162749890330263, 0.9214137995863885,
#                    0.9229805101209501, 0.8621294729585762, 0.8815566835871405,
#                    0.856990662405214], [0.9898483519237999, 0.98652628940277, 
#                                         0.9868396315096822, 0.9872156420379771,
#                                         0.9860249420317102, 0.9879049946731842,
#                                         0.98652628940277, 0.9854609262392681, 
#                                         0.9884063420442439, 0.9845208999185311], 
#                   [0.9476751472615615, 0.9664097261389986, 0.9652190261327317, 
#                    0.9643416682333772, 0.9653443629754966, 0.9652816945541142,
#                    0.9652190261327317, 0.913956257441875, 0.9361408786112678, 
#                    0.914896283762612], [0.9956761498934703, 0.9944225104969606,
#                                         0.9940464999686658, 0.9952998683963151, 
#                                         0.9937958262831359, 0.9952371999749327,
#                                         0.9950491947107852, 0.9949865262894028,
#                                         0.9952371999749327, 0.9964278999811995],
#                   [0.9498684045619752, 0.9585135050448079, 0.9587641787303378, 
#                    0.956508115560569, 0.9570094629316288, 0.9585135050448079,
#                    0.9590775208372501, 0.9292473522591966, 0.9454158049758726,
#                    0.9251112364479539], [0.9896603584409074, 0.9867142946669173,
#                                          0.9870903051952121, 0.986400952560005, 
#                                          0.985586263082033, 0.9864636209813875, 
#                                          0.9874036473021245, 0.9851475841323557,
#                                          0.9895970420505108, 0.9870276367738297]]


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


# In[98]:


# Output of above cell
# Best accuracy : {'toxic': 0.9107612864412303, 'severe_toxic': 0.9900233123605645, 'obscene': 0.9515955180106785, 'threat': 0.9973679592911037, 'insult': 0.9509688416514176, 'identity_hate': 0.9910259945353821}
# Best parameter : {'toxic': 0.1, 'severe_toxic': 0.001, 'obscene': 0.1, 'threat': 0.001, 'insult': 0.1, 'identity_hate': 0.001}

