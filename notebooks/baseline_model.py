
# coding: utf-8

# # Baseline model classification

# The purpose of this notebook is to make predictions for all six categories on the given dataset using some set of rules.
# <br>Let's assume that human labellers have labelled these comments based on the certain kind of words present in the comments. So it is worth exploring the comments to check the kind of words used under every category and how many times that word occurred in that category. So in this notebook, six datasets are created from the main dataset, to make the analysis easy for each category. After this, counting and storing the most frequently used words under each category is done. For each category, then we are checking the presence of `top n` words from the frequently used word dictionary, in the comments, to make the prediction.

# ### 1. Import libraries and load data

# For preparation lets import the required libraries and the data

# In[ ]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import operator
import pickle
import sys  
sys.path.append('F:/AI/Toxic-comment-classifier/src')
from clean_comments import clean


# In[ ]:


## Load dataset
df = pd.read_csv('../data/raw/train.csv')


# ### <br>2. Datasets for each category

# Dataset with toxic comments

# In[ ]:


#extract dataset with toxic label
df_toxic = df[df['toxic'] == 1]
#Reseting the index
df_toxic.set_index(['id'], inplace = True)
df_toxic.reset_index(level =['id'], inplace = True)


# Dataset of severe toxic comments

# In[ ]:


#extract dataset with Severe toxic label
df_severe_toxic = df[df['severe_toxic'] == 1]
#Reseting the index
df_severe_toxic.set_index(['id'], inplace = True)
df_severe_toxic.reset_index(level =['id'], inplace = True)


# Dataset with obscene comment 

# In[ ]:


#extract dataset with obscens label
df_obscene = df[df['obscene'] == 1]
#Reseting the index
df_obscene.set_index(['id'], inplace = True)
df_obscene.reset_index(level =['id'], inplace = True)
#df_obscene =df_obscene.drop('comment_text', axis=1)


# Dataset with comments labeled as "identity_hate" 

# In[ ]:


df_identity_hate = df[df['identity_hate'] == 1]
#Reseting the index
df_identity_hate.set_index(['id'], inplace = True)
df_identity_hate.reset_index(level =['id'], inplace = True)


# Dataset with all the threat comments

# In[ ]:


df_threat = df[df['threat'] == 1]
#Reseting the index
df_threat.set_index(['id'], inplace = True)
df_threat.reset_index(level =['id'], inplace = True)


# Dataset of comments with "Insult" label

# In[ ]:


df_insult = df[df['insult'] == 1]
#Reseting the index
df_insult.set_index(['id'], inplace = True)
df_insult.reset_index(level =['id'], inplace = True)


# Dataset with comments which have all six labels

# In[ ]:


df_6 = df[(df['toxic']==1) & (df['severe_toxic']==1) &
          (df['obscene']==1) & (df['threat']==1)& 
          (df['insult']==1)& (df['identity_hate']==1)]


# In[ ]:


df_6.set_index(['id'], inplace = True)
df_6.reset_index(level =['id'], inplace = True) 
# df6 = df_6.drop('comment_text', axis=1)


# ### <br> 3. Preperation of vocab

# In[1]:


### frequent_words function take dataset as an input and returns two arguments - 
### all_words and counts.
### all_words gives all the words occuring in the provided dataset
### counts gives dictionary with keys as a words those exists in the entire dataset and values
### as a count of existance of these words in the dataset.

def frequent_words(data):
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


# In[ ]:


## descend_order_dict funtion takes dataframe as an input and outputs sorted vocab dictionary
## with the values sorted in descending order (keys are words and values are word count)

def descend_order_dict(data):
    all_words, word_count = frequent_words(data)
    sorted_dict = dict( sorted(word_count.items(), key=operator.itemgetter(1),reverse=True))
    return sorted_dict


# In[ ]:


label_sequence = df.columns.drop("id")
label_sequence = sequence.drop("comment_text").tolist()
label_sequence


# #### <br>Getting the vocab used in each category in descending order its count 

# For **`toxic`** category

# In[ ]:


descend_order_toxic_dict = descend_order_dict(df_toxic)


# These are the words most frequently used in toxic comments

# <br>For **`severe_toxic`** category

# In[ ]:


descend_order_severe_toxic_dict =descend_order_dict(df_severe_toxic)


# These are the words most frequently used in severe toxic comments

# <br>For **`obscene`** category

# In[ ]:


descend_order_obscene_dict = descend_order_dict(df_obscene)


# These are the words most frequently used in obscene comments

# <br>For **`threat`** category

# In[ ]:


descend_order_threat_dict = descend_order_dict(df_threat)


# These are the words most frequently used in severe threat comments

# <br>For **`insult`** category

# In[ ]:


descend_order_insult_dict = descend_order_dict(df_insult)


# These are the words most frequently used in comments labeled as an insult

# <br>For **`identity_hate`** category

# In[ ]:


descend_order_id_hate_dict = descend_order_dict(df_identity_hate)


# These are the most frequently used words in the comments labeled as identity_hate

# <br> For comments when all categories are 1

# In[ ]:


descend_order_all_label_dict = descend_order_dict(df_6)


# These are the most frequently used words in the comments labeled as identity_hate

# #### <br> Picking up the top n words from the descend vocab dictionary

# In this code, top 3 words are considered to make the prediction.

# In[ ]:


# list(descend_order_all_label_dict.keys())[3]


# In[ ]:


## combining descend vocab dictionaries of all the categories in one dictionary 
## with categories as their keys

all_label_descend_vocab = {'toxic':descend_order_toxic_dict,
                       'severe_toxic':descend_order_severe_toxic_dict,
                       'obscene':descend_order_obscene_dict,
                       'threat':descend_order_threat_dict,
                       'insult':descend_order_insult_dict,
                       'id_hate':descend_order_id_hate_dict
                       }


# In[ ]:


## this function takes two arguments - all_label_freq_word and top n picks
## and outputs a dictionary with categories as keys and list of top 3 words as their values.

def dict_top_n_words(all_label_descend_vocab, n):
  count = dict()
  for label, words in all_label_descend_vocab.items():
      word_list = []
      for i in range (0,n):
        word_list.append(list(words.keys())[i])
      count[label] = word_list
  return count


# In[ ]:


### top 3 words from all the vocabs
dict_top_n_words(all_label_descend_vocab,3)


# ### <br>4. Performance check of baseline Model

# In[ ]:


## Check if the any word from the top 3 words from the six categories exist in the comments
def word_intersection(input_str, n,  all_words =all_label_descend_vocab):
    toxic_pred = []
    severe_toxic_pred = []
    obscene_pred = []
    threat_pred = []
    insult_pred = []
    id_hate_pred = []
    rule_based_pred = [toxic_pred, severe_toxic_pred, obscene_pred, threat_pred, 
                   insult_pred,id_hate_pred ]
    # top_n_words = dict_top_n_words[all_label_freq_word,n]
    
    for count,ele in enumerate(list(dict_top_n_words(all_label_descend_vocab,3).values())):

        for word in ele:
            if (word in input_str):
                rule_based_pred[count].append(word)
    #print(rule_based_pred)
    for i in range (0,len(rule_based_pred)):
        if len(rule_based_pred[i])== 0:
                rule_based_pred[i]= 0
        else:
                rule_based_pred[i]= 1
    return rule_based_pred


# In[ ]:


### Test
word_intersection(df['comment_text'][55], 3)


# <br>Uncomment the below cell to get the prediction on the dataset but it is already saved in `rule_base_pred.pkl` in list form to save time

# In[ ]:


## store the values of predictions by running the word_intersection function on 
## all the comments

# rule_base_pred = df['comment_text'].apply(lambda x: word_intersection(x,3))


# After running above cell, we get the prediction of the entire dataset each category in `rule_base_pred`, the orginal type of `rule_base_pred` is pandas.core.series.Series. This pandas series is converted into list and saved for future use. This `.pkl` fine can be loaded by running below cell.

# In[ ]:


### save rule_base_pred
# file_name = "rule_base_pred.pkl"

# open_file = open(file_name, "wb")
# pickle.dump(rule_base_pred, open_file)
# open_file.close()


# In[ ]:


### Open the saved rule_base_pred.pkl
open_file = open("rule_base_pred.pkl", "rb")
pred_rule = pickle.load(open_file)
open_file.close()


# In[ ]:


## true prediction 
y_true = df.drop(['id', 'comment_text'], axis=1)


# In[ ]:


## check the type 
type(y_true), type(rule_base_pred)


# <br>Uncomment pred_rule line in below cell to convert the type of predictions from panda series to list,if not using saved `rule_base_pred.pkl`

# In[ ]:


### Change the type to list
pred_true = y_true.values.tolist()
# pred_rule = rule_base_pred.values.tolist()  


# #### Compute accuracy of Baseline Model

# In[ ]:


## Accuracy check for decent and not-decent comments classification
count = 0
for i in range(0, len(df)):
    if pred_true[i] == pred_rule[i]:
        count = count+1
print("Overall accuracy of rule based classifier : {}".format((count/len(df))*100))


# Based on the rule implimented here, baseline classifier is classifying decent and not-decent comments with the **accuracy of 76.6%**.Now we have to see if AI based models giver better performance than this.

# In[ ]:


## Category wise accuracy check
for j in range(0, len(pred_true[0])):
    count = 0
    for i in range(0, len(df)):
        if pred_true[i][j] == pred_rule[i][j]:
            count = count+1
    print("Accuracy of rule based classifier in predicting {} comments : {}".format(label_sequence[j],(count/len(df))*100))


# Minimum accuracy for predicting `toxic `, `severe_toxic `, `obscene `, `threat `, `insult `, or  `identity_hate ` class of the Baseline model is more that 88%.
# <br>Accuracies for:
# <ol>
# <li>`toxic `: 89.4%</li>
# <li>`severe_toxic `: 88.2%</li>
# <li>`obscene `: 96.3%</li>
# <li>`threat `: 87.8%</li>
# <li>`insult `: 95.8%</li>
# <li>`identity_hate `: 98.3%</li>
# </ol>
# <br>In my opinion this model is doing quite good. As we know the dataset have more samples for toxic comments as compared to rest of the categories but this model still managed to predict with 89.4% of accuracy by just considering the top 3 words from its very large vocabulary. It may perform better if we consider more than 3 words from its vocab, because top 3 words not necessarily a true representaion of this category.
# <br>On the other hand, `obscene `, `insult `, and  `identity_hate ` have very good accuracy rates, seems like human labellers looked for these top 3 words to label comments under these categories.
# <br>For `threat ` category, the model should perform well as the number of sample for this category is just 478, that means it has smaller vocab comparative to other classes. but seems like human labellers looked at more than these top 3 words of its vocab. It could be checked by tweaking the number of top n words.
# 
