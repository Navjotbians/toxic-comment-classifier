
# coding: utf-8

# # Exploratory data analysis (EDA)

# Exploration of data is the very first step in approaching any machine learnig problem with data in hand.
# <br>One of the purpose of EDA is to understand the mindset of the person who has created the dataset that will help us to clean the data and tackle the missing values in the senseble way.
# <br>Another purpose of doing EDA: Here we are using past events data to make predictions, so it is worth exploring these past events to understand the could be causes for the outputs associated with them, this understanding will help us to find the important features to feed into our model.

# ### Preparation

# For preparation lets import the required libraries and the data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import re
# import string
# import nltk


# In[2]:


df = pd.read_csv('../data/raw/train.csv')


# In[3]:


print("Number of rows in data =",df.shape[0])
print("Number of columns in data =",df.shape[1])
print("\n")
print("Sample data:")
df.head()


# In[13]:


## check for Null values
df.isnull().sum()


# <br>No null values in the dataset

# <br>In below cell, decent comments are those which are clean, doesn't include any level of toxicity and not-decent comments are those which has "toxic", "severe_toxic", "obscene", "threat","insult" and "identity_hate" comments

# In[14]:


non_toxic = len(df[(df['toxic']==0) & (df['severe_toxic']==0) & (df['obscene']==0) & (df['threat']== 0) & (df['insult']==0) & (df['identity_hate']==0)])
toxic = len(df)-non_toxic
print("Number of decent comments : {}".format(non_toxic))
print("Number of not-decent comments : {} \n".format(toxic))
print('Percentage of decent comments: {} %'.format(non_toxic / len(df)*100))
print('Percentage of not-decent comments: {} %'.format(toxic / len(df)*100))


# <br> It is clear that data is highly imbalanced, we can see 143346 comments are decent or labeled under zero class, whereas only 16225 comments are not-decent or labeled as class 1. With this much skewness in dataset, the model will give default accuracy of 90% in  classifying a comment as a decent comment without learning anything.
# <br>That means the purpose of classifying not-decent comments will not be served with the presence of skewness in this dataset. There are ways to handle this problem such as under sampling or oversampling.
# <br>Another thing to notice here, due to this imbalanced ratio of the classes, accuracy makes it hard to evaluate the model performance, so we will explore alternative matrics that provide better guidance in evaluating and selecting model such as F1 score, AUC

# ### Number of comments in each category

# Total comments are 159571, out of it only 16225 comments are not-decent. Furthermore, these not-decent comments are divided in six categories. These categories are : "toxic", "severe_toxic", "obscene", "threat","insult" and "identity_hate".
# <br> <br> Lets calculate the number of comments belongs to each category

# In[17]:


##df.iloc[:,2:].sum()
df_targets = df.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_targets.columns.values)
for i in categories:
    counts.append((i, df_targets[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats


# In[18]:


levels = list(df_targets.columns.values)
sns.set(font_scale = 1)
ax= sns.barplot(levels, df_targets.iloc[:,:].sum().values)
plt.title("Number of comments in each toxic category", fontsize=14)
plt.ylabel('Number of comments', fontsize=12)
plt.xlabel('Toxicity Type ', fontsize=12)

rects = ax.patches
labels = df_targets.iloc[:,:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# This bar graph shows the presence of skewness among these six categories, we have significantly more number of comments in toxic category as compared to "severe_toxic", "identity_hate" and "threat" category. This could lead model to have more cofidence in predicting toxic, obscene and insult class than predicting severe_toxic, threat and insult class

# ### Number of comments have multi-labels

# Note that here comments could have more than one labels assigned to them, so let's see count of comments that have multiple labels 

# In[24]:


rowSums = df_targets.iloc[:,:].sum(axis=1)
x= rowSums.value_counts().iloc[1:]
ax = sns.barplot(x.index, x.values)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# In[23]:


print("Percentage of comments with multi-labels : \n \n{} ".format((x/toxic)*100))


# It can be seen that in this dataset, most of the comments have 1 to 3 labels attached to them and have significantly less number of comments which are labelled with 5 or 6 categories. 
# <br> These subtleties needs to be taken in account when splitting the train-test data. We need to make sure that our test data is a complete representation of the train data, for that we can try various cross validation techniques. Here stratified K-fold cross validation could be used because we have skewed dataset.
# 
# <br>We know comments have more than one label attached to it, that’s make it a multi-label classification problem. To tackle this, OneVsRest stratergy could be used.

# ### Check for similarities between the categories

# #### Comparing toxic and severe_toxic categories

# In[31]:


print('Total number of comments labeled as toxic are: {}\n Total number of severe toxic comments are: {}'.format(df['toxic'].sum(), df['severe_toxic'].sum())) 


# In[32]:


# Dataframe that contains all the comments that are labeled as a severe toxic comments
df1 = df[df['severe_toxic']==1]
df1.head()


# In[33]:


# Can be seen all the comments which are severe_toxic are labeled as toxic comments 
(df1.severe_toxic == df1.toxic).sum()


# In[34]:


pd.crosstab(df.toxic,df.severe_toxic,margins=True).style.background_gradient(cmap='Set3')


# <br>All the severe_toxic comments are by default gets toxic label

# #### Comparing toxic with obscene category

# In[35]:


pd.crosstab(df.toxic,df.obscene,margins=True).style.background_gradient(cmap='Set3')


# <br>It can be seen that 8449 comments are labeled as an obscene comments and it’s interesting to notice that out of 8449 obscene comments, 523 comments are not toxic. From the given dataset, I am assuming that the human labeler, labeled the comment as an obscene comment when the comment gives in-general negative vibe but doesn't contain vulgar words
# <br>
# <br>
# If this is the case then - comments which are toxic and obscene as well!!
# <br> what is the explanation for that?
# 

# In[37]:


# pd.crosstab(df.obscene,df.toxic,margins=True).style.background_gradient(cmap='Set3')


# In[38]:


# df_obs = df[df['obscene']== 1]


# In[39]:


# df2 = df[(df['toxic']== 0) & (df['obscene']== 1)]


# In[40]:


# df.comment_text[2897]


# <br>
# Its noted that here comments which are marked obscene but not toxic are those - where commenter is not targeting a particular person but they are using foul words out of their bad habbit.
# 

# #### Comparing obscene with severe_toxic category

# In[43]:


pd.crosstab(df.obscene,df.severe_toxic,margins=True).style.background_gradient(cmap='Set3')


# #### Comparing toxic with threat category

# In[21]:


pd.crosstab(df.toxic,df.threat,margins=True).style.background_gradient(cmap='Set3')


# <br> Here total 478 comments are labeled as a threat, out of which 449 are toxic and 29 are non-toxic.
# <br> These 29 comments have these common words "kill", "die", "warning" but no vulgar or insulting words are used. This could be the reason that these 29 comments qualify for threat label but not for toxic.
# 

# #### Comparing toxic with insult category

# In[22]:


pd.crosstab(df.toxic,df.insult,margins=True).style.background_gradient(cmap='Set3')


# #### Comparing toxic with identity_hate category

# In[23]:


pd.crosstab(df.toxic,df.identity_hate,margins=True).style.background_gradient(cmap='Set3')


# <br> Here out of 1405 identity hate comments, 103 comments are not toxic. 
# <br> After observing this category of comments, I could assume that - Identity hate comments targets race, color, religion, community etc. So for a comment to be qualified under this category doesn't have to use vulgar or foul words.
# 

# <br>Toxic - 
# <br>Severe toxic - those comments which have highly vulger
# <br>Obscene - comments with a negativity but no use of curse words or I can say soft toxic
# <br>Threat - 
# <br>Insult - 
# <br>Identity_hate - 
#  
