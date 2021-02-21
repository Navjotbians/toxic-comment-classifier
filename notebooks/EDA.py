
# coding: utf-8

# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string


# In[2]:


df = pd.read_csv('../Input/train.csv')


# In[68]:


print("Number of rows in data =",df.shape[0])
print("Number of columns in data =",df.shape[1])
print("\n")
print("Sample data:")
df.head(2400)


# In[9]:


df.isnull().sum()


# <br>No null values in the dataset

# In[69]:


df.comment_text[2372]


# In[125]:


non_toxic = len(df[(df['toxic']==0) & (df['severe_toxic']==0) & (df['obscene']==0) & (df['threat']== 0) & (df['insult']==0) & (df['identity_hate']==0)])
toxic = len(df)-non_toxic
print("No of non toxic comments : {}".format(non_toxic))
print("No of toxic comments : {}".format(toxic))
print('Percentage of non toxic comments: {}'.format(non_toxic / len(df)*100))
print('Percentage of toxic comments: {}'.format(toxic / len(df)*100))


# <br> It is clear that data is highly imbalanced. With a base line model, it will give 90% of accuracy to classify a comment as a nontoxic comment. That means the purpose of this task will not be served with this imbalance dataset. There are ways to handle this problem such as under sampling or oversampling

# #### Number of comments in each category

# In[10]:


##df.iloc[:,2:].sum()
df_toxic = df.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats


# In[11]:


levels = list(df_toxic.columns.values)
sns.set(font_scale = 1)
ax= sns.barplot(levels, df_toxic.iloc[:,:].sum().values)
plt.title("Number of comments in each toxic category", fontsize=14)
plt.ylabel('Number of comments', fontsize=12)
plt.xlabel('Toxicity Type ', fontsize=12)

rects = ax.patches
labels = df_toxic.iloc[:,:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# #### Number of comments have multi-labels

# In[12]:


rowSums = df_toxic.iloc[:,:].sum(axis=1)
x= rowSums.value_counts().iloc[1:]
print("Percentage of comments with multi-labels : \n{}".format((x/toxic)*100))


# In[13]:


ax = sns.barplot(x.index, x.values)
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# <br>We see, one comment can have more than one label attached to it, that’s make it a multi-label classification problem. To tackle this, OneVsRest stratergy could be used

# #### All severe_toxic comments are also by default gets toxic label

# In[14]:


print('Total number of comments labeled as toxic are: {}\n Total number of severe toxic comments are: {}'.format(df['toxic'].sum(), df['severe_toxic'].sum())) 


# In[15]:


# Dataframe that contains all the comments that are labeled as a severe toxic comments
df1 = df[df['severe_toxic']==1]
df1.head()


# In[16]:


# Can be seen all the comments which are severe_toxic are labeled as toxic comments 
(df1.severe_toxic == df1.toxic).sum()


# In[17]:


pd.crosstab(df.toxic,df.severe_toxic,margins=True).style.background_gradient(cmap='Set3')


# <br>All the Severe toxic comments are also toxic

# In[18]:


pd.crosstab(df.toxic,df.obscene,margins=True).style.background_gradient(cmap='Set3')


# <br>It can be seen that 8449 comments are labeled as an obscene comments and it’s interesting to notice that out of 8449 obscene comments, 523 comments are not toxic. From the given dataset, I am assuming that the human labeler, labeled the comment as an obscene comment when the comment gives in-general negative vibe but doesn't contain vulgar words
# <br>
# <br>
# If this is the case then - comments which are toxic and obscene as well!!
# <br> what is the explanation for that?
# 

# In[119]:


pd.crosstab(df.obscene,df.toxic,margins=True).style.background_gradient(cmap='Set3')


# In[22]:


df_obs = df[df['obscene']== 1]


# In[31]:


df2 = df[(df['toxic']== 0) & (df['obscene']== 1)]


# In[120]:


df.comment_text[2897]


# <br>
# Its noted that here comments which are marked obscene but not toxic are those - where commenter is not targeting a particular person but their language is unacceptable.
# But it can't be said with confidence because the comment with ID 2897 contradicts it. In my understanding this has to be labeled as toxic as well.
# 

# In[116]:


pd.crosstab(df.obscene,df.severe_toxic,margins=True).style.background_gradient(cmap='Set3')


# In[70]:


pd.crosstab(df.toxic,df.threat,margins=True).style.background_gradient(cmap='Set3')


# <br> Here total 478 comments are labeled as a threat
# <br> Out of which 29 are labeled as non-toxic
# <br> It can be seen from the training dataset that these 29 comments have these common words "kill", "die", "warning" but no vulgar or insulting words are used. This could be the reason that these 29 comments qualify for threat label but not for toxic.
# 

# In[71]:


pd.crosstab(df.toxic,df.insult,margins=True).style.background_gradient(cmap='Set3')


# In[121]:


pd.crosstab(df.toxic,df.identity_hate,margins=True).style.background_gradient(cmap='Set3')


# <br> Here out of 1405 identity hate comments, 103 comments are not toxic. 
# <br> After observing this category of comments, I could assume that - Identity hate comments targets race, color, religion, community etc. So for a comment to be qualified under this quality doesn't have to use vulgar or foul words.
# 

# <br>Toxic - 
# <br>Severe toxic - those comments which have highly vulger
# <br>Obscene - comments with a negativity but no use of curse words or I can say soft toxic
# <br>Threat - 
# <br>Insult - 
# <br>Identity_hate - 
#  

# ## Data Pre-processing

# In[79]:





# In[80]:





# In[81]:





# In[83]:





# In[22]:





# In[131]:



    


# In[132]:




