from src import pattern
from src import processing, word_embeddings
import pandas as pd
from sklearn import model_selection


input_str = "D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)"
clean_input = pattern.clean(input_str)
print("...Output of clean function... \n {} \n".format(clean_input))


out = processing.make_dict(clean_input)
print("...Output of a processing funtion... \n {} \n".format(out))

df = pd.read_csv('data/processed/processed_data.csv')
df = df.drop('comment_text', axis = 1)
df = df.rename({'clean_comment': 'comment_text'}, axis=1)
print("...Sneakpeak of dataset... \n {} \n".format(df.head()))

df['comment_text'].fillna("missing", inplace=True)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corpus = df['comment_text']

### Split data
X_train, X_test, y_train, y_test = model_selection.train_test_split(corpus,df[labels],test_size=0.25,random_state=42)

X_train, X_test = word_embeddings.embeddings(X_train, X_test, embedding_type = "tfidf")

print("...Output of word embedding function... \n {} ".format(X_train[3]))