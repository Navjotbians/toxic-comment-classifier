import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import os
from clean_comments import  clean


### Frequently used words in the obscene comments
def process_txt(d, stemm = False,lemm = True):
    #all_word = []
        ### Clean input data
    processed_text = clean(d)

        ### Tokenization
    processed_text = word_tokenize(processed_text)

     ### remove stop words
    processed_text = [word for word in processed_text if word not in stopwords.words('english')]
    #all_word.append(processed_text)

    ### Stemming
    if stemm == True:
      ps = nltk.stem.porter.PorterStemmer()
      processed_text = [ps.stem(word) for word in processed_text]

    ### Lemmatization
    if lemm == True:
      lem = nltk.stem.wordnet.WordNetLemmatizer()
      processed_text = [lem.lemmatize(word) for word in processed_text]

    text = " ".join(processed_text)
    
    return text




if __name__ == "__main__":
	dir_path = os.path.dirname(os.getcwd())
	train_path = os.path.join(dir_path, 'data', 'raw', 'train.csv')

	print("... [INFO]  Loading the train dataset")

	df = pd.read_csv(train_path)

	    ### processing each comment and appending it to the dataset
	print("... [INFO]  Cleaning and processing each comment and appending it to the dataset ")
	df['clean_comment'] = df['comment_text'].apply(lambda x:process_txt(x))

	    ### dropping the original unclean comment coloum from dataset

	df = df.drop('comment_text', axis = 1)

	    ### Renaming the clean comment colum to comment_text for ease
	df = df.rename({'clean_comment': 'comment_text'}, axis=1)

	# print("dataset after dropping old comment {}".format(df.comment_text))

	 ### Save processed data to 
	df.to_csv(os.path.join(dir_path, 'data', 'processed','processed_data.csv'))
	print("... [INFO]  Processed data is saved in {} location ".format(os.path.join(dir_path, 'data', 'processed')))



