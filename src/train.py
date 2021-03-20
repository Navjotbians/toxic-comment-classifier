import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pickle
from word_embeddings import get_embeddings
from clean_comments import clean
from processing import process_txt
import os
dir_path = os.path.dirname(os.getcwd())


def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
    return jaccard.mean()*100

def print_score(y_pred, y_test, clf):
    print("Clf: ",clf.__class__.__name__)
    print("Jaccard score: {}".format(j_score(pd.DataFrame(y_test), pd.DataFrame(y_pred))))
    print("F1 Score : {}".format(f1_score(y_test, y_pred,average='macro')))
    

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
    print("... Computing accuracy")
    prediction = clf.predict(Xv_test)

    ## Accuracy score
    score = (accuracy_score(y_test, prediction))
    type2_score = j_score(y_test, prediction)
    f1_s = f1_score(y_test, prediction,average='macro')
    roc_auc = roc_auc_score(y_test, prediction)
    confusion_matrix = multilabel_confusion_matrix(y_test, prediction)
    score_sumry = [score, type2_score, f1_s, roc_auc]
    
    
    # ## Save model
    # print("... Saving model in model directory")
    # pkl_file = os.path.join(dir_path,'model', classifier.__class__.__name__)
    # file = open(pkl_file,"wb")
    # pickle.dump(clf,file)
    # file.close()
    
    #### Testing purpose only #### 
    #### Prediction on comment ### 

    # input_str = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]
    # input_str = clean(input_str[0])
    # input_str = process_txt(input_str, stemm= True)
    # input_str = vectorizer.transform([input_str])
    

    print('\n')
    print("Model evaluation")
    print("------")
    print(print_score(prediction,y_test, classifier))
    print('Accuracy is {}'.format(score))
    print("ROC_AUC - {}".format(roc_auc))
    # print(print("check model accuracy on input_string {}".format(clf.predict(input_str))))
    print("------")
    print("Multilabel confusion matrix \n {}".format(confusion_matrix))
    
    return clf, vectorizer, score_sumry
    
