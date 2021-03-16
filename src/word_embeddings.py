from sklearn import feature_extraction

def get_embeddings(X_train, X_test, max_feature = 100, embedding_type = "tfidf"):
    if embedding_type == "bow":
        vectorizer = feature_extraction.text.CountVectorizer(max_features= max_feature)
        vectorizer.fit_transform(X_train).toarray() 

        train_feat = vectorizer.transform(X_train).toarray()
        test_feat = vectorizer.transform(X_test).toarray()

        return train_feat, test_feat, vectorizer
                                                             
                                                             
    if embedding_type == "tfidf":
        vectorizer = feature_extraction.text.TfidfVectorizer(max_features=max_feature)
        vectorizer.fit_transform(X_train).toarray()

        train_feat = vectorizer.transform(X_train).toarray()
        test_feat = vectorizer.transform(X_test).toarray()

        return train_feat, test_feat, vectorizer