from sklearn import feature_extraction

def w_embeddings(X_train, X_test, embedding_type = "tfidf"):
    if embedding_type == "bow":
        bw_vectorizer = feature_extraction.text.CountVectorizer(max_features= 100)
        X_train = bw_vectorizer.fit_transform(X_train).toarray()
        X_test = bw_vectorizer.fit_transform(X_test).toarray()
    if embedding_type == "tfidf":
        tf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=100)
        X_train = tf_vectorizer.fit_transform(X_train).toarray()
        X_test = tf_vectorizer.fit_transform(X_test).toarray()
    return X_train, X_test
    