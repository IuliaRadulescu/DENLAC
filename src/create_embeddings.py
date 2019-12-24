# coding: utf-8

__author__      = "Ciprian-Octavian Truică"
__copyright__   = "Copyright 2015, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "ciprian.truica@cs.pub.ro"
__status__      = "Development"

from tokenization import Tokenization
from documentembeddings import DocumentEmbeddings
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys
import os
import numpy as np
import pymongo
# This uses Document Embedding -> the mean vector of the word vectors in the document
# The word vectors can be normalize using TF-IDF
# the output is a file (ConfEmb.csv) where the first n-1 columns represent the embedding and the last column is the class
if __name__ =="__main__":
	# values for training the word2vec model
    sg = 0 # 0 - CBOW Model, 1 - Skipped-Gram Model
    workers = 30 # number of workes to train the model
    epochs = 100 # number of epochs to train the model
    normalize_tfidf = True # True normalize word2vec by TF-IDF, False otherwise
    
    dbname = 'ConferenceDB' # database name
    limit = 0 # limit = 0 take all documents
    projection = {"rawText":1, "tags": 1} # fields to be selected

    client = pymongo.MongoClient()
    
    db = client[dbname]
    doc_cursor = db.documents.find(projection=projection, limit=limit, no_cursor_timeout=True)
    
    # pass through the cursor
    documents = []
    for elem in doc_cursor:
    	doc = {"text": elem["rawText"], "class": elem["tags"][0]}
    	documents.append(doc)

    # Create a pandas DataFrame to store the text and class
    df = pd.DataFrame(documents)    

    # if weights are needed
    weights = df.groupby(df.columns[-1]).count()[df.columns[0]].to_dict()

    # text tokenization
    corpus = []
    tkn = Tokenization()
    for document in df["text"].tolist():       
        corpus.append(tkn.createCorpus(document))


    labels = df["class"].tolist()

    # save the encoder to get the id to class conversion
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels) 

    de = DocumentEmbeddings(corpus, normalize_tfidf=normalize_tfidf)     

    documents_vector = de.doc2VecEmbedding(sg=sg, workers=workers, epochs=epochs)
    X = np.array(documents_vector)
    res = np.c_[X, y]
    results = pd.DataFrame(res)
    results.to_csv("ConfEmb.csv",  header=False, index=False)