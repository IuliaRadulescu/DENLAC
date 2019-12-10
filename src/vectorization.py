# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2015, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@cs.pub.ro"
__status__ = "Production"


import pymongo
import math
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
from time import time
import pandas as pd

"""
    construct different csr (Compressed Sparse Row) matrices
"""
class Vectorization:
    def __init__(self, dbname='ConferenceDB', parallel = 0):
        client = pymongo.MongoClient()
        self.dbname = dbname
        self.db = client[self.dbname]
        self.doc_cursor = None
        self.voc_cursor = None
        self.words  = {}
        self.documents = {}
        self.id2word = {}
        self.word2id = {}
        self.id2docid = {}
        self.docid2id = {}
        self.num_rows = 0
        self.num_columns = 0
        self.avgDocLen = 0
        self.parallel = parallel
        self.class2id = {}
        self.doc2classid = {}

    """
        input:
            all: if True then use vocabulary_query, if False use the entire vocabulary
            limit: parameter used to limit the numeber of returned line, based on idf
            query: if all=True is used to select the documents for the query
    """
    def prepareData(self, all=True, query={}):
        if all:
            self.voc_cursor = self.db.vocabulary.find(no_cursor_timeout=True)
            self.doc_cursor = self.db.documents.find(no_cursor_timeout=True)
        else:
            self.voc_cursor = self.db.vocabulary_query.find(no_cursor_timeout=True)
            self.doc_cursor = self.db.documents.find(query, no_cursor_timeout=True)
        
        # parallel 
        if self.parallel == 1:
            print ("TO DO")
        elif self.parallel == 0: # single thread 
            print ('Single thread')
            self.avgDocLen = 0

            print ("Start vocabulary")
            idx = 0 
            for elem in self.voc_cursor:
                self.id2word[idx] = elem["word"]
                self.word2id[elem["word"]] = idx
                self.words[elem["word"]] = {
                    "IDF": elem["IDF"], 
                    "GTF": elem["GTF"], 
                    "SIDF": elem["SIDF"], 
                    "PIDF": elem["PIDF"], 
                    "SPIDF": elem["SPIDF"]
                }
                idx += 1
            print ("Finish vocabulary")

            self.num_columns = idx - 1

            print ("Start documents")
            idx = 0
            idx_class = 0
            for elem in self.doc_cursor:
                d = {}
                docLen = 0
                for w in elem["words"]:
                    d[self.word2id[w["word"]]] = {
                        "TF": round(w["tf"], 2),  # TF double normalized
                        "count": round(w["count"], 2), # TF raw frequency
                        "NTF": round(1 + math.log(w["count"]), 2), # TF normalized
                        "IDF": round(self.words[w["word"]]["IDF"], 2), # IDF
                        "GTF": round(self.words[w["word"]]["GTF"], 2), # GTF the number of documents where the term appears
                        "SIDF": round(self.words[w["word"]]["SIDF"], 2),  # IDF smooth
                        "PIDF": round(self.words[w["word"]]["PIDF"], 2),  # IDF probabilistic
                        "SPIDF": round(self.words[w["word"]]["SPIDF"], 2) # IDF probabilistic smooth
                    }
                    docLen += w["count"]
                if d:
                    self.id2docid[idx] = elem["_id"]
                    self.docid2id[elem["_id"]] = idx
                    if self.class2id.get(elem["tags"][0]) is None:
                        self.class2id[elem["tags"][0]] = idx_class
                        idx_class += 1
                    self.doc2classid[idx] = self.class2id[elem["tags"][0]]
                    self.documents[idx] = { "words": d, "docLen": docLen }
                    self.avgDocLen += docLen
                    idx += 1
            print("Finish documents")
            self.num_rows = idx - 1
            self.avgDocLen /= self.num_rows


    # normalize a csr matrix using l1, l2 or max norm on rows
    def normalize_CSR(self, csr, norm='l2'):
        normilized_csr = normalize(csr, norm=norm, axis=1, copy=True)
        return normilized_csr

    """
        constructs the binary vectorization (0, 1)
        output:
            the binary csr
    """
    def build_Binary(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                row.append(key)
                col.append(word)
                data.append(1)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the count vectorization
        output:
            the count csr matrix
    """

    def build_Count(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                count = self.documents[key]['words'][word]['count']
                row.append(key)
                col.append(word)
                data.append(count)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the TF vectorization
        output:
            the TF csr matrix
    """
    def build_TF(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                row.append(key)
                col.append(word)
                data.append(tf)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr


    """
        constructs the TF*IDF vectorization
        output:
            the TF*IDF csr matrix
    """
    def build_TF_IDF(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['IDF']
                tfidf = tf * idf
                row.append(key)
                col.append(word)
                data.append(tfidf)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)
        
        return csr

    """
        constructs the TF*SIDF vectorization
        output:
            the TF*SIDF csr matrix
    """
    def build_TF_SIDF(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SIDF']
                tfidf = round(tf * idf, 2)
                row.append(key)
                col.append(word)
                data.append(tfidf)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the TF*PIDF vectorization
        output:
            the TF*PIDF csr matrix
    """
    def build_TF_PIDF(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['PIDF']
                tfidf = round(tf * idf, 2)
                row.append(key)
                col.append(word)
                data.append(tfidf)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the TF*SPIDF vectorization
        output:
            the TF*PIDF csr matrix
    """
    def build_TF_SPIDF(self, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SPIDF']
                tfidf = round(tf * idf, 2)
                row.append(key)
                col.append(word)
                data.append(tfidf)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the Okapi BM25 (using TF*IDF) vectorization
        output:
            the Okapi BM25 csr matrix
    """
    def build_Okapi_TF_IDF(self, k1=1.6, b=0.75, filename=None):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            docLen = self.documents[key]['docLen']
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['IDF']
                okapi = (tf*idf*(k1+1))/(tf+k1*(1-b+b*docLen/self.avgDocLen))
                row.append(key)
                col.append(word)
                data.append(okapi)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    """
        constructs the Okapi BM25 (using TF*SPIDF) vectorization
        output:
            the Okapi BM25 csr matrix
    """
    def build_Okapi_TF_SPIDF(self, k1=1.6, b=0.75):
        data = []
        row = []
        col = []
        for key in sorted(self.documents):
            docLen = self.documents[key]['docLen']
            for word in sorted(self.documents[key]['words']):
                tf = self.documents[key]['words'][word]['TF']
                idf = self.documents[key]['words'][word]['SPIDF']
                okapi = (tf*idf*(k1+1))/(tf+k1*(1-b+b*docLen/self.avgDocLen))
                row.append(key)
                col.append(word)
                data.append(okapi)
            row.append(key)
            col.append(self.num_columns + 1)
            data.append(self.doc2classid[key])
        # add class
        data = np.array(data)
        row = np.array(row)
        col = np.array(col)
        csr = csr_matrix((data, (row, col)), dtype=np.float64)

        return csr

    def csr2df(self, csr):
        return pd.DataFrame(c.todense())

# this are just for tests
def printMatGensim(mat, id2word):
    maximum = 0
    minimum = 100000.0
    for elem in mat:
        for id_p in elem:
            print(id2word[id_p[0]], id_p[1])
            if maximum < id_p[1]:
                maximum = id_p[1]
            if minimum > id_p[1]:
                minimum = id_p[1]
    print(maximum)
    print(minimum)

def printMatSklearn(mat, id2word):
    maximum = 0
    minimum = 100000.0
    for elem in mat.toarray():
        idx_col = 0
        for c in elem:
            if c > 0:
                print(id2word[idx_col], c)
                if maximum < c:
                    maximum = c
                if minimum > c:
                    minimum = c
            idx_col += 1
    print(maximum)
    print(minimum)

# these are just tests
if __name__ == '__main__':
    start = time()
    vec = Vectorization(dbname='ConferenceDB', parallel=0)
    # query = {"_id": {"$in": ["1", "2", "3", "4", "5"]}}
    # mm.prepareData(all=False, query=query)
    
    vec.prepareData(all=True)

    # map the word id (becomes the column in the matrix/dataframe) to the word
    # id2word = vec.id2word

    # map the python document id to the class id
    # doc2classid = vec.doc2classid
    
    # map the class name to id
    # class2id = vec.class2id
    
    # map python document ids to document id in the database 
    id2docid = vec.id2docid
    c = vec.build_TF_IDF()
    
    df_tfidf = vec.csr2df(c) # get a dataframe
    print(df_tfidf)
    '''
        if you want to save it to csv
    '''
    # df_tfidf.to_csv("test.csv", header=False, index=False)

    '''
        if you want to notmalize de s
    '''
    # nc = vec.normalize_CSR(c, 'l2')
    # df_norm_tfidf = vec.csr2df(c) # get a dataframe
    # df_norm_tfidf.to_csv("test.csv", header=False, index=False)