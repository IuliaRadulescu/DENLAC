# coding: utf-8

__author__      = "Ciprian-Octavian Truică"
__copyright__   = "Copyright 2015, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "ciprian.truica@cs.pub.ro"
__status__      = "Development"

from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import TfidfModel
import numpy as np

class DocumentEmbeddings:

    def __init__(self, corpus, normalize_tfidf=False):
        self.corpus = corpus
        self.normalize_tfidf = normalize_tfidf

        self.documents = []
        self.sentences = []

        word_id = 0
        for document in self.corpus:
            doc = []
            for sentence in document:
                self.sentences.append(sentence)
                for word in sentence:
                    doc.append(word)
            self.documents.append(doc)
        if self.normalize_tfidf:
            self.tfidfVectorization()
        # print(self.documents)
        # print(self.sentences)


    def doc2VecEmbedding(self, window_size=10, no_components=128, epochs=100, workers=4, sg=0, learning_rate=0.05):
        self.doc2vec = []
        model = Word2Vec(self.sentences, size=no_components, window=window_size, min_count=1, workers=workers, sg=sg, alpha=learning_rate, iter=epochs)

        # for word in model.wv.vocab:
        #     print(word, model.wv[word])
        # print(model.wv['nefarious'])

        docidx = 0
        for document in self.documents:
            document_vector = np.array([0] * no_components)
            idx = 0
            for word in document:
                if self.normalize_tfidf:
                    document_vector = np.add(document_vector, self.tfidfs[docidx][word] * np.array(model.wv[word]))
                else:
                    document_vector = np.add(document_vector, np.array(model.wv[word]))
                idx += 1
            self.doc2vec.append(document_vector/idx)
            docidx += 1
        return self.doc2vec

    def tfidfVectorization(self, smartirs='atc'):
        dictionary = corpora.Dictionary(self.sentences)
        corpus = [ dictionary.doc2bow(document) for document in self.documents]
        tfidf = TfidfModel(corpus, smartirs=smartirs)
        self.tfidfs = []
        for documnet in tfidf[corpus]:
            doc = {}
            for id, freq in documnet:
                doc[dictionary[id]] = freq
            self.tfidfs.append(doc)
        return self.tfidfs


if __name__ == '__main__':
    corpus = [
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
        [
            ['Hello', 'this','tutorial', 'on', 'how','convert' ,'word',' integer','format'],
            ['this' ,'beautiful', 'day'],
            ['Jack','going' , 'office']
        ],
    ]

    we = DocumentEmbeddings(corpus)
    
    print(we.doc2VecEmbedding())
    print(we.doc2VecEmbedding(sg=1))
    
