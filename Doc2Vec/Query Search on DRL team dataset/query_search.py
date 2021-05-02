#  Increase min_count (maybe 500) on line 92 to set minumum frequency count to be included in model
#  We haven't made any changes to the original docs, changing doc_list and tagged_data for preprocessing which is list of words
#  Put query in a loop for getting results for multiple query. line 112- Code for that commented out at the bottom

from sklearn.svm import LinearSVC
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import numpy as np
import sys
from pprint import pprint
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle
from random import shuffle
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
import sys
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
from collections import namedtuple
import nltk
import sys
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

#open the documents of dataset into list docs
with open('/home/mili/dev/data/doc2vec/docs.data','rb') as handle:
    docs=pickle.load(handle)

#open the query in variable query
with open('/home/mili/dev/data/doc2vec/query.data','rb') as handle:
    query=pickle.load(handle)

print(query)

print(len(docs))
#print(docs[9])

#preprocessing on data
def preprocessing(text,stem=False, stop=False, sent=False):
    # Remove punctuations
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    tokens = word_tokenize(text)
    if stop:
        stop = stopwords.words('english')
        tokens =[word for word in tokens if word not in stop]
        tokens = [word.lower() for word in tokens]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    if sent:
        tokens = ' '.join(tokens)
    return tokens

#Preprocessing for Doc2Vec 
def clean_docs(articles):
    clean = []
    for article in articles:
        clean.append(preprocessing(article,stop=True,sent=False, stem=False))
        #print clean
        #sys.exit(1)
    return clean
    
#to convert to a form that doc2vec model can accept using tagged documents
def convert_newsgroup(docs):
    #global doc_count500
    tagged_documents = []
    
    for i,v in enumerate(docs):
        label = '%s'%(i)
        tagged_documents.append(TaggedDocument(v, [label]))
    return tagged_documents

#print(docs[7])
docs_list= clean_docs(docs)
tagged_data = convert_newsgroup(docs_list)

#if you donot want this preprocessing, use the below mentioned line and remove the above one
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
    
dm_model =  Doc2Vec(dm=1, dm_mean=1, sample=1e-5, vector_size=300, window=10, negative=5, hs=0, min_count=10, workers=cores)

#dm_model.load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
dm_model.build_vocab(tagged_data)  # PV-DM/concat requires one special NULL word so it serves as template

#change passes for different number of iterations
alpha, min_alpha, passes = (0.025, 0.001, 100)
alpha_delta = (alpha - min_alpha) / passes
#from sklearn.grid_search import GridSearchCV

##### TRAINING
for epoch in range(passes):
    shuffle(tagged_data)
    #printing alpha here
    print(alpha)
    dm_model.alpha, dm_model.min_alpha = alpha, alpha
    dm_model.train(tagged_data,total_examples=dm_model.corpus_count,epochs=dm_model.epochs)       
    alpha -= alpha_delta


#########################################################
# put query in a loop to get result for list of queries
""" for query in queries:
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data1)
    similar_doc = dm_model.dv.most_similar([v1],topn=100)
    print(similar_doc)
    #you can index it as  similar_doc[idx]
 """

test_data = word_tokenize(query.lower())
v1 = dm_model.infer_vector(test_data)

print("V1_infer", v1)
similar_doc = dm_model.dv.most_similar([v1],topn=100)
print(similar_doc)




