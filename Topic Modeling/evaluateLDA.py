import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from gensim.models import Word2Vec
from sklearn import metrics
from nltk.corpus import stopwords
import numpy as np
import sys
import os
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
import re 
from collections import defaultdict

#fetch data from 20NG
parent_path = os.getcwd()
child_path = "20news-bydate-train"
path = os.path.join(parent_path, child_path) 
os.chdir(path)
labels=[]
docs = []
for subdirectory in os.listdir(os.getcwd()):
    sub_path = os.path.join(path,subdirectory)
    #print("Sub directory : ",subdirectory)
    #print(sub_path)
    os.chdir(sub_path)
    for textfile in os.listdir(os.getcwd()):
        textfilepath = os.path.join(sub_path,textfile)
        with open(textfilepath, 'rb') as f:
            content = f.read()
            docs.append(content)
            labels.append(subdirectory)
    #print(docs[-1])
    os.chdir(path)

os.chdir(parent_path)


print(len(docs))
#docs = docs[:200]


#preprocessing on data
def preprocessing(text,stop=True, sent=False, stem=False):
    # Remove punctuations
    #print(text)
    exclude = set(string.punctuation)
    #exclude.append = set((string.digits)
    text = str(text)
    #Remove newline charecter from text
    text = text.replace(r'\n', '\n')
    text = text.replace(r'\t', '\t')
    #text = text.replace('Lines:', '\t')
    #Remove header : from till subject  Re
    subject_idx = text.find("Subject: Re")
    if(subject_idx!=-1):
        text = text[subject_idx+12:]
    

    #Remove header : from till subject (no re)
    subject_idx = text.find("Subject:")
    if(subject_idx!=-1):
        text = text[subject_idx+9:] 

    subject_idx = text.find("Lines:")
    if(subject_idx!=-1):
        text = text[subject_idx+10:]   

    subject_idx = text.find("writes:")
    if(subject_idx!=-1):
        text = text[subject_idx+8:]  
    #print(text)
    #remove punctuations
    text = ''.join(ch for ch in text if ch not in exclude) 
    #print(stopwords.words('english'))
    #print(text)
    tokens = word_tokenize(text)
    if stop:
        customstop = ["nntppostinghost","aaa","abc","wwii"]
        stop = stopwords.words('english')
        tokens =[word for word in tokens if word not in stop and word not in customstop]
        tokens = [word.lower() for word in tokens]
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    if sent:
        tokens = ' '.join(tokens)
    remove_list = ['.','0','1','2','3','4','5','6','7','8','9','@','#','*','!','$']
    toklist=[]
    for tok in tokens:
        f=1

        for ch in tok:
            if ch in remove_list:
                #print("here ",ch)
                f=0
                break
        #print(tok)
        if f==1 and len(tok)>2 and len(tok)<17:
            #print(tok)
            toklist.append(tok)
    tokens=toklist
    return tokens

#Preprocessing for Doc2Vec 
def clean_docs(articles):
    clean = []
    for article in articles:
        clean.append(preprocessing(article,stop=True,sent=False, stem=False))
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

###### REMOVE 70-5000 FREQ WORDS
def getfrequency(docs):
    for doc in docs:
        for token in doc:
            frequency[token] += 1
            #print("token ",token," freq ",frequency[token])
    for word in frequency.keys():
        if frequency[word]>70 and frequency[word]<5000:
            vocab.append(word)
    processed_corpus = [[token for token in doc if frequency[token] >70 and frequency[token]<5000] for doc in docs]
    
    #print(frequency)
    return processed_corpus

#print(docs[7])
vocab = []
docs_list= clean_docs(docs)
frequency = defaultdict(int)
processed_corpus = getfrequency(docs_list)
#print(processed_corpus)

from gensim.corpora import Dictionary
dictionary = Dictionary(processed_corpus)
corpus = [dictionary.doc2bow(doc) for doc in processed_corpus]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 20
chunksize = 300
#passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.


model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=150
)

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

import sklearn.metrics.pairwise as pw

#query_list = []
#query = input("Enter query")
#query_list.append(query)
#query_list= clean_docs(query_list)
#frequency = defaultdict(int)
#new_doc = getfrequency(query_list)
#print(new_doc[0])
#query_token = query.split()
#new_bow = dictionary.doc2bow(new_doc[0])
#new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in model[corpus]])
#print(doc_topic_dist.shape)
#print(new_doc_distribution.shape)
# print the top 8 contributing topics and their words
''' for i in new_doc_distribution.argsort()[-5:][::-1]:
    print(i, model.show_topic(topicid=i, topn=10), "\n") '''
def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    #p = query[None,:].T # take transpose
    print(query.shape)
    print(matrix.shape)
    p = query[None,:].T + np.zeros([20, 11314])
    q = matrix.T # transpose matrix
    m = 0.5*(p + q)
    return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

def get_most_similar_documents(query,matrix,k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query,matrix) # list of jensen shannon distances
    return sims.argsort()[:k] # the top k positional index of the smallest Jensen Shannon distances

def closest_docs_by_index(corpus_vectors, query_vectors, n_docs):
    docs = []
    query_vectors = query_vectors.reshape(1, -1)
    sim = pw.cosine_similarity(corpus_vectors, query_vectors)
    order = np.argsort(sim, axis=0)[::-1]
    for i in range(len(query_vectors)):
        docs.append(order[:, i][0:n_docs])
    #print("docs",docs)        
    return np.array(docs)

""" #### TO get closest using COSINE
most_sim_ids = closest_docs_by_index(doc_topic_dist, new_doc_distribution, 10)
#print(closest)
###### TO get closest usint JENSEN SHANON
#most_sim_ids = get_most_similar_documents(new_doc_distribution,doc_topic_dist)

print(most_sim_ids)
for idx in most_sim_ids[0]:
  print(docs[idx])
 """
##### TESTING PRECISION 


########################## TESTING FOR TEST DATA

####GET TEST DOCUMENTS : 
#fetch data from 20NG
parent_path = os.getcwd()
child_path = "20news-bydate-test"
path = os.path.join(parent_path, child_path) 
os.chdir(path)
labelstest=[]
docstest = []
for subdirectory in os.listdir(os.getcwd()):
    sub_path = os.path.join(path,subdirectory)
    #print("Sub directory : ",subdirectory)
    #print(sub_path)
    os.chdir(sub_path)
    for textfile in os.listdir(os.getcwd()):
        textfilepath = os.path.join(sub_path,textfile)
        with open(textfilepath, 'rb') as f:
            content = f.read()
            docstest.append(content)
            labelstest.append(subdirectory)
    #print(docs[-1])
    os.chdir(path)

os.chdir(parent_path)
print("len is test data",len(docstest))
######PRE PROCESS TEST DATA
docs_list= clean_docs(docstest)
#frequency = defaultdict(int)
processed_corpus_test = getfrequency(docs_list)

query_list = []

precision = 0
#### count for 10
for idx in range(0,len(labelstest)):
    tokens = processed_corpus_test[idx]
    new_bow = dictionary.doc2bow(tokens)
    new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])
    most_sim_ids = closest_docs_by_index(doc_topic_dist, new_doc_distribution, 10)
    docprecision = 0
    for topdoc in most_sim_ids[0]:
        if(labelstest[idx]==labels[topdoc]):
            docprecision = docprecision + 1
    docprecision = docprecision/10
    precision = precision + docprecision
print("Precision 10 : ",precision/len(labelstest))

precision = 0
#### count for 5
for idx in range(0,len(labelstest)):
    tokens = processed_corpus_test[idx]
    new_bow = dictionary.doc2bow(tokens)
    new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])
    most_sim_ids = closest_docs_by_index(doc_topic_dist, new_doc_distribution, 5)
    docprecision = 0
    for topdoc in most_sim_ids[0]:
        if(labelstest[idx]==labels[topdoc]):
            docprecision = docprecision + 1
    docprecision = docprecision/5
    precision = precision + docprecision
print("Precision 5 : ",precision/len(labelstest))


precision = 0
#### count for 3
for idx in range(0,len(labelstest)):
    tokens = processed_corpus_test[idx]
    new_bow = dictionary.doc2bow(tokens)
    new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])
    most_sim_ids = closest_docs_by_index(doc_topic_dist, new_doc_distribution, 3)
    docprecision = 0
    for topdoc in most_sim_ids[0]:
        if(labelstest[idx]==labels[topdoc]):
            docprecision = docprecision + 1
    docprecision = docprecision/3
    precision = precision + docprecision
print("Precision 3 : ",precision/len(labelstest))


precision = 0
#### count for 1
for idx in range(0,len(labelstest)):
    tokens = processed_corpus_test[idx]
    new_bow = dictionary.doc2bow(tokens)
    new_doc_distribution = np.array([tup[1] for tup in model.get_document_topics(bow=new_bow)])
    most_sim_ids = closest_docs_by_index(doc_topic_dist, new_doc_distribution, 1)
    docprecision = 0
    for topdoc in most_sim_ids[0]:
        if(labelstest[idx]==labels[topdoc]):
            docprecision = docprecision + 1
    docprecision = docprecision/1
    precision = precision + docprecision
print("Precision 1 : ",precision/len(labelstest))


   

