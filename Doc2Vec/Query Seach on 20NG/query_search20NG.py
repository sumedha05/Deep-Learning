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
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

""" #open the documents of dataset into list docs
with open('/home/mili/dev/data/doc2vec/docs.data','rb') as handle:
    docs=pickle.load(handle)

#open the query in variable query
with open('/home/mili/dev/data/doc2vec/query.data','rb') as handle:
    query=pickle.load(handle)

print(query)

print(len(docs))
#print(docs[9]) """

#fetch data from 20NG
parent_path = os.getcwd()
child_path = "20news-bydate-train"
path = os.path.join(parent_path, child_path) 
os.chdir(path)

docs = []
for subdirectory in os.listdir(os.getcwd()):
    sub_path = os.path.join(path,subdirectory)
    print(sub_path)
    os.chdir(sub_path)
    for textfile in os.listdir(os.getcwd()):
        textfilepath = os.path.join(sub_path,textfile)
        with open(textfilepath, 'rb') as f:
            content = f.read()
            docs.append(content)
    #print(docs[-1])
    os.chdir(path)


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
    
    #print(text)
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

###### REMOVE 50-500 FREQ WORDS
def getfrequency(docs):
    for doc in docs:
        for token in doc:
            frequency[token] += 1
            #print("token ",token," freq ",frequency[token])

    processed_corpus = [[token for token in doc if frequency[token] >2] for doc in docs]
    #print(processed_corpus)
    return processed_corpus

#print(docs[7])
docs_list= clean_docs(docs)
frequency = defaultdict(int)
processed_corpus = getfrequency(docs_list)
#print("here")
#print(processed_corpus)
tagged_data = convert_newsgroup(processed_corpus)

#if you donot want this preprocessing, use the below mentioned line and remove the above one
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
    
dm_model =  Doc2Vec(dm=1, dm_mean=1, sample=1e-5, vector_size=300,epochs = 100, window=8, min_count=2, workers=cores)

#dm_model.load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
dm_model.build_vocab(tagged_data)  # PV-DM/concat requires one special NULL word so it serves as template

#change passes for different number of iterations
alpha, min_alpha, passes = (0.025, 0.001, 100)
alpha_delta = (alpha - min_alpha) / passes
#from sklearn.grid_search import GridSearchCV

##### TRAINING
dm_model.train(tagged_data,total_examples=dm_model.corpus_count,epochs=dm_model.epochs)       


""" for epoch in range(passes):
    shuffle(tagged_data)
    #printing alpha here
    print(alpha)
    dm_model.train(tagged_data,total_examples=dm_model.corpus_count,epochs=dm_model.epochs)       
    dm_model.alpha, dm_model.min_alpha = alpha, alpha
    alpha -= alpha_delta """


#########################################################
# put query in a loop to get result for list of queries
""" for query in queries:
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data1)
    similar_doc = dm_model.dv.most_similar([v1],topn=100)
    print(similar_doc)
    #you can index it as  similar_doc[idx]
 """
for test in range(10):
    query = input("Enter input ")
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data)

    #print("V1_infer", v1)
    similar_doc = dm_model.dv.most_similar([v1],topn=100)

    print(similar_doc)
    for index in range(3):
        print("_________________doc",similar_doc[index][0],"______________\n")
        print("Processes text " , processed_corpus[int(similar_doc[index][0])])
        print("Original text " , docs[int(similar_doc[index][0])])





