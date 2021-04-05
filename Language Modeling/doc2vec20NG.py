from sklearn.svm import LinearSVC
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#from abstract import *
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import numpy as np
import sys

categories = ['alt.atheism','rec.sport.baseball','talk.politics.mideast','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

from pprint import pprint
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer


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
def clean_news(articles):
    
    clean = []
    
    for article in articles:
        clean.append(preprocessing(article,stop=True,sent=False, stem=False))
        #print clean
        #sys.exit(1)
    
    return clean
        

newsgroups_train.data = clean_news(newsgroups_train.data)
newsgroups_test.data = clean_news(newsgroups_test.data)

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



#NewsgroupDocument = namedtuple('NewsGroupDocument', 'words tags split category')
#doc_count = 0 # Used to generate unique id for all documents across both train and test

all_newsgroup_documents = []

#Used to convert newsgroup corpus into Doc2Vec formats
def convert_newsgroup(docs,split):
    #global doc_count
    tagged_documents = []
    
    for i,v in enumerate(docs):
        label = '%s_%s'%(split,i)
        tagged_documents.append(TaggedDocument(v, [label]))
    
    return tagged_documents
    
    #for doc, label in zip(docs,labels):
      #  doc_count += 1
        #print doc
        #words = gensim.utils.to_unicode(doc).split() # expected by gensim
        #tags = [doc_count] #needs to be a list. Exp with having multiple tags
        #all_newsgroup_documents.append(NewsgroupDocument(words,tags,split,label))
        #print words
        
        #if doc_count == 5:
        #    print all_newsgroup_documents
        #    break
            #sys.exit(0)
    
test_docs = convert_newsgroup(newsgroups_test.data,'test')
train_docs = convert_newsgroup(newsgroups_train.data,'train')

all_newsgroup_documents.extend(train_docs)
all_newsgroup_documents.extend(test_docs)
#train_docs = [doc for doc in all_newsgroup_documents if doc.split == 'train']
#test_docs = [doc for doc in all_newsgroup_documents if doc.split == 'test']
doc_list = all_newsgroup_documents[:]  # for reshuffling per pass

print('%d docs: %d train, %d test' % (len(doc_list), len(train_docs), len(test_docs)))
print(len(newsgroups_train.target))

#Doc2Vec(dm=0, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=3, workers=cores)
#Doc2Vec(dm=0, dm_mean=1, size=300, window=5, negative=5, hs=0, min_count=3, workers=cores),

dbow_model = Doc2Vec(dm=0, dm_concat=1,sample=1e-5, vector_size=300, window=5, negative=5, hs=0, min_count=2, workers=cores)
dm_model =  Doc2Vec(dm=1, dm_mean=1, sample=1e-5, vector_size=300, window=10, negative=5, hs=0, min_count=2, workers=cores)

# TODO speed setup by sharing results of 1st model's vocabulary scan
#dbow_model.wv('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
dbow_model.build_vocab(all_newsgroup_documents)  # PV-DM/concat requires one special NULL word so it serves as template

#dm_model.load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
dm_model.build_vocab(all_newsgroup_documents)  # PV-DM/concat requires one special NULL word so it serves as template



# Models to evaluate
#simple_models = [
 
    # PV-DBOW  0.86 with Stem & hs=0
    #Doc2Vec(dm=0, dm_concat=1,sample=1e-5, size=300, window=5, negative=5, hs=0, min_count=2, workers=cores),
    
    #
    #Doc2Vec(dm=0, dm_mean=1, sample=1e-5,size=300, window=5, negative=5, hs=0, min_count=2, workers=cores),
    
    
    #Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=1, workers=cores),
    
    # PV-DM w/average No good 0.84
    #Doc2Vec(dm=1, dm_mean=1, sample=1e-5, size=300, window=10, negative=5, hs=0, min_count=2, workers=cores),
    
    # PV-DM w/sum
    #Doc2Vec(dm=1, dm_mean=0, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
#]

# speed setup by sharing results of 1st model's vocabulary scan
#simple_models[0].load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
#simple_models[0].build_vocab(all_newsgroup_documents)  # PV-DM/concat requires one special NULL word so it serves as template


#print(simple_models[0])

#for model in simple_models[1:]:
    #model.reset_from(simple_models[0])
#    model.load_word2vec_forma('/home/skillachie/nlpArea51/doc2vec/GoogleNews-vectors-negative300.bin', binary=True)
#    model.build_vocab(all_newsgroup_documents)
#    print(model)

#models_by_name = OrderedDict((str(model), model) for model in simple_models)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
dbow_dmm_model = ConcatenatedDoc2Vec([dbow_model, dm_model])
#models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

from collections import defaultdict
best_error = defaultdict(lambda :1.0)  # to selectively-print only best errors achieved

#Get Vectors From Word2Vec
def extract_vectors(model,docs):
    
    vectors_list = []
    
    for doc_no in range(len(docs)):
        doc_label = docs[doc_no].tags[0]
        doc_vector = model.docvecs[doc_label]
        vectors_list.append(doc_vector)
        
    return vectors_list

# TODO inferred vectors

def get_infer_vectors(model,docs):
    
    vecs = []
    for doc in docs:
        vecs.append(model.infer_vector(doc.words))
    return vecs

from random import shuffle
alpha, min_alpha, passes = (0.025, 0.001, 10)
alpha_delta = (alpha - min_alpha) / passes
#from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC


for epoch in range(passes):
    shuffle(doc_list)
    
    #for name, train_model in models_by_name.items():
        
    #Train
    print(alpha)
    
    dbow_model.alpha, dbow_model.min_alpha = alpha, alpha
    dbow_model.train(doc_list,total_examples=dbow_model.corpus_count,epochs=dbow_model.epochs)
    
    dm_model.alpha, dm_model.min_alpha = alpha, alpha
    dm_model.train(doc_list,total_examples=dm_model.corpus_count,epochs=dm_model.epochs)
    
    
    dbow_dmm_model.alpha, dbow_dmm_model.min_alpha = alpha, alpha
    dbow_dmm_model.train(doc_list)
    
    
    
        
    alpha -= alpha_delta

    #Evaluation
#train_vectors = extract_vectors(dbow_dmm_model,train_docs)
#test_vectors = extract_vectors(dbow_dmm_model,test_docs)

#print("Give query string : ")
query_string = input("Enter query string :")
test_data = word_tokenize(query_string.lower())
v1 = dm_model.infer_vector(test_data)
#print("V1_infer", v1)

similar_doc = dm_model.dv.most_similar([v1],topn=100)
print("TOP 100 : Document Name, Similarity Score\n")
#print(similar_doc)
print(similar_doc)

#model = LinearSVC()
#penalties = np.array([0.001,0.002,0.003,0.004,0.005,0.007,0.008,0.009,0.01,0.05,0.04,0.03,0.02])
#grid = GridSearchCV(estimator=model ,n_jobs=7,param_grid=dict(C=penalties))
#grid.fit(train_vectors, newsgroups_train.target)
        
# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_estimator_.C)

#clf = LinearSVC(C=0.009)
#clf = LinearSVC(C=0.0025)
#clf.fit(train_vectors, newsgroups_train.target)

#predDoc = clf.predict(test_vectors)
        
#print classification_report(le.inverse_transform(newsgroups_test.target),le.inverse_transform(predDoc))

