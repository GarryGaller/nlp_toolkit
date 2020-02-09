import time
from typing import List
from collections import Counter,defaultdict
from pprint import pprint

from nlptk.measures.tfidf import TfidfModel
import numpy as np

class Vocabulary():
    
    
    def __iadd__(self,other):
        if other.tokens:
            self._n_docs() 
            self._vocab(other.tokens)
            self._cfs(other.tokens)
            self._ccfs(other.tokens)
            self._dfs(other.tokens)
            #self._tfidf(other.tokens)
        return self
    
    def __init__(self,tokens:List[str]=[]):
        
        #all_tokens = chain.from_iterable(corpus)
        self.tokens = tokens 
        self.n_docs = 0
        # словарь всех токенов в корпусе отображающий их частоту
        self.ccfs = Counter()
        # словарь всех токенов корпуса отображающий их идентификаторы
        self.vocab = defaultdict()
        self.vocab.default_factory = self.vocab.__len__
        # список частотных словарей для каждого документа в корпусе
        self.cfs = []
        # словарь всех токенов в корпусе отображающий число документов в которые
        # входит слово
        self.dfs = defaultdict(int)
        # словарь idf значений для токенов
        self.idf = {}
        # список словарей значений tf для токенов
        self.tf = []
        # список словарей с TF-IDF для каждого документа в корпусе
        self.tfidf  = []  
        
        if tokens:
            self._n_docs() 
            self._vocab(tokens)
            self._cfs(tokens)
            self._ccfs(tokens)
            self._dfs(tokens)
            #self._tfidf(tokens)
        
    def _n_docs(self):
        self.n_docs += 1
    
    
    def _ccfs(self,tokens):
        
        self.ccfs += Counter(tokens)     
        
    
    def _vocab(self,tokens):
        
        tokens = set(tokens)
        for key in tokens:
            self.vocab[key]
        
    
    '''
    def _dfs(self,tokens):    
        start = time.time()
        for token in self.vocab:
            res = 1.0  if token in tokens else 0
            self.dfs[token] += res
        print('_dfs',time.time() - start, len(self.vocab), len(tokens))
    '''
    
    def _dfs(self,tokens):    
        
        for token in set(tokens):
            self.dfs[token] += 1
        
    def _cfs(self,tokens):        
        
        self.cfs.append(Counter(tokens))
                      
    
    def compute_tfidf__(self,corpus:[List[str]]):        
        
        
            tfidf_dict = {}
            tf_dict = {}
            for token in set(tokens): 
                n_tokens = len(tokens)
                compute = TfidfModel()
                tfidf = compute(
                    token, 
                    self.cfs[n][token], 
                    self.dfs[token], 
                    self.n_docs,
                    n_tokens 
                    )
                tfidf_dict[token] = tfidf
                self.idf[token] = compute.idf
                tf_dict[token] = compute.tf
                
            self.tf.append(tf_dict)
            self.tfidf.append(tfidf_dict)
    
    
    
    def compute_tfidf(self,corpus:List[List[str]]):        
        
        for n,tokens in enumerate(corpus):
            tfidf_dict = {}
            tf_dict = {}
            for token in set(tokens): 
                n_tokens = len(tokens)
                compute = TfidfModel()
                tfidf = compute(
                    token, 
                    self.cfs[n][token], 
                    self.dfs[token], 
                    self.n_docs,
                    n_tokens 
                    )
                tfidf_dict[token] = tfidf
                self.idf[token] = compute.idf
                tf_dict[token] = compute.tf
                
            self.tf.append(tf_dict)
            self.tfidf.append(tfidf_dict)
    
    
    def text2seq(self,tokens):
        return [self.vocab.get(token) for token in tokens]   
     
    
    def text_to_onehot_vector(self,tokens,n=None,vocab=None):
        vocab = vocab or self.vocab
        vector = np.zeros((len(vocab),))
        
        for i,token in enumerate(tokens):
            if token in self.vocab:
                vector[i] = 1 
        
        return vector

    
    def text_to_freq_vector(self, tokens,n,vocab=None):
        vocab = vocab or self.vocab
        vector = np.zeros((len(vocab),))
        
        for i,token in enumerate(tokens):
            if token in self.vocab:
                vector[i] = self.cfs[n][token] 
        
        return vector 
    
    
    def text_to_tfidf_vector(self,tokens,n,vocab=None):
        vocab = vocab or self.vocab
        vector = np.zeros((len(vocab),))
        n_tokens = len(tokens)
        for i,token in enumerate(tokens):
            if token in self.vocab:
                vector[i] = TfidfModel()(
                    token, 
                    self.cfs[n][token], 
                    self.dfs[token], 
                    self.n_docs, 
                    n_tokens
                )
        
        return vector
    
    
    def texts_to_matrix(self,texts,vocab=None,typ="onehot"):
         '''texts - список списков слов для каждого 
         токенизированного текста'''
         
         vocab = vocab or self.vocab
         matrix = np.zeros_like(texts)
         
         for i,tokens in enumerate(texts):
            matrix[i] = getattr(
                            self, 
                            'text_to_' + typ + '_vector'
                )(tokens,i,vocab) 
         
         return matrix
    
    
    def freq(self,vocab):    
        '''отсортированный по убыванию частотности список слов'''
        return sorted(self.cfs,key=self.csf.get,reverse=True)
       
    def idx2tok(self):
        return {idx:tok for tok,idx in self.vocab.items()}

    def hapaxes(self,n=-1):
        '''Возвращает список слов имеющих только одно вхождение 
        в текст или весь корпус'''    
        
        if n != -1:
            cfs = self.cfs[n] # берем частотный словарь конкретного документа
        else:
            cfs = self.ccfs # иначе - общий словарь частот для всео корпуса
        
        return [token for token, freq in cfs.items() if freq == 1]

if __name__ == "__main__":
    
    import pandas as pd


    corpus = [
        "the sky is blue",
        "the sun is bright",
        "the sun in the sky is bright"
    ]

    corpus = """
Simple example example with Cats and Mouse
Another simple example with dogs and cats
Another simple example with mouse and cheese
""".lower().split("\n")[1:-1] 
    
    print(corpus)    
    
    
    #vocab = Vocabulary([1,2,3,4,5])
    #vocab += Vocabulary([1,2,3,4,5,6])

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    vocab = Vocabulary()
    for doc in corpus:
        vocab += Vocabulary(doc.split())


    print(vocab.n_docs)
    print('--vocab__')
    pprint(vocab.vocab)
    print('--cfs--')
    pprint(vocab.cfs)
    print('--ccfs--')
    pprint(vocab.ccfs)
    print('--dfs--')
    pprint(vocab.dfs)


    vocab.compute_tfidf([doc.split() for doc in corpus])
    print('--tfidf--')

    pprint(vocab.tfidf)
    print('--idf--')
    pprint(vocab.idf)
    print('--tf--')
    pprint(vocab.tf)

    columns = vocab.vocab.keys()
    

    df = pd.DataFrame(vocab.tfidf,columns=columns)
    df.fillna(0.0, inplace=True)
    print(df)

    #print(vocab.texts_to_matrix([doc.split() for doc in corpus]))
    print('-' * 20)
    
    
    #corpus = ['1 2 3 4 5','1 2 3 4 5 6']

    vec = TfidfVectorizer( 
        corpus, 
        stop_words=[],
        #token_pattern=r"\d", 
        smooth_idf=True,
        use_idf=False,
        #sublinear_tf=True,
        norm=None
    )
    print()
    X = vec.fit_transform(corpus)
    print(vec.get_feature_names())
    #print(vec.idf_)
    print(X.todense())


    df = pd.DataFrame(X.todense(),columns=columns)
    print(df)
