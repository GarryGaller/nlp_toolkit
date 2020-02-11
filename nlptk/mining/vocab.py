import time
from typing import List
from collections import Counter,defaultdict
from pprint import pprint

from nlptk.measures.tfidf import TfidfModel
import numpy as np

class Vocabulary():
    
    
    def __iadd__(self,other):
        if other.tokens:
            self._iter_ndocs() 
            self._iter_vocab(other.tokens)
            self._iter_cfs(other.tokens)
            self._iter_ccfs(other.tokens)
            self._iter_dfs(other.tokens)
            self._iter_tfidf()
        return self
    
    def __init__(self,tokens:List[str]=[]):
        
        #all_tokens = chain.from_iterable(corpus)
        self.tokens = tokens 
        self.ndocs = 0
        # словарь всех токенов в корпусе отображающий их частоту
        self._ccfs = Counter()
        # словарь всех токенов корпуса отображающий их идентификаторы
        self._vocab = defaultdict()
        self._vocab.default_factory = lambda :len(self._vocab) + 1
        self._inv_vocab = {} # обратный для _vocab словарь; создается по запросу
        # список частотных словарей для каждого документа в корпусе
        self._cfs = []
        # словарь всех токенов в корпусе отображающий число документов в которые
        # входит слово
        self._dfs = defaultdict(int)
        # словарь idf значений для токенов
        self._idf = {}
        # список словарей значений tf для токенов
        self._tf = []
        # список словарей с TF-IDF для каждого документа в корпусе
        self._tfidf  = []  
        
        '''
        if tokens:
            self._iter_ndocs() 
            self._iter_vocab(tokens)
            self._iter_cfs(tokens)
            self._iter_ccfs(tokens)
            self._iter_dfs(tokens)
            self._iter_tfidf()    
        '''
    
    def tok2id(self,tokens):
        return [self._vocab[token] for token in tokens]
    
    def id2tok(self,ids):
        if not self._inv_vocab:
            self._inv_vocab = {
                idx:tok for tok,idx in self._vocab.items()
            }
        return [self._inv_vocab[idx] for idx in ids]
    
    def id2map(self,idx=None):
        if not self._inv_vocab:
            self._inv_vocab = {
                idx:tok for tok,idx in self._vocab.items()
            }
        if idx is not None:
            return self._inv_vocab[idx]
        return self._inv_vocab

    def tok2map(self,token=None):
        if token is not None:
            return self._vocab[token]
        return self._vocab
    
    
    def _iter_ndocs(self):
        self.ndocs += 1
    
    
    def _iter_ccfs(self,tokens):
        tokens = self.tok2id(tokens)
        self._ccfs += Counter(tokens)     
        
    
    def _iter_vocab(self,tokens):
        
        tokens = set(tokens)
        for key in tokens:
            self._vocab[key]
        
    '''
    def _iter_dfs(self,tokens):    
        start = time.time()
        for token in self._vocab:
            res = 1.0  if token in tokens else 0
            self._dfs[token] += res
        print('_iter_dfs',time.time() - start, len(self._vocab), len(tokens))
    '''
    
    def _iter_dfs(self,tokens):    
        tokens = self.tok2id(tokens)
        for token in set(tokens):
            self._dfs[token] += 1
        
    def _iter_cfs(self,tokens):        
        tokens = self.tok2id(tokens)
        self._cfs.append(Counter(tokens))
    
    
    def tf(self, n_doc, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._tf[n_doc].get(idx,0)
       
        res = {self.id2map(idx):val 
            for idx,val in self._tf[n_doc].items()
        }
        return res 
                      
    
    def idf(self, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._idf.get(idx,0)
        res = {
            self.id2map(idx):val for idx,val in self._idf.items()
        }
        return res
    
    
    def cfs(self, n_doc, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._cfs[n_doc].get(idx,0)
       
        res = {self.id2map(idx):val 
            for idx,val in self._cfs[n_doc].items()
        }
        return res 
    
       
    def dfs(self, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._dfs.get(idx,0)
        res = {self.id2map(idx):val 
            for idx,val in self._dfs.items()
        } 
        return res 
    
        
    def ccfs(self, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._ccfs.get(idx,0)
        res = {self.id2map(idx):val 
            for idx,val in self._ccfs.items()
        }
        return res
    
    
    def tfidf(self, n_doc, token=None):
        
        if token:
            idx = self.tok2map(token)
            return self._tfidf[n_doc].get(idx,0)
        res = {self.id2map(idx):val 
            for idx,val in self._tfidf[n_doc].items()
        }
        return res    
        
    # итеративный вариант с пересчетом всех значений для каждого текста
    def _iter_tfidf(self):        
        #print('---',self.ndocs,'---')
        for n,text in enumerate(self._cfs):
            tfidf_dict = {}
            tf_dict = {}
            tokens = text.keys()
            n_words = sum(text.values())
            
            for token in tokens: 
                compute = TfidfModel()
                tfidf = compute(
                    self._cfs[n][token], 
                    n_words,
                    self._dfs[token],
                    self.ndocs 
                )
                tfidf_dict[token] = tfidf
                self._idf[token] = compute.idf
                tf_dict[token] = compute.tf
            
            self._tf.insert(n,tf_dict)
            self._tfidf.insert(n,tfidf_dict)
    
    # вариант с вычислением по всему корпусу
    def compute_tfidf(self,corpus:List[List[str]]):        
        
        for n,tokens in enumerate(corpus):
            tfidf_dict = {}
            tf_dict = {}
            tokens = self.tok2id(tokens)
            n_words = len(tokens)
            for token in set(tokens): 
                compute = TfidfModel()
                tfidf = compute(
                    self._cfs[n][token], 
                    n_words,
                    self._dfs[token],
                    self.ndocs 
                )
                tfidf_dict[token] = tfidf
                self._idf[token] = compute.idf
                tf_dict[token] = compute.tf
                
            self._tf.insert(n,tf_dict)
            self._tfidf.insert(n,tfidf_dict)
    
    
    def text2seq(self,tokens):
        return self.tok2id(tokens)   
     
    
    def text_to_onehot_vector(self,tokens,n=None,vocab=None):
        vocab = vocab or self._vocab
        vector = np.zeros((len(vocab),))
        
        for i,token in enumerate(tokens):
            if token in self._vocab.values():
                vector[i] = 1 
        
        return vector

    
    def text_to_freq_vector(self, tokens,n,vocab=None):
        vocab = vocab or self._vocab
        vector = np.zeros((len(vocab),))
        
        for i,token in enumerate(tokens):
            if token in self._vocab.values():
                vector[i] = self._cfs[n][token] 
        
        return vector 
    
    
    def text_to_tfidf_vector(self,tokens,n,vocab=None):
        vocab = vocab or self._vocab
        vector = np.zeros((len(vocab),))
        n_tokens = len(tokens)
        for i,token in enumerate(tokens):
            if token in self._vocab.values():
                vector[i] = TfidfModel()(
                    token, 
                    self._cfs[n][token], 
                    self._dfs[token], 
                    self.ndocs, 
                    n_tokens
                )
        
        return vector
    
    
    def texts_to_matrix(self,texts,vocab=None,typ="onehot"):
         '''texts - список списков слов для каждого 
         токенизированного текста'''
         
         vocab = vocab or self._vocab
         matrix = np.zeros_like(texts)
         
         for i,tokens in enumerate(texts):
            matrix[i] = getattr(
                            self, 
                            'text_to_' + typ + '_vector'
                )(tokens,i,vocab) 
         
         return matrix
    
    
    def freq(self,n_doc=None,reverse=True):    
        '''отсортированный по убыванию (default) частотности список слов'''
        if n_doc is not None:
            res = sorted(
                self._cfs[n_doc],
                key=self._cfs[n_doc].get,
                reverse=reverse
            )
        else:
            res = sorted(self._ccfs,key=self._ccfs.get,reverse=reverse)
        return self.id2tok(res)
       

    def hapaxes(self,n_doc=None):
        '''Возвращает список слов имеющих только одно вхождение 
        в текст или весь корпус'''    
        
        if n_doc is not None:
            cfs = self._cfs[n_doc] # берем частотный словарь конкретного документа
        else:
            cfs = self._ccfs # иначе - общий словарь частот для всего корпуса
        
        result = []
        #result = [token for token, freq in cfs.items() if freq == 1]
        for token,freq in cfs.most_common()[:-len(cfs)-1:-1]:
            if freq == 1:
                result.append(token)
            elif freq > 1:
                break   
        
        return self.id2tok(result)

if __name__ == "__main__":
    
    import pandas as pd
    pd.options.display.max_rows = 9999    # 60 по умолчанию
    pd.options.display.max_columns = 999  # 20 по умолчанию
    pd.options.display.width = 500        # 80 по умолчанию
    pd.options.display.max_colwidth = 150 # 50 по умолчанию
    

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

    print(vocab.tfidf(0,'example'))
    
    print(vocab.ndocs)
    print('--vocab__')
    pprint(vocab._vocab)
    print('--cfs--')
    pprint([vocab.cfs(n) for n in range(vocab.ndocs)])
    print('--ccfs--')
    pprint(vocab.ccfs())
    print('--dfs--')
    pprint(vocab.dfs())

    print('--tfidf--')
    tfidf = [vocab.tfidf(n) for n in range(vocab.ndocs)]
    pprint(tfidf)
    print('--compute tfidf--')
    vocab.compute_tfidf([doc.split() for doc in corpus])
    tfidf2 = [vocab.tfidf(n) for n in range(vocab.ndocs)]
    pprint(tfidf2)
    
    
    
    print('--idf--')
    pprint(vocab.idf())
    print('--tf--')
    pprint(vocab.tf(0))
    print('--hapaxes--')
    pprint(vocab.hapaxes())
    print('--freq--')
    pprint(vocab.freq())
    
    #tfidf = lambda cf,nwords,df,ndocs: (cf/nwords) * ((math.log(ndocs + 1)/(df + 1)) + 1)
    
    columns = vocab._vocab.keys()
    
    
    df = pd.DataFrame(tfidf,columns=columns)
    df.fillna(0.0, inplace=True)
    print(df)
    
    df = pd.DataFrame(tfidf2,columns=columns)
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
