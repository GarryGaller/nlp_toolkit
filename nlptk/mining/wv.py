from typing import List
from gensim.models.word2vec import Word2Vec
from nlptk.mining.utils import datapath

class word2vec():
    
    def __init__(self,**kwargs):
        self.model = None
        self.params = dict(size=300, window=5, min_count=5)
        self.params.update(kwargs)
        
        
    def load(self, name, datadir='data'):
        loadpath = datapath(name, datadir=datadir, ext=".model").full
        self.model = Word2Vec.load(loadpath)    
    
    def save(self, name, datadir='data'):
        loadpath = datapath(name, datadir=datadir, ext=".model").full
        if self.model:
            self.model.save(loadpath)  
        else:
            raise ValueError('The model is not created')
    
    def train(self, sents:List[List[str]], **kwargs):
        self.params.update(kwargs)
        self.model = Word2Vec(
                sents,
                **self.params
            )
        return self.model
    
    def __call__(self, sents:List[List[str]], **kwargs):
        datadir = kwargs.pop(datadir,'')
        name = kwargs.pop(name,None)
        self.params.update(kwargs)
        
        if name is not None:
            loadpath = datapath(
                name, 
                datadir=datadir, 
                ext=".model").full
            if os.path.exists(loadpath):
                self.model = Word2Vec.load(loadpath)
            else:
                raise FileNotFoundError(loadpath)
            
        else:
            self.model = Word2Vec(
                sents,
                **self.params
            )
        return self.model        
        
