import math

class TfidfModel():

    def __init__(self):
        '''
        cfs: the frequency of the token in a specific document.
        dfs: the document frequency of the token in the collection 
        n_docs: number of documents in the collection
        n_words: number of words in the document
        '''
        self.idf = None
        self.tf = None
        
    def __call__(self, term, cfs, dfs, n_docs, n_words):
        return  self._tfidf(term, cfs, dfs, n_docs, n_words)
    
   
    def _tf(self, term, cf, n_words, sublinear_tf=False):
        '''Relative frequency of the term:
        Number of occurrences of the term in the document/Total number of words in the document
        '''
        
        self.tf = (cf / n_words) 
        #eps = 1e-15
        #self.tf += eps
        if sublinear_tf:
            self.tf = math.log(tf) + 1
      
        return self.tf
  
        
    def _idf(self, term, df, n_docs, smooth_idf=True):
        """
        idf - логарифм от количества текстов (n_docs) в корпусе, 
        разделенное на число текстов, где термин встречается (df). 
        Если термин не появляется в корпусе или появляется во всех документах, 
        возвращается 0.0.
        Принимает слово, для которого считаем IDF, его документную частоту и 
        общее число документов в коллекции.
        """
        
        df += int(smooth_idf)
        n_docs += int(smooth_idf)
        # формула из sklearn.feature_extraction.text
        result = math.log(n_docs / df ) + 1
        # idf(t) = log [ n / (df(t) + 1) ]) standard textbook notation 
        return result
    
    
    def _tfidf(self, term, 
            cf,  # term frequency
            df,  # document frequency
            n_docs, 
            n_words, 
            smooth_idf=True, 
            sublinear_tf=False):
        ''' tf-idf(t, d) = tf(t, d) * idf(t)
        
        df: document frequency is the number of documents
            in the document set that contain the term t
        idf:
        '''
        self.idf = self._idf(term, df, n_docs, smooth_idf)
        self.tf = self._tf(term, cf, n_words, sublinear_tf)
        return self.tf * self.idf