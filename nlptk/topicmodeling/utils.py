import glob,os,sys

class Path():
    '''
    >>> paths = Path(source,"*.txt")
    >>> for path in paths:
            lines = Stream(path)
                for line in lines:
                    print(line)
    '''
    
    def __init__(self, source, pattern):
        self.source = source
        self.pattern = pattern     
  
    def __getpaths__(self):
        source = os.path.join(self.source, self.pattern)
        
        files = glob.glob(source)
        for filename in files:
            yield os.path.join(source, filename)
                
    def __iter__(self):
        return self.__getpaths__()         
                

class Stream():
    '''
    >>> lines = Stream(path)
    >>> for line in lines:
            print(line)
    '''        
    def __init__(self, 
            encoding=None,
            sentencizer=None,
            text_filters=[]
           ):
        
        self.encoding = encoding
        self.__sentencizer = sentencizer
        self.__text_filters = text_filters
        
    def __call__(self,path):
        """Read lines from filepath."""
        
        
        with open(path,'r',
            encoding = (
                self.encoding(path)  
                if callable(self.encoding) 
                else self.encoding)  
            ) as fd:
            # обрабатываем либо по предложению    
            if self.__sentencizer:
                text = self.preprocess_text(fd.read())
                for sentence in self.__sentencizer(text):
                    yield sentence
            # либо по строке
            else:  
                for line in fd:
                    yield line
    
    def preprocess_text(self,text):
        for text_filter in self.__text_filters:
            text = text_filter(text)
        return text
    

class Lemmatizer():
    
    def __init__(self, lemmatizer=None,
                allowed_tags=set(), disallowed_tags=set()):
        self.lemmatize = lemmatizer
        self.allowed_tags = set(allowed_tags) - set(disallowed_tags)
        
        
    def __call__(self,data):
        
        if isinstance(data,(str)):
            data = [data]   
        self.allowed_tags 
        
        for lemma,pos in self.lemmatize(data,pos=True):
            if self.allowed_tags:
                if (self.allowed_tags) and (pos in self.allowed_tags):
                    yield lemma   
            else:
                 yield lemma
            


class Tokenizer():
    
    def __init__(self,tokenizer=None):
        self.tokenize = tokenizer
        
    def __call__(self,data):
        return self.tokenize(data)    
  

class CharCleaner():
    
    def __init__(self,cleaners=None):
        self.cleaners = cleaners
        
    def __call__(self,data):
        for cleaner in self.cleaners:
            data = cleaner(data)
        
        return data  


class TokenCleaner():
    
    def __init__(self,cleaners=None):
        self.cleaners = cleaners
        
    def __call__(self,data):
        for cleaner in self.cleaners:
            data = cleaner(data)
        
        return data  


class LemmaCleaner():
    
    def __init__(self,cleaners=None):
        self.cleaners = cleaners
        
    def __call__(self,data):
        for cleaner in self.cleaners:
            data = cleaner(data)
        
        return data  
