import os, sys, io
import io, glob
import string
import dawg
import pickle
import math
import time
import heapq

import nltk
from nltk import FreqDist, ConditionalFreqDist

from textwrap import shorten
from typing import List
from collections import Counter,defaultdict
from pprint import pprint,pformat
from functools import partial
from itertools import chain
from collections import OrderedDict
from itertools import islice
#from docx import Document
import nlptk
from nlptk.mining.utils import chardetector, datapath
from nlptk.mining.vocab import Vocabulary
from nlptk.misc.mixins import (
                StripperMixin,
                RemoverMixin,
                TokenizerMixin,
                TaggerMixin,
                LemmatizerMixin,
                SentencizerMixin)


if sys.version_info < (3,6):
    import win_unicode_console
    win_unicode_console.enable()


class Lexicon():
    def __init__(self,name,tokens=[]):
        self.name = name
        self.tokens = tokens
        
    def save(self):
        if self.tokens:
            c_dawg = dawg.CompletionDAWG(self.tokens)
            c_dawg.save(r'{}.dawg'.format(self.name))
        
    def load(self):
        c_dawg = dawg.CompletionDAWG()
        return c_dawg.load(r'{}.dawg'.format(self.name)) 


class Prep(TokenizerMixin,
            LemmatizerMixin,
            SentencizerMixin,
            TaggerMixin
            ):
        
    
    def __init__(self, *args, **kwargs):
        
        defaults = {
            'sentencizer':self.sentencize_nltk, 
            'tokenizer':partial(
                self.toktok_tokenize,       #self.treebank_word_tokenize,
                strip=string.punctuation    # ''.join(set(string.punctuation) - {"'"})
            ),
            'tagger':self.tagger_nltk, # self.tagger_4ngram
            'lemmatizer':partial(
                    self.lemmatize_nltk,
                    pos=True,
                    normalize_uppercase=str.capitalize) #tagset='universal'
        }
        
        self.__dict__.update({
            key:kwargs[key] if key in kwargs else val 
                for key,val in defaults.items() 
            }
        )



class TextCleaner(StripperMixin):
    
    rules = OrderedDict(
                    hyphenation=(True,),
                    accent=(True,),
                    # нельзя удалять, если мы сегментируем текст на предложения
                    #punctuation=False, 
                    tags=(True,),
                    urls=(True,),
                    numeric=(True,),
                    nonletter_sequences=(True,),
                    quotes=(True,),
                    multiple_whitespaces=(True,)
                )
    
    def __init__(self, rules=dict(), update=True):
        if update:
            for k,v in rules.items():
                if isinstance(v,(tuple,list)):
                    type(self).__dict__['rules'].update({k:v})
                elif isinstance(v,bool):
                    val = type(self).__dict__['rules'].get(k)
                    if val:
                        type(self).__dict__['rules'].update({k: (v,*val[1:])})
                    else:
                        type(self).__dict__['rules'].update({k: (v,)})      
        else:
            type(self).__dict__['rules'] = rules
        
    
    def __call__(self,text):    
        mixin = self.__class__.__bases__[0].__dict__
        
        for pipe, (val, *arg) in type(self).rules.items():
            
            if val:
                pipe = mixin.get('strip_' + pipe)  #  lambda t,*x: t
                #print(pipe)
                if type(pipe) is staticmethod:
                    pipe = pipe.__get__(RemoverMixin)
                    #print(pipe, (val, *arg))
                text = pipe(text,*arg)
                
        return text
        

class TokenFilter(RemoverMixin):
     
    allowed_tags = {
        'NN','NNS','NNP','NNPS', # существительные
        'RB','RBR','RBS', # прилагательные
        'JJ','JJR','JJS',  # наречия
        'VB','VBZ','VBP','VBD','VBN','VBG', # глаголы
        #'FW' # иностранные слова
    }

    disallowed_tags = set()
    APPDIR = os.path.abspath(os.path.dirname(__file__))
    LEXPATH = os.path.abspath(os.path.join(APPDIR,'..','lexicons'))
    LEXICON_PROPER_NAMES = Lexicon(os.path.join(LEXPATH, 'names')).load()
    LEXICON_COMMON = Lexicon(os.path.join(LEXPATH, 'lexicon')).load()
    #STOPWORDS = Lexicon(os.path.join(LEXPATH, 'stopwords')).load()
    STOPWORDS = nltk.corpus.stopwords.words('english') + [
        'mr', 'mrs',
        'mr.', 'mrs.',
        'Mr', 'Mrs', 
        'Mr.', 'Mrs.',
        'st', 'st.',
        'St', 'St.',
        'sir', 'Miss',
        'www','htm','html',
        'shall','must'
    ]
    
    rules = OrderedDict(
              punctuation=(True,),
              short=(True,3),
              stopwords=(True,STOPWORDS),
              ifnotin_lexicon=(False,[LEXICON_COMMON]),
              if_proper_name=(False,LEXICON_PROPER_NAMES),
              by_tagpos=(False,allowed_tags,disallowed_tags)
              #case=(True,)
    )
    
    
    
    def __init__(self, rules=dict(),update=True):
        
        if update:
            for k,v in rules.items():
                if isinstance(v,(tuple,list)):
                    type(self).__dict__['rules'].update({k:v})
                elif isinstance(v,bool):
                    val = type(self).__dict__['rules'].get(k)
                    if val:
                        type(self).__dict__['rules'].update({k: (v,*val[1:])})
                    else:
                        type(self).__dict__['rules'].update({k: (v,)})            
        else:
            type(self).__dict__['rules'] = rules
    
    def __call__(self,tokens):    
        mixin = self.__class__.__bases__[0].__dict__
        for pipe, (val, *arg) in type(self).rules.items():
            
            if val:
                pipe = mixin.get('remove_' + pipe)  #  lambda t,*x: t
                #print(pipe)
                if type(pipe) is staticmethod:
                    pipe = pipe.__get__(RemoverMixin)
                    #print(pipe, (val, *arg))
                tokens = pipe(tokens,*arg)
                
        return tokens



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
        self._iter = self.__paths__() 
    
    def __paths__(self):
        source = os.path.join(self.source, self.pattern)
        
        files = glob.glob(source)
        
        for filename in files:
            yield os.path.join(source, filename)
    
    def __iter__(self):
        return self._iter        
    
    def __next__(self):
        return next(self._iter)



class Stream():
    '''
    >>> lines = Stream(path)
    >>> for line in lines():
            print(line)
    '''        
    
    def __init__(self, input=None, encoding=None):
        
        self._encoding = encoding
        self._input = input
        if not isinstance(self._input,io.TextIOBase):
            self._encoding = (
                self._encoding(self._input)  
                if callable(self._encoding) 
                else self._encoding
            )  
            self._path = self._input
            self._input = open(self._input,'r',encoding=self._encoding)
        else:
            self._path = input.__class__.__name__
                
    
    def __call__(self,sentencizer=None,clean=None):
        """Read lines from filepath."""
        
        with self._input as fd:
            if sentencizer:
                try:
                    text = fd.read()
                except UnicodeDecodeError as err:
                    print(fd.encoding)
                    raise 
                if clean:
                    text = clean(text)
                yield from sentencizer(text)
            else:
                for line in fd:
                    yield line
   
        


class Corpus():
    ''''''
    
    def __init__(self, 
            inputs:List[str],
            prep:Prep, 
            clean:TextCleaner, 
            filters:TokenFilter=None, 
            datadir="data",
            verbose=False,
            autosave=True, 
            filtrate=False,
            rewrite=False
        ):
        self._inputs = inputs
        self._paths = []
        self._datadir = datadir
        #self._texts = [] 
        self._prep = prep
        self._clean = clean
        self._filters = filters
        self._vocab = Vocabulary()
        self.autosave = autosave
        self.verbose = verbose
        self.rewrite = rewrite
        self.filtrate = filtrate
        self.name = 'corpus'
        self._iter = self.__texts__()
    
    def __texts__(self):
        
        for path in self._inputs:
            self._paths.append(path)
            dtxt = datapath(path, ext=".txt").full
            ddawg = datapath(path, ext=".dawg").full
            if self.rewrite or (
                not os.path.exists(dtxt)
                ):
                prep, clean,filters = self._prep, self._clean, self._filters 
                if self.verbose:
                    print('creating: ', dtxt.replace(MODULEDIR,'..'))
            else:
                prep, clean, filters = None,None, self._filters 
                if self.verbose:
                    print('reading:  ', dtxt.replace(MODULEDIR,'..'))
                
            if self.verbose:
                start = time.time()
            
            text = Text(path, prep, clean, filters, inplace=True)
            
            if self.verbose:
                print('elapsed:  ',time.time() - start)
            
            words = text.words(filtrate=self.filtrate) # не фильтруем?
            self._vocab += Vocabulary(words)
            
            if self.verbose:
                wraptext = shorten(
                    str(words[:10]), width=80, placeholder="...]"
                )
                print('vocab:', wraptext)
                print('add vocab:',time.time() - start)
            
            if self.autosave and (self.rewrite or not os.path.exists(dtxt)): 
                text.save(format="txt")
                if self.verbose:
                    print('saving:   ', time.time() - start)
            #self._texts.append(text)
            
            yield text
    
    
    def filenames(self,n_doc=None):
        if n_doc is not None:
            res = os.path.basename(self._paths[n_doc])
        else:
            res = [os.path.basename(filename) 
                for filename in self._paths
            ]
        return res
        
    @property
    def names(self):
        return self.filenames()
    
    @property
    def ndocs(self):
        return self._vocab.ndocs
    
    @property
    def nwords(self):
        return self._vocab.nwords
    
    @property
    def nlemmas(self):
        return len(self._vocab._ccfs)
    
    @property
    def nhapaxes(self):
        return len(self._vocab.hapaxes())
    
    def hapaxes(self,n_doc=None):
         return self._vocab.hapaxes(n_doc)
    
    
    def tf(self, n_doc, token=None, sort=False, top=0):
        if token:
            return self._vocab.tf(n_doc,token)
        if top:
            res = self._top(self._vocab.tf(n_doc), top)    
        else:
            res = self._sort(self._vocab.tf(n_doc), sort) 
        return res
    
    
    def idf(self, token=None, sort=False, top=0):
        if token:
            return self._vocab.idf(token)
        if top:
            res = self._top(self._vocab.idf(), top) 
        else:
            res = self._sort(self._vocab.idf(), sort)
        return res  
    

    def tfidf(self, 
        n_doc, 
        token=None,
        texts:List[List[str]]=[], 
        sort=False,
        top=0
        ):
        if texts:
            self._vocab.compute_tfidf(
                [text for text in texts]
                
            )
        
        if token:
            return self._vocab.tfidf(n_doc,token)
        if top:
            res = self._top(self._vocab.tfidf(n_doc), top) 
        else:
            res = self._sort(self._vocab.tfidf(n_doc), sort)
        return res 
          
        
    def cfs(self, n_doc, token=None, sort=False, top=0):
        if token:
            return self._vocab.cfs(n_doc,token)
        if top:    
            res = self._top(self._vocab.cfs(n_doc), top)
        else:
            res = self._sort(self._vocab.cfs(n_doc),sort) 
        return res   
    
    
    def ccfs(self, token=None, sort=False, top=0):
        if token:
            return self._vocab.ccfs(token)
        if top:    
            res = self._top(self._vocab.ccfs(), top)
        else:
            res = self._sort(self._vocab.ccfs(), sort) 
        return res
    
    
    def dfs(self, token=None, sort=False, top=0):
        if token:
            return self._vocab.dfs(token)
        if top:    
            res = self._top(self._vocab.dfs(), top)
        else:
            res = self._sort(self._vocab.dfs(), sort)
        return res  
        
    
    def _sort(self, obj, typ):
        if typ == 1:
            res = sorted(obj.items(),key=lambda t:t[1])
        elif typ == -1:
            res = sorted(obj.items(),key=lambda t:-t[1])
        else:
            res = obj     
        return res
    
    
    def _top(self, obj, n):
        if n > 0:
            res = heapq.nlargest(n, obj, key=obj.get)   
        else:
            res = heapq.nsmallest(abs(n), obj, key=obj.get)
        return res    
            
    
    '''
    def save(self):
        path = datapath(self.name,self._datadir,".pickle").full
        pickle.dump(self, open(path,'wb'))
    '''
   
    @staticmethod
    def load(name,datadir):
        path = datapath(name, datadir, ".pickle").full
        return pickle.load(open(path,'rb'))
    
    
    def __iter__(self):
        return self._iter
    
    def __next__(self):
        return next(self._iter)
    
    #def __str__(self):
    #    return str(self.names)
    
    def __repr__(self):
        fmt = (
            "Corpus(\n\tnames={},\n\t"
            "ndocs={},\n\tnwords={},\n\t"
            "nlemmas={},\n\tnhapaxes={}\n)"
        ) 
         
        return fmt.format(
                shorten(str(self.names[:5]), width=80, placeholder="...]"),
                self.ndocs,
                self.nwords, 
                self.nlemmas,
                self.nhapaxes,
            )


class Text():
    '''
    >>> source = os.path.abspath(r"..\CORPUS\en")
    >>> sents = Text(Stream(Path(source,"*.txt")))
    >>> for sent in sents:
            print(sent)
    '''    
    
    def __init__(self, 
                    intake, 
                    prep:Prep=None, 
                    clean:TextCleaner=None, 
                    filters:TokenFilter=None,
                    inplace=False, 
                    datadir="data",
                    encoding=chardetector,
                    verbose=True,
                    input='filename' # str {'filename', 'file', 'text'}
        ):
        
        self._path = ''
        self.filename = '' 
        self.name = ''
        self._encoding = None

        if input == 'filename': 
            self._path = intake
            #self.filename = os.path.basename(os.path.splitext(self._path)[0])
            self.filename = os.path.basename(self._path)
            self.name = os.path.splitext(self.filename)[0]
            self._encoding = encoding
            self._input = intake
            
        elif input == "text":
            self._input = io.StringIO(intake)
            self._path = ''
            self.filename = self._input.__class__.__name__ 
            self.name = self.filename
            
        elif input == "file":
            self._input = intake
            self._path = ''
            self.filename = self._input.__class__.__name__ 
            self.name = self.filename
        
        self._datadir = datadir
        self._sents = []  # all sentences from the text
        #self._words = [] # all words from the text
        self._prep = prep
        self._clean = clean
        self._filters = filters
        self._vocab = FreqDist() # Counter()  # all unique normalized words from the text
        self._nwords = 0
        self.verbose = verbose
        self._iter = self.__sents__()
        
        if inplace:
            list(self._iter)  
            
 
    def __sents__(self):
        if self._prep:
            encoding=self._encoding
            sentencizer = self._prep.sentencizer
            clean = self._clean
            path = self._input
            
        else:
            path = datapath(self._path,self._datadir,".txt").full
            encoding='utf-8'
            sentencizer = None
            clean = None
            
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        
        stream = Stream(path, encoding=encoding)
        for num,sent in enumerate(stream(sentencizer,clean)):
            tagged_sent = TaggedSentence(
                    sent.strip(),
                    num,
                    self._prep,
                    self._filters
            )    
            lemmas = tagged_sent.lemmas()
            # в этот словарь попадают все леммы, так как здесь ничего не фильтруется
            self._vocab += FreqDist(lemmas) 
            self._nwords += tagged_sent.nwords
            self._sents.append(tagged_sent)
            #self._words.extend(tagged_sent.words())
            
            yield tagged_sent
            
            
            
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def nsents(self):
        return len(self._sents) 
    
    @property
    def nwords(self):
        return self._nwords
    
    @property
    def nlemmas(self):
        return len(self._vocab)
    
    def sents(self, n_sent=None, max_words=None, min_words=None):
        if n_sent is not None:
            res = self._sents[n_sent]
        else:
            if min_words is not None and max_words is not None:
                res = [sent for sent in self._sents 
                    if min_words <= sent.nwords <= max_words
                ]    
            else:
                if max_words is not None:
                    res = [sent for sent in self._sents 
                        if sent.nwords <= max_words
                    ]
                elif min_words is not None:
                    res = [sent for sent in self._sents 
                        if sent.nwords >= min_words
                    ]    
                    
                else:
                    res = self._sents
        return res
    
    def words(self, filtrate=True, lower=True, uniq=False):
        result = []
        for sent in self._sents:
            tokens = sent.tokens(filtrate=filtrate,lower=lower)
            words = [token.token for token in tokens]
            result.extend(words)
        if uniq:
            result = list(set(result))
        
        return result
    
    def lemmas(self, filtrate=True, lower=True, uniq=False):
        result = []
        for sent in self._sents:
            tokens = sent.tokens(filtrate=filtrate,lower=lower)
            words = [token.lemma for token in tokens]
            result.extend(words)
        if uniq:
            result = list(set(result))
        
        return result

    def postags(self, pos=None, universal_tagset=False, ret_cond=True):
        
        def merge(tags):
            result = FreqDist()
            for tag in tags:
                result += cfd[tag]
            return result 
        
        maps = {
            'NOUN':{'NN','NNS','NNP','NNPS'},
            'VERB':{'VB','VBD','VBG','VBN','VBP','VBZ'},
            'ADJ': {'JJ','JJR','JJS'},
            'ADV': {'RB','RBR' 'RBS'}, 
        } 
        
        cfd = ConditionalFreqDist()
        
        for sent in self._sents:
            tokens = sent.untagging() 
            for tok,tag,lemma in tokens:
                cfd[tag][lemma.lower()] += 1
        cond = cfd.conditions()
        
        result = cfd
        
        if pos:
            if not universal_tagset and pos in maps:
                result = merge(maps[pos])
            else:
                result = cfd[pos]
        if ret_cond:    
            result = result, cond   
        
        return result
    
    '''    
    def postags2(self, pos=None):
        words = []
        for sent in self._sents:
            words.extend(sent.untagging())    
        words.sort()
        tags = defaultdict(list)
        for key, group in groupby(words, lambda make: make[1]):
            tags[key].extend([l for t,p,l in group])
        
        if pos:
           return tags.get(pos)
        return tags
    '''
    
    def doc2bow(self):
        pass
    
    def ngrams(self, n,**kwargs):
        yield from nltk.ngrams(self.lemmas(), n, **kwargs)
    
    def skipgrams(self ,n, k,**kwargs):
        yield from nltk.skipgrams(self.lemmas(), n, k,**kwargs)
    
    def collocations(self):
        pass
    
    def hapaxes(self, words=False, filtrate=False):
        if not words:
            res = self._vocab
        else:
            res = FreqDist(self.words(filtrate=filtrate))
        return res.hapaxes()
  
    '''
    def save(self, path=None, format=("txt","dawg")):
        path = path or datapath(self._path).short
        
        if not isinstance(format,(tuple,list)):
            format = (format,)
        
        for fmt in format:
            if fmt == "txt":
                with open('{}.txt'.format(path),'w', encoding='utf8') as f:
                    f.writelines('\n'.join(map(str,self._sents)))
            elif fmt == 'dawg':
               data = [(str(sent), idx) 
                    for idx, sent in enumerate(self._sents)
               ]
               self.dawg = dawg.IntCompletionDAWG(data)   
               self.dawg.save('{}.dawg'.format(path))
    '''
    
    def __iter__(self):
        return self._iter
    
    def __next__(self):
        return next(self._iter)
     
    def __str__(self):
        return '\n'.join([str(sent) for sent in self.sents()])
    
    def __repr__(self):
        fmt = "Text(\n\tname='{}',\n\tnsents={},\n\tnwords={},\n\tnlemmas={}\n)"
        return fmt.format(
            self.name, 
            self.nsents, 
            self.nwords, 
            self.nlemmas
        )


class TaggedSentence():
    # \u2215 - знак деления
    # \u2044  дробная наклонная черта
    def __init__(self, sent, num, prep=None, filters=None, delim='\u2044'):
        self.raw_sent = sent    
        self._n = num
        self._delim = delim
        self._prep = prep
        self._filters = filters or (lambda s:s)
        self._nwords = 0
        if prep:
            self._sent = self.tagging(sent)
        else:
            self._sent = sent
            self._nwords = len(self.untagging())    
            
    def tagging(self, sent):
        
        tokens = [token for token in self._prep.tokenizer(sent) if token]
        lemmas = list(self._prep.lemmatizer(tokens,tagger=self._prep.tagger))
        
        threes = []
        for token,(lemma,pos) in zip(tokens,lemmas):
           threes.append(self._delim.join([token,pos,lemma]))
           self._nwords += 1    
        tagged_sent = ' '.join(threes)    
        
        return tagged_sent
        
    def untagging(self, sent=None):
        threes = []
        sent = sent or self._sent
        for  word in sent.split(' '):
            try:
                res = word.split(self._delim)
                if len(res) == 3:
                    token, pos, lemma = res
                else:
                    token, pos, lemma = res, '', ''
                threes.append((token,pos,lemma)) 
            except Exception as err:
                print(repr(sent),self._n,word)
                raise ValueError(err)
                        
        return threes
    
    
    @property
    def n(self):
        return self._n
    
    @property
    def nwords(self):
       return self._nwords
       
    
    def raw(self):
        sent = self.untagging(self._sent)
        return ' '.join(chunk[0] for chunk in sent)
    
    def words(self, lower=True, pos=None, uniq=False):
        sent = self.untagging(self._sent)
        lower = str.lower if lower else lambda s: s
        
        if isinstance(pos,set):
            res = [lower(chunk[0]) 
                    for chunk in sent if chunk[1] in pos
            ]
        elif isinstance(pos,str):
            res = [lower(chunk[0]) 
                    for chunk in sent if chunk[1].startswith(pos) 
            ]   
        else:
            res = [lower(chunk[0]) for chunk in sent]
        
        if uniq:
            res = set(res)
        
        return tuple(res)
        
    def pos(self):
        sent = self.untagging(self._sent)
        return tuple(chunk[1] for chunk in sent)
    
    def lemmas(self, lower=True, pos=None, uniq=False):
        sent = self.untagging(self._sent)
        lower = str.lower if lower else lambda s: s
        
        if isinstance(pos,set):
            res = [lower(chunk[2]) 
                    for chunk in sent if chunk[1] in pos
            ]
        elif isinstance(pos,str):
            res = [lower(chunk[2]) 
                    for chunk in sent if chunk[1].startswith(pos) 
            ]  
        else:
            res = [lower(chunk[2]) for chunk in sent]
        
        if uniq:
            res = set(res)
        
        return tuple(res)
        
    def tokens(self, filtrate=False, **kwargs):
        sent = self.untagging(self._sent)
        
        result = [Token(*chunk,self._n,idx,**kwargs) 
            for idx,chunk in enumerate(sent)
        ]
           
        if filtrate: 
            result = self._filters(result)
    
        return tuple(result)   
    
    def __str__(self):
        return self._sent

    def __repr__(self):
        fmt = "TaggedSentence(\n\t'{}',\n\t n={}\n)"
        return fmt.format(self._sent, self._n)


class Token():
    
    def __init__(self, token, pos, lemma, nsent=-1, idx=-1,**kwargs):
        
        if not kwargs.get('lower'):
            lower = lambda s:s
        else:
            lower = str.lower
        
        self.token = lower(token)
        self.pos = pos
        self.lemma = lower(lemma)
        self.nsent = nsent
        self.idx = idx
        
        
    def __str__(self):
        return self.lemma
        
    def __repr__(self):
        fmt = (
            "Token(token='{token}', idx={idx}, "
            "pos='{pos}', lemma='{lemma}', nsent={nsent})"
        )
        return fmt.format(
            token=self.token,
            idx=self.idx,
            pos=self.pos,
            lemma=self.lemma,
            nsent=self.nsent
        )
    


#------------------------------------------------------------------------------#
# tests
#------------------------------------------------------------------------------#
if __name__ == "__main__":
    
    source = os.path.abspath(os.path.join(nlptk.MODULEDIR,r'corpus\en'))
    APPDIR = os.path.abspath(os.path.dirname(__file__))
    
    intake = '''
    Squire Trelawney, Doctor Livesey, and the rest of these gentlemen having
asked me to write down the whole particulars about Treasure Island, from
the beginning to the end, keeping nothing back but the bearings of the
island, and that only because there is still treasure not yet lifted, I
take up my pen in the year of grace 17--, and go back to the time when
my father kept the "Admiral Benbow" Inn, and the brown old seaman, with
the saber cut, first took up his lodging under our roof.'''
    
    
    prep = Prep()
    rules_clean=OrderedDict(
        roman_numerals=(True,)
    )
    
    rules_filter=OrderedDict(
        by_tagpos=(True,{},{"FW","POS"}), #игнорировать иностранные слова и possessive ending parent'
        short=(True,3),
        ifnotin_lexicon=True
        #trailing_chars=True
    )
    clean = TextCleaner(rules_clean)
    filters = TokenFilter(rules_filter)
    #pprint(filters.rules)
    #pprint(TokenFilter.rules)
    
    #text = clean('It is nothing," he said, 1 IV  <a href=www.google.com </> at Last bla-bla-bla.')
    #print(text)
    #print(filters(text.split()))
    
    #intake = os.path.join(nlptk.MODULEDIR,r'corpus\test\1.txt')
    #inp = 'filename'
    #intake = 'hello world'
    inp='text'
    #intake = sys.stdin
    #inp = 'file'
    text = Text(intake, prep=prep, clean=clean, filters=filters, input=inp)
    print(list(text))
    print(text)
    '''
    Squire⁄NNP⁄Squire Trelawney⁄NNP⁄Trelawney Doctor⁄NNP⁄Doctor Livesey⁄NNP⁄Livesey 
    and⁄CC⁄and the⁄DT⁄the rest⁄NN⁄rest of⁄IN⁄of these⁄DT⁄these gentlemen⁄NNS⁄gentleman 
    having⁄VBG⁄have asked⁄VBD⁄ask me⁄PRP⁄me to⁄TO⁄to write⁄VB⁄write down⁄RP⁄down 
    the⁄DT⁄the whole⁄JJ⁄whole particulars⁄NNS⁄particular about⁄IN⁄about Treasure⁄NNP⁄Treasure 
    Island⁄NNP⁄Island from⁄IN⁄from the⁄DT⁄the beginning⁄NN⁄beginning to⁄TO
    ⁄to the⁄DT⁄the end⁄NN⁄end keeping⁄VBG⁄keep nothing⁄NN⁄nothing back⁄RB⁄back 
    but⁄CC⁄but the⁄DT⁄the bearings⁄NNS⁄bearing of⁄IN⁄of the⁄DT⁄the island⁄NN⁄island 
    and⁄CC⁄and that⁄IN⁄that only⁄RB⁄only because⁄IN⁄because there⁄EX⁄there is⁄VBZ⁄be 
    still⁄RB⁄still treasure⁄JJ⁄treasure not⁄RB⁄not yet⁄RB⁄yet lifted⁄VBN⁄lift I⁄PRP⁄I 
    take⁄VBP⁄take up⁄RP⁄up my⁄PRP$⁄my pen⁄NN⁄pen in⁄IN⁄in the⁄DT⁄the year⁄NN⁄year of⁄I
    N⁄of grace⁄NN⁄grace and⁄CC⁄and go⁄VB⁄go back⁄RB⁄back to⁄TO⁄to the⁄DT⁄the 
    time⁄NN⁄time when⁄WRB⁄when my⁄PRP$⁄my father⁄NN⁄father kept⁄VBD⁄keep the⁄DT⁄the 
    Admiral⁄NNP⁄Admiral Benbow⁄NNP⁄Benbow Inn⁄NNP⁄Inn and⁄CC⁄and the⁄DT⁄the brown⁄JJ⁄brown 
    old⁄JJ⁄old seaman⁄NN⁄seaman with⁄IN⁄with the⁄DT⁄the saber⁄NN⁄saber cut⁄NN⁄cut 
    first⁄RB⁄first took⁄VBD⁄take up⁄RP⁄up his⁄PRP$⁄his lodging⁄VBG⁄lodge under⁄IN⁄unde
    r our⁄PRP$⁄our roof⁄NN⁄roof
    '''
    print(repr(text))
    '''
    Text(
            name='StringIO',
            nsents=1,
            nwords=89,
            nlemmas=65
    )
    '''
