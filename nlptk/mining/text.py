import os, sys
import io, glob
import dawg
import pickle
import math
import time
import nltk
from nltk import FreqDist, ConditionalFreqDist

from typing import List
from collections import Counter,defaultdict
from pprint import pprint,pformat
from functools import partial
from itertools import chain
from collections import OrderedDict
from itertools import islice

from docx import Document


from nlptk.mining.utils import chardetector, datapath
from nlptk.mining.vocab import Vocabulary
from nlptk.measures.metrics import MeasuresAssociation
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


class Collocate():
    
    method = {
            'mi':MeasuresAssociation.mi,
            'student':MeasuresAssociation.t_score,
            't-score':MeasuresAssociation.t_score,
            'z_score':MeasuresAssociation.z_score,
            'dice':MeasuresAssociation.dice,
            'logdice':MeasuresAssociation.log_dice,
            'log_dice':MeasuresAssociation.log_dice,
            'mi3':MeasuresAssociation.mi3,
            'salience':MeasuresAssociation.salience,
            'min_sens':MeasuresAssociation.min_sens
            #'likelihood':MeasuresAssociation.likelihood
    }
    
    def __init__(self):
        pass
    
    def collocations(self,tokens,bigrams,measure='student'):
        # словарь методов реализующих меры ассоциаций
        
        
        self.dict_bigrams = Counter(bigrams)           # частотный словарь биграмм
        dict_freq = self.freq(tokens)                  # частотный словарь (Counter) всех слов в выборке
        total = len(tokens)                            # общее число всех слов в выборке
        collocat_dict = dict()                         # словарь биграмм\коллокаций, где каждому ключу присваивается значение меры ассоциации в зависимости от метода
        bigrams = set(bigrams)                         # уникализируем список биграмм
        
        for bigram in bigrams:
            n_req  = dict_freq.get(bigram[0],0)        # частота ключевого слово с биграмме
            c_freq = dict_freq.get(bigram[1],0)        # частота коллоката в биграмме
            cn_freq = self.dict_bigrams.get(bigram)    # частота самой биграммы
            collocat_dict[bigram] = type(self).method[measure](cn_freq,n_req,c_freq,total)
        
        return Counter(collocat_dict)



        

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
  
    def __paths__(self):
        source = os.path.join(self.source, self.pattern)
        
        files = glob.glob(source)
        
        for filename in files:
            yield os.path.join(source, filename)
    
    def __iter__(self):
        return  self.__paths__()            
    
    def __next__(self):
        return next(self.__paths__())

class Stream():
    '''
    >>> lines = Stream(path)
    >>> for line in lines():
            print(line)
    '''        
    
    def __init__(self, path=None, encoding=None):
        
        self._encoding = encoding
        self._input = path
        
        if not isinstance(self._input,io.TextIOBase):
            self._encoding = (
                self._encoding(self._input)  
                if callable(self._encoding) 
                else self._encoding
            )  
            self._path = self._input
            self._input = open(self._input,'r',encoding=self._encoding)
        else:
            self._path = self._input.name
                
    
    def _liner(self,sentencizer,clean):
        
        with self._input as fd:
            if sentencizer:
                try:
                    text = fd.read()
                except UnicodeDecodeError as err:
                    print(fd.encoding)
                    raise 
                if clean:
                    text = clean(text)
                return sentencizer(text)
            else:
                return fd
    
    def __call__(self,sentencizer=None,clean=None):
        """Read lines from filepath."""
        
        for line  in self._liner(sentencizer,clean):
            yield line
        


class Corpus():
    ''''''
    
    def __init__(self, 
            inputs,
            prep, clean, filters=None, 
            save=True, verbose=False
        ):
        self._inputs = inputs
        #self._texts = [] 
        self._prep = prep
        self._clean = clean
        self._filters = filters
        self._vocab = Vocabulary()
        self._save = save
        self._id2words = []
        self.verbose = verbose
        
    def __texts__(self):
        
        for path in self._inputs:
            
            dtxt = datapath(path, ext=".txt").full
            ddawg = datapath(path, ext=".dawg").full
            if (
                not os.path.exists(dtxt)
                ):
                prep, clean,filters = self._prep, self._clean, self._filters 
                if self.verbose:
                    print('Creating: ',dtxt)
            else:
                prep, clean, filters = None,None, self._filters 
                if self.verbose:
                    print('Reading:  ',dtxt)
                
            if self.verbose:
                start = time.time()
            
            text = Text(path, prep, clean, filters, inplace=True)
            
            if self.verbose:
                print('elapsed:  ',time.time() - start)
            
            if self._save and not os.path.exists(dtxt): 
               text.save(format="txt")
               print('save:     ', time.time() - start)
            #self._texts.append(text)
            self._vocab += Vocabulary(text.words())
            
            if self.verbose:
                print('add vocab:',time.time() - start)
            
            yield text
    
    @property
    def idf(self, term=None):
        if term:
            return self._vocab._idf.get(term)
        return self._vocab.idf
    
    @property
    def tfidf(self, texts:List[List[str]]=[], n_doc=None,term=None):
        if not self._vocab.tfidf:
            self._vocab.compute_tfidf(
                [text for text in texts]
            )
        
        if n_doc:
            if term:
                return self._vocab._tfidf[n_doc].get(term)
            return self._vocab._tfidf[n_doc]    
        return self._vocab.tfidf
    
    @property
    def cfs(self, n_doc=None, term=None):
        if n_doc:
            if term:
                return self._vocab._cfs[n_doc].get(term)
            return self._vocab._cfs[n_doc]
        return self._vocab.cfs   
    
    '''
    @property
    def text_vocab(self, n_doc, term=None):
        if term:
            return self._texts[n_doc]._vocab.get(term)
        return self._texts[n_doc]._vocab
    '''
    
    @property
    def ccfs(self, term=None):
        if term:
            return self._vocab._ccfs.get(term)
        return self._vocab.ccfs 
    
    @property
    def dfs(self, term=None):
        if term:
            return self._vocab._dfs.get(term)
        return self._vocab.dfs   
    

    def __iter__(self):
        return self.__texts__()
    
    def __next__(self):
        return next(self.__texts__())
    
    def __str__(self):
        pass
    
    def __repr__(self):
        fmt = "Corpus(nwords={}, nlemmas={})".format()


class Text():
    '''
    >>> source = os.path.abspath(r"..\CORPUS\en")
    >>> sents = Text(Stream(Path(source,"*.txt")))
    >>> for sent in sents:
            print(sent)
    '''    
    
    def __init__(self, path, 
                    prep=None, 
                    clean=None, 
                    filters=None,
                    inplace=False, 
                    datadir="data",
                    encoding=chardetector,
                    verbose=True
        ):
        self._path = path
        self._encoding = encoding
        self._datadir = datadir
        self._sents = []  # all sentences from the text
        #self._words = [] # all words from the text
        self._prep = prep
        self._clean = clean
        self._filters = filters
        self._vocab = FreqDist() # Counter()  # all unique normalized words from the text
        self.filename = os.path.basename(os.path.splitext(self._path)[0])
        self.verbose = verbose
        
        if inplace:
            list(self.__sents__())  
            
 
    def __sents__(self):
        if self._prep:
            stream = Stream(self._path,encoding=self._encoding)
            for num,sent in enumerate(
                    stream(self._prep.sentencizer, self._clean)
                ):
                
                tagged_sent = TaggedSentence(sent,num,self._prep,self._filters)    
                lemmas = tagged_sent.lemmas()
                # в этот словарь попадают все леммы, так как здесь ничего не фильтруется
                self._vocab += FreqDist(lemmas) 
                self._sents.append(tagged_sent)
                #self._words.extend(tagged_sent.words())
                
                yield tagged_sent
        else:
            path = datapath(self._path,self._datadir,".txt").full
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            
            yield from self.load()
    
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def sents(self):
        return self._sents
    
    @property
    def nsents(self):
        return len(self._sents) 
    
    @property
    def nwords(self):
        return len(self.words(filtrate=False))
    
    @property
    def nlemmas(self):
        return len(self._vocab)
    
    
    def words(self, filtrate=True, lower=True):
        result = []
        for sent in self._sents:
            tokens = sent.tokens(filtrate=filtrate,lower=lower)
            words = [token.token for token in tokens]
            result.extend(words)
        return result
    
    def lemmas(self, filtrate=True, lowre=True):
        result = []
        for sent in self._sents:
            tokens = sent.tokens(filtrate=filtrate,lower=lower)
            words = [token.lemma for token in tokens]
            result.extend(words)
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
            tokens = sent.untagged() 
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
        
    def postags2(self, pos=None):
        words = []
        for sent in self._sents:
            words.extend(sent.untagged())    
        words.sort()
        tags = defaultdict(list)
        for key, group in groupby(words, lambda make: make[1]):
            tags[key].extend([l for t,p,l in group])
        
        if pos:
           return tags.get(pos)
        return tags
 
    def doc2bow(self):
        pass
    
    def ngrams(self, n,**kwargs):
        yield from nltk.ngrams(self.lemmas(), n, **kwargs)
    
    def skipgrams(self ,n, k,**kwargs):
        yield from nltk.skipgrams(self.lemmas(), n, k,**kwargs)
    
    def collocations(self):
        pass
    
    def hapaxes(self,n=-1):
        return self._vocab.hapaxes()[:n]
  
    
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
    
    
    def load(self, path=None, format=("txt","dawg")):
        
        path = path or datapath(self._path).short
        if not isinstance(format,(tuple,list)):
            format = (format,)
        
        for fmt in format:
            if fmt == "txt":
                fullpath = '{}.txt'.format(path)
                with open(fullpath,'r', encoding='utf8') as fd:
                    
                    for n,line in enumerate(fd):
                        tagged_sent = TaggedSentence(
                                            line.strip(),n,
                                            filters=self._filters
                                )
                        lemmas = tagged_sent.lemmas()
                        self._vocab += Counter(lemmas)
                        self._sents.append(tagged_sent)
                        yield tagged_sent
            
            if fmt =='dawg':
                if not os.path.exists('{}.dawg'.format(path)):
                    return
                
                self.dawg = dawg.IntCompletionDAWG()   
                self.dawg.load('{}.dawg'.format(path))   
               
                if not self._sents:
                    sents = sorted(self.dawg.items(),key=lambda t:t[1])
                    tagged_sent = TaggedSentence(
                                        line.strip(),n,
                                        filters=self._filters
                                    )
                    lemmas = tagged_sent.lemmas()
                    self._vocab += Counter(lemmas)
                    self._sents.append(tagged_sent)
                    yield tagged_sent

    
    def __iter__(self):
        return self.__sents__()
    
    def __next__(self):
        return next(self.__sents__())
     
    def __str__(self):
        pass
    
    def __repr__(self):
        fmt = "Text(name={}, nsents={}, nwords={}, nlemmas={})"
        return fmt.format(
            self.filename, self.nsents, self.nwords, self.nlemmas
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
        if prep:
            self._sent = self.tagged(sent)
        else:
            self._sent = sent
    
    def tagged(self, sent):
        tokens = [token for token in self._prep.tokenizer(sent) if token]
        lemmas = list(self._prep.lemmatizer(tokens))
        
        threes = []
        for token,(lemma,pos) in zip(tokens,lemmas):
           threes.append(self._delim.join([token,pos,lemma]))    
        tagged_sent = ' '.join(threes)    
        
        return tagged_sent
        
    def untagged(self, sent=None):
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
    
    def raw(self):
        sent = self.untagged(self._sent)
        return ' '.join(chunk[0] for chunk in sent)
    
    def words(self):
        sent = self.untagged(self._sent)
        return tuple(chunk[0].lower() for chunk in sent)
        
    def pos(self):
        sent = self.untagged(self._sent)
        return tuple(chunk[1] for chunk in sent)
    
    def lemmas(self):
        sent = self.untagged(self._sent)
        return tuple(chunk[2].lower() for chunk in sent)
    
    def tokens(self, filtrate=True, **kwargs):
        sent = self.untagged(self._sent)
        
        result = [Token(*chunk,self._n,idx,**kwargs) 
            for idx,chunk in enumerate(sent)
        ]
           
        if filtrate: 
            result = self._filters(result)
    
        return tuple(result)   
    
    def __str__(self):
        return self._sent

    def __repr__(self):
        fmt = "TaggedSentence('{}', n={})"
        return fmt.format(self._sent.replace(' ','\n'), self._n)


class Token():
    
    def __init__(self, token, pos, lemma, nsent=-1, idx=-1,**kwargs):
        
        self.token = token
        self.pos = pos
        self.lemma = lemma
        self.nsent = nsent
        self.idx = idx
        if not kwargs.get('lower'):
            self._lower = lambda s:s 
        else:
            self._lower = str.lower
    
    def __str__(self):
        return self.lemma
        
    def __repr__(self):
        fmt = (
            "Token(token='{token}', idx={idx}, "
            "pos='{pos}', lemma='{lemma}', nsent={nsent})"
        )
        return fmt.format(
            token=self._lower(self.token),
            idx=self.idx,
            pos=self.pos,
            lemma=self._lower(self.lemma),
            nsent=self.nsent
        )
    

class Prep(TokenizerMixin,
            LemmatizerMixin,
            SentencizerMixin,
            TaggerMixin
            ):
        
    
    def __init__(self, *args, **kwargs):
        
        defaults = {
            'sentencizer':self.sentencize_nltk, 
            'tokenizer':self.treebank_word_tokenize,
            'tagger':self.tagger_nltk, # self.tagger_4ngram
            'lemmatizer':partial(self.lemmatize_nltk,pos=True) #tagset='universal'
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
        'Mr', 'Mrs', 
        'st','sir', 'Miss',
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




if __name__ == "__main__":
    APPDIR = os.path.abspath(os.path.dirname(__file__))
    text = '''
    "It is farther on," said I; "but observe the white web-work which gleams from these cavern walls."

He turned towards me, and looked into my eyes with two filmy orbs that distilled the rheum of intoxication. "Nitre?" he asked, at length.

"Nitre," I replied. "How long have you had that cough?" "Ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!--ugh! ugh! ugh!"

My poor friend found it impossible to reply for many minutes.

"It is nothing," he said, at last.'''
    
    
    
    APPDIR =  os.path.dirname(__file__)
    source = os.path.abspath(os.path.join(APPDIR,'..',r'CORPUS\en'))
 
    prep = Prep()
    rules_clean=OrderedDict(
        roman_numerals=(True,)
    )
    
    rules_filter=OrderedDict(
        by_tagpos=(True,{},{"FW"}),
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
    
    
    #for path in Path(source,"*.txt"):
    start_line = time.time()
    corpus = Corpus(Path(source,"*.txt"), prep, clean, filters)
    corpus.verbose = True
    for text in corpus:    
       
        for i,sent in enumerate(text):
            #print(sent)
            pass      
            
            '''
            print(sent.tokens())
            print(sent.words())
            print(sent.lemmas())
            print(sent.raw())
            '''
            '''
            tokens = filters(sent.tokens())
            if tokens:
                pprint(repr(tokens))
            '''    
        '''
        print(repr(text))
        
        words = text.words(filtrate=False)
        print(len(words))
        print(words[:10])
        words = text.words()
        print(len(words))
        print(words[:10])
        print(len(text._vocab))
        print(text._vocab.most_common(10))
        print(text.hapaxes())
        cfd, cond = text.postags()
        verb = cfd['VB']
        print(verb)
        print(verb.most_common(10))
        noun = cfd['NN']
        print(noun)
        print(noun.most_common(10))
        noun, _ = text.postags('NOUN')
        print(noun)
        print(noun.most_common(10))
        verb, _ = text.postags('VERB')
        print(verb)
        print(verb.most_common(10))
        print(cond)
        
        #pprint(list(islice(text.ngrams(3),10)))
        #pprint(list(islice(text.skipgrams(3,2),10)))
        input(">>")
        '''
      
    
    '''        
        
    #input("--tfidf--")    
    #pprint(corpus.tfidf[0])
    input("--cfc--")
    pprint(corpus.cfs[0])
    input("--ccfc--")
    pprint(corpus.ccfs)
    input("--dfc--")
    pprint(corpus.dfs)
    
    '''
    
    '''
    He/PPS/He prided/VBD/prided himself/PPL/himself on/IN/on his/PP$/his connoisseurship/NN/connoisseurship in/IN/in wine/NN/wine
    ("[Token(token='prided', idx=1, pos='VBD', lemma='prided', nsent=8), "
     "Token(token='connoisseurship', idx=5, pos='NN', lemma='connoisseurship', "
     "nsent=8), Token(token='wine', idx=7, pos='NN', lemma='wine', nsent=8)]")
    
    Few/AP/Few Italians/NPS/Italians have/HV/have the/AT/the true/JJ/true virtuoso/NN/virtuoso spirit/NN/spirit
    ("[Token(token='Italians', idx=1, pos='NPS', lemma='Italians', nsent=9), "
     "Token(token='true', idx=4, pos='JJ', lemma='true', nsent=9), "
     "Token(token='virtuoso', idx=5, pos='NN', lemma='virtuoso', nsent=9), "
     "Token(token='spirit', idx=6, pos='NN', lemma='spirit', nsent=9)]")
    
    
    ['NNPS', 'EX', 'WP', 'VBN', 'MD', 'VBP', 'VBD', 'WP$', 'JJS', 'JJR', 'PRP',
    'PRP$', 'NNS', 'VBG', 'TO', 'VB', 'UH', 'FW', 'JJ', 'CC', 'RP', 'POS', 
    'PDT', 'CD', 'WDT', 'WRB', 'NN', 'VBZ', 'RBS', 'IN', 'DT', 'RB', 'RBR', 'NNP'
    ]
    '''
