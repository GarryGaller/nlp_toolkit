import os,sys,io,glob
import dawg
import pickle
from docx import Document
from chardet.universaldetector import UniversalDetector
from collections import Counter
from pprint import pprint,pformat
from collections import defaultdict
from functools import partial

from nlptk.patterns.patterns import RE_WHITESPACE, RE_PUNCT2



def chardetector(filepath):
    default_encoding = sys.getfilesystemencoding()
    #default_enc = locale.getpreferredencoding()
    result = None
    detector = UniversalDetector()
    detector.reset()
    
    for line in open(filepath, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()
    encoding = detector.result.get('encoding') or default_encoding
    
    return encoding


def pad_sequences(sequences, 
    maxlen,
    padding='pre',
    truncating='pre', 
    value=0.0
    ):
    
    sequences = sequences.copy()
    
    for seq in sequences:
        if not len(seq):
            continue 
        
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq[:] = seq[-maxlen:]
            elif truncating == 'post':
                seq[:] = seq[:maxlen]
            else:
                raise ValueError('Truncating type "%s" '
                                 'not understood' % truncating)
        else:
            pad_len = maxlen - len(seq) 
            if padding == 'post':
                seq[len(seq):] =  [value] * pad_len 
            elif padding == 'pre': 
                seq[:] = [value] * pad_len  + seq[:]
            else:
                raise ValueError('Padding type "%s" '
                                'not understood' % padding)
            
    return sequences 


class TaggedSentence():
    pass




class Dictionary():
    
    def __init__(self,data):
        data = set(data)
        self.vocab = defaultdict()
        self.vocab.default_factory = self.vocab.__len__
        for key in data:
            self.vocab[key]

    def text2seq(self,tokens):
        return [self.vocab[token] for token in tokens]   
    
    def text2onehot(self, tokens, dim=1000):
        #onehot = [0] * dim
        onehot = np.zeros((len(dim),),dtype=np.float)
        for i,token in enumerate(tokens):
            if token in self.vocab:
                onehot[i] = 1 
        return onehot 


    def freq(self,vocab):    
        '''отсортированный по убыванию частотности список слов'''
        return sorted(self.vocab,key=self.vocab.get,reverse=True)
       
    def idx2tok(self):
        return {idx:tok for tok,idx in self.vocab.items()}

    def hapaxes(self):
        '''Возвращает список слов имеющих только одно вхождение'''    
        return [t[0] for t in self.vocab if t[1] == 1]



class Lexicon():
    def __init__(self,name,words=[]):
        self.name = name
        self.words = words
        
    def save(self):
        if self.words:
            c_dawg = dawg.CompletionDAWG(self.words)
            c_dawg.save(r'{}.dawg'.format(self.name))
        
    def load(self):
        c_dawg = dawg.CompletionDAWG()
        return c_dawg.load(r'{}.dawg'.format(self.name)) 



class Stream():
    
    def __init__(self, 
            inputs=None,
            encoding=None,
            sentencizer=None,
            text_filters=None,
            **kwargs
            ):
        
        self.__inputs = inputs  
        self.__encoding = encoding
        if not isinstance(self.__inputs,io.TextIOBase):
            self.__encoding = (
                chardetector(self.__inputs) 
                    if self.__encoding == 'chardetect' else self.__encoding
            )  
            self.__filepath = self.__inputs
            self.__inputs = open(self.__inputs,'r',encoding=self.__encoding)
            self.__ext = os.path.splitext(self.__filepath )[-1]
        else:
            self.__ext = ''
            
        self.__sentencizer = sentencizer
        self.__text_filters = text_filters or []
        self.__endl = ''
        self.__iter = self.getstream()
    
    
    def preprocess_text(self,text):
        for text_filter in self.__text_filters:
            text = text_filter(text)
        return text
    
    
    def getstream(self):
        """Read lines from filepath."""
        
        if self.__ext in ['.doc','.docx']:
            doc = Document(self.__filepath)
            self.__endl = '\n'
            for line in doc.paragraphs:
                yield line.text + self.__endl
        else:
          
            with  self.__inputs as fd:
                # обрабатываем либо по предложению    
                if self.__sentencizer:
                    text = self.preprocess_text(fd.read())
                    for sentence in self.__sentencizer(text):
                        yield sentence + self.__endl
                # либо по строке
                else:  
                    for line in fd:
                        yield line + self.__endl
    
    
    
    def __iter__(self):
        return self.__iter
    
    def __next__(self):
       return next(self._iter)
            
        


class BaseCorpus():
    '''
    sourcedir каталог-источник документов
    wildcard фильтр файлов из директории-источника
    encoding кодировка для текстовых файлов; для документов .docx не требуется
    preprocess_class - класс препроцессинга, который просто передается 
    экземплярам класса Text
    '''
    
    def __init__(self, cfg, preprocess_class,**kwargs):
        
        self.sourcedir = cfg.get('SOURCEDIR','.')
        self.wildcard = cfg.get('WILDCARD',"*.txt")     
        self.encoding = cfg.get('ENCODING','chardetect')  
        
        self.preprocess_class = preprocess_class
        self.__dict__.update(kwargs)
        self._iter = self.__get_texts__() 

    def __getpaths__(self):
        source = os.path.join(self.sourcedir,self.wildcard)
        
        files = glob.glob(source)
        for filename in files:
            yield os.path.join(source,filename)
    
    
    def __get_texts__(self):
        
        for path in  self.__getpaths__():
            yield Text(path,self.preprocess_class,**kwargs)
        
    def __iter__(self):
        return self._iter
    
    def __next__(self):
        return next(self._iter)  

    
    def __str__(self):
        pass
    
    def __repr__(self):
        pass


class PlainCorpus(BaseCorpus):
    
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        

class Text():
    '''Text class'''
    
    def __init__(self,inputs,preprocess_class,**kwargs):
        
        self.inputs = inputs
        self.encoding = kwargs.get('encoding','chardetect') 
        self.pickledir = kwargs.get('pickledir')
        
        if not isinstance(self.inputs,io.TextIOBase):
            self.filepath = self.inputs
            self.name = os.path.basename(self.filepath)
            self.encoding = (
                chardetector(self.filepath) 
                    if self.encoding == 'chardetect' else self.encoding
            )  
        else:
            self.filepath = None
            self.name = inputs.__class__.__name__
        
        
        self.lines = []
        self.nlines = 0
        self.ntokens = 0
        self.ntokens_ = 0
        self.nlemmas = 0
        self.vocab = Counter()
        self.vocab_ = Counter()
        self._iter = self.__getlines__()
        self._inplace = kwargs.pop("inplace",False)
        
        self.sentencizer = None
        self.tokenizer = None
        self.lemmatizer = None
        self.text_filters = []
        self.char_filters = []
        self.token_filters = [] 
        self.token_rules = {}
        self.lemma_filters = []
        self.lexicon_filters = []
        
        self.__dict__.update(preprocess_class.preprocess_funcs)
        
        if self._inplace:
            list(self._iter)    
    
    def __getlines__(self):
        self.streamlines = Stream(
            self.inputs,
            encoding=self.encoding,
            sentencizer=self.sentencizer,
            text_filters=self.text_filters
            )
        
        
        for lineno,line in enumerate(self.streamlines):
            line = line.strip()
            raw_line = RE_WHITESPACE.sub(" ", line).strip() # self._corpus.strip_multiple_whitespaces(line) 
            #raw_line = self.strip_quotes(line)
            line = self.preprocess_line(line)
            
            if line in ('\r\n','\r','\n',''):continue
            
            class_ = Sentence if self.sentencizer else Line
            line = class_(
                    raw_line,
                    line,
                    lineno,
                    tokenizer=self.tokenizer,
                    lemmatizer=self.lemmatizer,
                    token_filters=self.token_filters,
                    token_rules=self.token_rules,
                    lemma_filters=self.lemma_filters,
                    lexicon_filters=self.lexicon_filters
                )
            self.lines.append(line)
            self.nlines   += 1
            self.ntokens  += line.ntokens
            self.ntokens_ += line.ntokens_
            self.vocab_ += Counter([tok for tok in line.tokens_])
            self.vocab  += Counter([tok.lemma for tok in line.tokens])
            self.nlemmas = len(self.vocab)
            yield line
            
    
    
    def preprocess_line(self,text,lineno=None):
        '''Предобработка текста указанными фильтрами'''
        
        # обработка  текста специфическим стриппером
        for char_filter in self.char_filters:
            text = char_filter(text)
        
        return text
    
    def sents(self,raw_sents=False,raw_tokens=False,lemmas=True):
        if raw_sents:
            lines = [line.line for line in self.lines]   
        elif raw_tokens:
            lines = [line.tokens_ for line in self.lines]   
        elif lemmas:
            lines = [[token.lemma for token in line.tokens] 
                        for line in self.lines
            ] 
        else:
            lines = self.lines
        
        return lines
    
    def raw_sents(self):
        return [line.tokens_ for line in self.lines] 
    
    def __iter__(self):
        return self._iter
    
    def __next__(self):
        return next(self._iter)    
    
    def __str__(self):
        return str([line for line in self.lines])   
    
    def __repr__(self):
        textfmt = ( 
            "Text(name={}, nlines={}, "
            "ntokens_={}, ntokens={}, "
            "nlemmas={}, lines=\n{})"
        )
        return textfmt.format(
            self.name,
            self.nlines,
            self.ntokens_,
            self.ntokens,
            self.nlemmas,
            pformat([line for line in self.lines],indent=2)
        )
    
    
    def prepare(self,**preprocess_funcs):
        self.__dict__.update(preprocess_funcs)
    
   
    def pickled(self,pickledir=None):
        
        self.pickledir = pickledir or self.pickledir
        if self.pickledir is None: 
            raise Exception("pickledir parameter not passed")
        
        filename = os.path.splitext(
            os.path.basename(self.filepath)
            )[0] + '.vcb'
        
        filepickle = os.path.realpath(
            os.path.join(self.pickledir,filename)
        )
        
        return  os.path.exists(filepickle)
    
    
    def save(self,pickledir=None):
        
        self.pickledir = pickledir or self.pickledir
        if self.pickledir is None: 
            raise Exception("pickledir parameter not passed")
        
        filename = os.path.splitext(
            os.path.basename(self.filepath)
            )[0] + '.vcb'
        
        filepath = os.path.realpath(
            os.path.join(self.pickledir,filename)
        )
        with open(filepath,'wb') as fd:
            obj = (
                self.lines,
                self.vocab,
                self.vocab_,
                self.nlines,
                self.ntokens_,
                self.ntokens,
                self.nlemmas
            )
            pickle.dump(obj, fd)
    
       
    def load(self,pickledir=None):
        
        self.pickledir = pickledir or self.pickledir
        if self.pickledir is None: 
            raise Exception("pickledir parameter not passed")
        
        filename = os.path.splitext(
            os.path.basename(self.filepath)
            )[0] + '.vcb'
        
        filepath = os.path.realpath(
            os.path.join(self.pickledir,filename)
        )
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        
        with open(filepath,'rb') as fd:
            obj = pickle.load(fd)   
            (
                self.lines,
                self.vocab,
                self.vocab_,
                self.nlines,
                self.ntokens_,
                self.ntokens,
                self.nlemmas
            ) = obj
    
    
    def info(self):
        textfmt = (
            "Text(name={}, encoding={}, "
            "nlines={}, ntokens_={}, "
            "ntokens={}, nlemmas={})"
        )
        return textfmt.format(
            self.name,
            self.encoding,
            self.nlines,
            self.ntokens_,
            self.ntokens,
            self.nlemmas
            )
    
    
class Line(Text):
    '''Line class'''
   
    def __init__(self, raw_line, line, lineno, 
        tokenizer,
        lemmatizer=None,
        token_filters=None,
        lemma_filters=None,
        token_rules=None,
        lexicon_filters=None
        ):
        super()
        
        self.line = raw_line
        self.lineno = lineno
        self.ntokens = 0
        self.ntokens_ = 0
        self.__tokenizer = tokenizer 
        self.__lemmatizer = lemmatizer
        self.__token_filters = token_filters or []
        self.__lemma_filters = lemma_filters or []
        self.__lexicon_filters = lexicon_filters or []
        self.__token_rules = token_rules or {}
        self.tokens = self.__tokenize__(line)
        
        
    def __tokenize__(self,line):
        
        if self.__tokenizer is None: 
            self.__tokenizer = str.split   #RE_PUNCT2.split
            #raise NotImplementedError('tokenizer is not implemented')
        
        tokens = []
        self.tokens_ = list(self.__tokenizer(line))
        
        dict_tokens = defaultdict(list)
        for idx,token in enumerate(self.tokens_):
            dict_tokens[token].append(idx)
        
        self.ntokens_ = len(self.tokens_)
        for token in self.preprocess_tokens(self.tokens_):
            
            token = Token(
                token,
                dict_tokens[token], 
                self.lineno, 
                self.__lemmatizer,
                self.__token_rules
                )
            # если леммы нет в лексиконе - помечаем ее <>
            for lexf in self.__lexicon_filters:
                if not lexf(token):
                    token.lemma = "<{}>".format(token.lemma)
                    #print(repr(token))    
                   
            for lf in self.__lemma_filters:
                if not lf(token):
                    break
            else:
                tokens.append(token)
        
        self.ntokens = len(tokens)
        return tokens
    
    def preprocess_tokens(self,tokens):
        '''Предобработка токенов указанными фильтрами'''
        
        for token_filter in self.__token_filters:
            tokens = token_filter(tokens)
        
        return tokens
     
    def __str__(self):
        return str([token.lemma for token in self.tokens])
     
    def __repr__(self):
        textfmt = (
            "{}(lineno={},ntokens_={}, "
            "ntokens={}, line='{}', "
            "tokens_={}, tokens=\n{})"
        )
        return textfmt.format(
            self.__class__.__name__,
            self.lineno,
            self.ntokens_,
            self.ntokens,
            self.line,
            self.tokens_,
            pformat([token for token in self.tokens],indent=4)
        )
        

    
class Sentence(Line):
    '''Sentence'''
    pass


class Token(Text):
    '''Token class'''
    
    def __init__(self,token,indexes,lineno,
        lemmatizer=None,
        token_rules=None,
        ):
        super()
        self.token = token
        self.indexes = indexes
        self.lineno = lineno
        self.__lemmatizer = lemmatizer
        self.lemma,self.pos = None,None
        result = self.__lemmatize__(token)
        
        if result:
            
            if isinstance(result[0],tuple):
                self.lemma = result[0][0]                            
                self.pos = result[0][1]
            else:
                self.lemma = result[0]
            self.__make_case(token_rules)
    
    def __make_case(self,token_rules):
        if token_rules:
            if token_rules.get('make_token_lower',True):
                self.token = self.token.lower()
            elif token_rules.get('make_lemma_lower',True):
                self.lemma = self.lemma.lower()
    
    
    def __lemmatize__(self,token):   
         
        if self.__lemmatizer is not None: 
            token = self.token.lower() if self.token.isupper() else self.token
            return self.__lemmatizer([token])
        return token,None
    
    def __str__(self):
        return self.lemma
        
    def __repr__(self):
        textfmt = (
            "Token(token='{token}', indexes={indexes}, "
            "pos='{pos}', lemma='{lemma}', lineno={lineno})"
        )
        return textfmt.format(
            token=self.token,
            indexes=self.indexes,
            pos=self.pos,
            lemma=self.lemma,
            lineno=self.lineno
        )




         
if __name__ == "__main__":
    pass        
        
        
