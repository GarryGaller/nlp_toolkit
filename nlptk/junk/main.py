import sys,os


import glob,time

from nlptk.postagging.taggers import  get_tagger
from nlptk.junk.preprocess import Preprocessor
from nlptk.junk.vocabulary import (Stream,
                                        BaseCorpus,
                                        PlainCorpus,
                                        Text,
                                        Lexicon,
                                        chardetector)
from nlptk.misc.mixins import *
from nlptk.junk.configs import Config

from pprint import pprint
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize
import gensim
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora import Dictionary
from functools import partial
from pprint import pprint
import pattern.en
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.corpus import names
from nltk.corpus import gazetteers


if sys.version_info < (3,6):
    import win_unicode_console
    win_unicode_console.enable()




APPDIR = os.path.abspath(os.path.dirname(__file__))


'''
countries = gazetteers.words('countries.txt')  # 289
uscities  = gazetteers.words('uscities.txt')
usstates =  gazetteers.words('usstates.txt')
nationalities = gazetteers.words('nationalities.txt')
'''

'''
ALL_GAZET = gazetteers.words() # 1211


PROPER_NAMES = names.words() # 7944

#PROPER_NAMES = list(set(map(str.lower,PROPER_NAMES)))

LEXICON = list(pattern.en.lexicon.keys())  # lazy dict    94185
words_lst = words.words()        # list                   236736
wordnet_lst = list(wordnet.words()) # dict_keyiterator    144018
LEXICON += words_lst + wordnet_lst + ALL_GAZET + PROPER_NAMES

LEXICON = set(LEXICON)

print(len(LEXICON))  #  400164; if set lower - 373821
'''

#names = nltk.corpus.names   # nltk.corpus.util.LazyCorpusLoader

#names_lst = ([name for name in names.words('male.txt')] +
#         [name for name in names.words('female.txt')])   # 7944


def test_lexicons():
    global LEXICON, PROPER_NAMES
    print(os.path.join(APPDIR,'lexicons',"*.dawg"))           
    path = os.path.join(APPDIR,'lexicons')
    if not glob.glob(os.path.join(path,"*.dawg")):
                
        Lexicon(os.path.join(path,'lexicon'), LEXICON).save()
        Lexicon(os.path.join(path,'names'), PROPER_NAMES).save()
    else:
        LEXICON = Lexicon(os.path.join(path,'lexicon')).load()
        PROPER_NAMES = Lexicon(os.path.join(path,'names')).load()

    print('hooey' in LEXICON)
    print('lovelier' in LEXICON)

    print('Mary' in PROPER_NAMES)




'''
https://docs.huihoo.com/nltk/0.9.5/api/nltk.corpus-module.html

corpus.words(): list of str
corpus.sents(): list of (list of str)
corpus.paras(): list of (list of (list of str))
corpus.tagged_words(): list of (str,str) tuple
corpus.tagged_sents(): list of (list of (str,str))
corpus.tagged_paras(): list of (list of (list of (str,str)))
corpus.chunked_sents(): list of (Tree w/ (str,str) leaves)
corpus.parsed_sents(): list of (Tree with str leaves)
corpus.parsed_paras(): list of (list of (Tree with str leaves))
corpus.xml(): A single xml ElementTree
corpus.raw(): unprocessed corpus contents
'''

      
        
 
def test_stream():
    
    
    sentencizer =  SentencizerMixin().sentencize_nltk
    lemmatizer = LemmatizerMixin().lemmatize_nltk
    tokenizer =  TokenizerMixin().toktok_tokenize
    #text.prepare(**preprocess_funcs)
    
    config = Config()
    cfg = config.to_dict()
    
    filepath = os.path.join(config.SOURCEDIR,r"txt\fitzgerald_great_gatsby_txt.txt")
    stream = Stream(filepath,encoding='chardetect',sentencizer=sentencizer)

    for lineno, s in enumerate(stream):
        print(lineno,s)



def test_corpus():
    config = Config()
    cfg = config.to_dict()

    prp = Preprocessor(cfg, disallowed_tags={'NNP','NNPS'})
    crp = PlainCorpus(cfg,prp)
    text = next(crp) # генератор будет выдавать по одному тексту-генератору
    sent = next(txt) # генератор будет выдавать по одному предложению
    print(sent)
    print(repr(sent))
    
   
   
def test_text(inplace=True):
    config = Config()
    cfg = config.to_dict()
    filepath = os.path.join(config.SOURCEDIR,r"txt\1.txt")
    #filepath = os.path.join(config.SOURCEDIR, "txt\Edgar Allan Poe The Cask of Amontillado.txt")
    sentencizer =  SentencizerMixin().sentencize_nltk
    lemmatizer = LemmatizerMixin().lemmatize_nltk
    tokenizer =  TokenizerMixin().toktok_tokenize

    preprocess_funcs = {
         'sentencizer':sentencizer,
         'tokenizer':tokenizer,        
         'lemmatizer': partial(lemmatizer,pos=True),   
         'text_filters':[],
         'character_filters':[],
         'token_filters':[],
         'lemma_filters':[],
         'lexicon_filters':[],
         'token_rules':{},
    } 
    tagger = get_tagger()
    prp = Preprocessor(
        cfg,
        pf=preprocess_funcs,
        disallowed_tags={'NNP','NNPS'},
        tagger=tagger
    )
    
    
    if inplace:
        # возврат всех данных 
        text = Text(filepath,prp,inplace=True)
        print(text.sents())
        print(text.sents(raw_sents=True))
    else:
        # генератор
        text = Text(filepath,Preprocessor(cfg,preprocess_funcs={}))
        for txt in text:
            print(repr(txt))


def test_text2(inplace=True):
    config = Config()
    cfg = config.to_dict()
    filepath = os.path.join(config.SOURCEDIR,r"txt\1.txt")
    #filepath = os.path.join(config.SOURCEDIR, "txt\Edgar Allan Poe The Cask of Amontillado.txt")
 
    prp = Preprocessor(cfg,pf={},replace=True)
    
    if inplace:
        # возврат всех данных 
        text = Text(filepath,prp,inplace=True)
        print(text.sents())
        print(text.sents(raw_sents=True))
    else:
        # генератор
        text = Text(filepath,prp)
        for txt in text:
            print(repr(txt))


def create_frequency_lists():

    start_text = time.time()
    time_lines = []

    config = Config()
    config.OUTDIR = os.path.join(config.APPDIR,r'FREQLIST')
    config.PICKLEDIR = os.path.join(config.APPDIR,r'store2')
    cfg = config.to_dict()
    tagger = get_tagger()
    prp = Preprocessor(cfg, disallowed_tags={'NNP','NNPS'},tagger=tagger)
    
    
    for text in PlainCorpus(cfg,prp):
        
        if text.pickled(config.PICKLEDIR):
            text.load(config.PICKLEDIR)
        else:
            start_line = time.time()
            for n,line in enumerate(text):
                end_line = time.time() - start_line
                start_line = time.time()
                time_lines.append(end_line)
            text.save(config.PICKLEDIR)    
        
        pprint(text.info())
        pprint(text.lines[0])
        pprint(text.lines[-1])
        
        print(time.time() - start_text)
        start_text = time.time()
        #----------------------------------------
        
        name, ext = os.path.splitext(text.name)
        path = os.path.join(config.OUTDIR, name + '.txt')
        with open(path,'w') as f:
            n = len(text.vocab)
            #lst = text.counter.most_common()[:-n:-1] 
            lst = sorted(text.vocab.items(), key = lambda tup: (tup[1], tup[0]))
            #pprint(lst)
            lst = ["{:<20}{:}".format(k,v) for k,v in lst]
            f.writelines('\n'.join(lst))
        
        print('--------------------')
        
        
        """
        nnp = [repr(token) 
            for sent in text.lines 
                for token in sent.tokens 
                    if token.pos == 'NNP']
        pprint(nnp)
    """
    

if __name__ == "__main__":
    #test_stream()    
    #test_text(inplace=True)
    test_text2(inplace=True)



    











"""
's притяжательный падеж существительных
-s\-es как  множественное число существительных
-s как форманта 3-го лица глаголов
сравнительная и превосходная степень прилагательных
"""
