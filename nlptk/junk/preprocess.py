import os,sys
from functools import partial



from nlptk.mining.text import Lexicon        
from nlptk.misc.mixins import (
                StripperMixin,
                RemoverMixin,
                FilterMixin,
                TokenizerMixin,
                LemmatizerMixin,
                SentencizerMixin)

class Preprocessor(StripperMixin,
                TokenizerMixin,
                LemmatizerMixin,
                SentencizerMixin):        
        
    allowed_tags = {
        'NN','NNS','NNP','NNPS', # существительные
        'RB','RBR','RBS', # прилагательные
        'JJ','JJR','JJS',  # наречия
        'VB','VBZ','VBP','VBD','VBN','VBG', # глаголы
        #'FW' # иностранные слова
    }

    disallowed_tags = set()
     
    token_rules = {
        'ignore_tokens_without_lemma': True,
        'make_token_lower':False,
        'make_lemma_lower':True
    } 
        
    def __init__(self,
            cfg,
            **kwargs
            ):        
        
        
        if cfg.get('lexpath'):
            self.LEXICON_PROPER_NAMES = Lexicon(
                os.path.join(cfg['lexpath'],'names')
                ).load()
            self.LEXICON_COMMON = Lexicon(
                os.path.join(cfg['lexpath'] ,'lexicon')
                ).load()
        else:
            self.LEXICON_PROPER_NAMES = []
            self.LEXICON_COMMON = []
        
        self.min_word_len = cfg.get('min_word_len',3)
        self.stopwords = cfg.get('stopwords',[])
        self.tagger = cfg.get('tagger')
      
        self.__dict__.update(kwargs)
      
        self.allowed_tags = set(self.allowed_tags) - set(self.disallowed_tags)
        
        
        self.preprocess_funcs = {
            'text_filters':[self.strip_hyphenation],
            'char_filters': [
                  self.strip_tags,
                  self.strip_accent,
                  self.strip_multiple_whitespaces
            ],
            # self.treebank_word_tokenize, #self.whitespace_tokenizer,   #self.toktok_tokenize
            'tokenizer':self.treebank_word_tokenize, 
            'token_filters':[
                  self.remove_quotes,
                  self.remove_trailing_chars,
                  self.remove_numeric,
                  self.remove_roman_numerals,
                  partial(self.remove_nonalphabetic,other="-’'"),
                  partial(self.remove_short,minsize=self.min_word_len), 
                  partial(self.remove_stopwords,stopwords=self.stopwords),
                
            ],
            'token_rules': self.token_rules,
            'lexicon_filters':[partial(self.in_lexicon, lexicons=[self.LEXICON_COMMON])],
            'lemma_filters': [
                  self.in_nonempty_lemmas,
                  partial(self.in_allowed_tags, allowed_tags=self.allowed_tags),
                  partial(self.in_nondisallowed_tags, disallowed_tags=self.disallowed_tags),
                  partial(self.isnot_proper_name,names=self.LEXICON_PROPER_NAMES),
                  self.isnot_proper_name2
            ],
            #'lemmatizer': partial(self.lemmatize_pt,pos=True),
            'lemmatizer': partial(self.lemmatize_nltk,pos=True,tagger=self.tagger),
            'sentencizer':self.sentencize_nltk
            }
        preprocess_funcs = kwargs.get('preprocess_funcs',{}) or kwargs.get('pf',{})
        if kwargs.get('replace'):
            self.preprocess_funcs = preprocess_funcs
        else:
            self.preprocess_funcs.update(preprocess_funcs)
        
        self.preprocess_funcs.update({
            key:val 
                for key,val in kwargs.items() 
                    if key in self.preprocess_funcs
            }
        )
