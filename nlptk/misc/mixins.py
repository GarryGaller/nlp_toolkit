import re
import sys,os
import string
import unicodedata
import gensim
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import SpaceTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import TabTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import razdel
#from spacy.pipeline import SentenceSegmenter

from nlptk.patterns import contractions
from nlptk.patterns import patterns
from nlptk.morphology import morphology 
from nlptk.postagging import taggers


class RepeatReplacer():
    def __init__(self, patterns=None, repl=None):
        self.regex = patterns or re.compile("r(\w*)(\w)\2(\w*)")
        self.repl = repl or r"\1\2\3"
        
    def replace(self,word):
        loop_res = regex.sub(self.repl, word)
        if word == loop_res:
            return loop_res
        else:
            return  self.replace(loop_res)



class RegexReplacer():
    def __init__(self, patterns):
        self.patterns = patterns 
        
    def replace(self,text):
        for regex,rep in self.patterns:
            text = regex.sub(rep[0],text)
               
        return text


class TaggerMixin():
    
    tagger4 = taggers.get_tagger('4-ngram_tagger')
    #tagger3 = taggers.get_tagger('3-ngram_tagger')
    
    @staticmethod
    def tagger_4ngram(tokens,  *args, lang="eng", **kwargs):
        '''4-gram tagger'''
        
        yield from TaggerMixin.tagger4(tokens)

    def tagger_3ngram(tokens, *args, lang="eng", **kwargs):
        '''3-gram tagger'''
        
        yield from TaggerMixin.tagger3(tokens)
    
    @staticmethod
    def tagger_nltk(tokens, *args, lang="eng", **kwargs):
        '''Perceptron tagger'''
        
        yield from nltk.pos_tag(tokens,lang=lang)


class SentencizerMixin():
    
    @staticmethod
    def sentencize_nltk(text, *args, lang="english", **kwargs):
        '''сегментация текста на предложения'''
        
        SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt/%s.pickle' % lang)
        #nltk.sent_tokenize(text, language=lang)
        min_len = kwargs.get('min_len',2)
        for  sent in SENTENCE_TOKENIZER.tokenize(text):  
            sent = sent.strip(string.punctuation)
            if len(sent) >= min_len:
                yield sent 
    
    @staticmethod
    def sentencize_nltk_ru(text, *args, lang="russian", **kwargs):
        '''сегментация текста на предложения'''
        
        SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt/%s.pickle' % lang)
        #nltk.sent_tokenize(text, language=lang)
        min_len = kwargs.get('min_len',2)
        for  sent in SENTENCE_TOKENIZER.tokenize(text):  
            sent = sent.strip(string.punctuation)
            if len(sent) >= min_len:
                yield sent 
    
    
    @staticmethod
    def sentencize_razdel(text, *args, **kwargs):
        '''сегментация текста на предложения'''
                             
        SENTENCE_TOKENIZER = razdel.sentenize
        min_len = kwargs.get('min_len',2)
        for  sent in SENTENCE_TOKENIZER(text):  
            sent = sent.strip(string.punctuation)
            if len(sent.text) >= min_len:
                yield sent.text 
    
    @staticmethod
    def sentencize_polyglot(text, *args, **kwargs):
        '''сегментация текста на предложения'''
        
        SENTENCE_TOKENIZER = polyglot.text.Text
        min_len = kwargs.get('min_len',2)
        for  sent in SENTENCE_TOKENIZER(text).sentenize:  
            sent = sent.strip(string.punctuation)
            if len(sent) >= min_len:
                yield sent 
        
  
    @staticmethod
    def sentencize_segtok(text, *args, **kwargs):
        '''сегментация текста на предложения'''
        
        SENTENCE_TOKENIZER = segtok.segmenter
        min_len = kwargs.get('min_len',2)
        for sent in SENTENCE_TOKENIZER.split_multi(text):
            sent = sent.strip(string.punctuation)
            if len(sent) >= min_len:
                yield sent       

         
        
       

#https://webdevblog.ru/podhody-lemmatizacii-s-primerami-v-python/

class LemmatizerMixin():

    @staticmethod
    def lemmatize_en(text, *args, **kwargs):
        '''возвращает слово вместе с частью речи
        работает через библиотеку pattern'''
        
        return gensim.utils.lemmatize(text,*args,**kwargs)
    
    @staticmethod
    def lemmatize_nltk(tokens, *args, lang='eng', **kwargs):
        return morphology.NLTKLemmatizer(lang=lang,**kwargs).lemmatize(tokens, **kwargs)
    
    @staticmethod
    def lemmatize_pt(tokens, *args, lang='en', **kwargs):
        return morphology.PatternLemmatizer(lang=lang,**kwargs).lemmatize(tokens, **kwargs)   
               
    @staticmethod
    def lemmatize(tokens, lang, **kwargs):
        return morphology.PymorphyLemmatizer(lang=lang).lemmatize(tokens, **kwargs) 
    
    @staticmethod
    def lemmatize_ru(tokens, lang='ru',**kwargs):
        return morphology.PymorphyLemmatizer(lang=lang).lemmatize(tokens, **kwargs) 
    
    @staticmethod
    def lemmatize_uk(tokens, lang='uk',**kwargs):
        return morphology.PymorphyLemmatizer(lang=lang).lemmatize(tokens, **kwargs) 
    

class TokenizerMixin():
    
    @staticmethod
    def simple_tokenize(text, strip=None):
        '''Tokenize input test using gensim.utils.PAT_ALPHABETIC.
        Using regexp (((?![\d])\w)+)
        >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
        >>> list(gensim.utils.simple_tokenize(s))
        ['Good', 'muffins', 'cost', 'in', 'New', 'York', 
        'Please', 'buy', 'me', 'two', 'of', 'them', 'Thanks']
        '''
        
        for token in gensim.utils.simple_tokenize(text):
            if token and not token.isspace():
                yield token.strip(strip)    
    
    
    @staticmethod
    def simple_tokenize2(text, strip=None):
        '''
        Using regexp '\b\w+?\b'
        >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
        >>> RE_WORD.findall(s)
        ['Good', 'muffins', 'cost', '3', '88', 'in', 'New', 'York', 
        'Please', 'buy', 'me', 'two', 'of', 'them', 'Thanks']
        '''
 
        for match in patterns.RE_WORD.finditer(text):
            token = match.group()
            if token and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def token_tokenize(text, strip=None):
        
        for token in patterns.RE_TOKEN.split(text): 
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def toktok_tokenize(text, strip=None):
        '''
        >>> text = u'Is 9.5 or 525,600 my favorite number?'
        >>> ToktokTokenizer().tokenize(text)
        ['Is', '9.5', 'or', '525,600', 'my', 'favorite', 'number', '?']
        s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> ToktokTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 
        'It', "'", 's', 'inexpensive.', 'Free-for-all.', 
        'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
        >>> 
        '''

        for token in ToktokTokenizer().tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def space_tokenizer(text, strip=None):
        ''' Only " " blank character
        Same as s.split(" ")
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> SpaceTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$3.88\nin', 'New', 'York.', 
        "It's", 'inexpensive.', 'Free-for-all.', 
        'Please', 'buy', 'me\ntwo', 'of', 'them.\n\nThanks.']
        >>> s.split(' ')
        ['Good', 'muffins', 'cost', '$3.88\nin', 'New', 'York.', 
        "It's", 'inexpensive.', 'Free-for-all.', 
        'Please', 'buy', 'me\ntwo', 'of', 'them.\n\nThanks.']
        >>>'''
        
        for token in SpaceTokenizer().tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    

    @staticmethod
    def whitespace_tokenizer(text, strip=None):
        ''' space, tab, newline
        Same as s.split()
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.', 
        "It's", 'inexpensive.', 'Free-for-all.', 
        'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']
        >>>
        >>> s.split()
        ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.', 
        "It's", 'inexpensive.', 'Free-for-all.', 
        'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']
        >>> 
        '''
        
        for token in WhitespaceTokenizer().tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def tab_tokenizer(text,strip=None):
        '''tab-based tokenization'''
        
        for token in TabTokenizer().tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    

    
    @staticmethod
    def wordpunct_tokenize(text, strip=None):
        '''
        Using the regexp \w+|[^\w\s]+
        
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> WordPunctTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.', 
        'It', "'", 's', 'inexpensive', '.', 'Free', '-', 'for', '-', 'all', '.', 
        'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
        >>> 
        
        >>> nltk.tokenize.word_tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 
        'It', "'s", 'inexpensive', '.', 'Free-for-all', '.', 
        'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
        '''
        
        
        for token in WordPunctTokenizer().tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
   

    @staticmethod
    def treebank_word_tokenize(text, strip=None):
        '''
        using NLTK’s recommended word tokenizer (currently an improved 
        TreebankWordTokenizer along with PunktSentenceTokenizer for the specified language)
        
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> nltk.word_tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 
        'It', "'s", 'inexpensive', '.', 'Free-for-all', '.', 
        'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
        '''
        
        
        for token in nltk.word_tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)

        
    @staticmethod
    def regexp_tokenize(text,pattern=None,strip=None):
        '''
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        >>> tokenizer.tokenize(s)
        ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York', '.', 
        'It', "'s", 'inexpensive', '.', 'Free', '-for-all.', 
        'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']
        '''
        
        tokenizer = RegexpTokenizer(pattern or patterns.RE_WORD2)
        for token in tokenizer.tokenize(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
   
    @staticmethod
    def punct_tokenize(text, strip=None):
        '''
        >>> s = "Good muffins cost $3.88\nin New York. It's inexpensive. Free-for-all. Please buy me\ntwo of them.\n\nThanks."
        >>> RE_PUNCT.split(s)
        ['Good muffins cost ', '$', '3', '.', '88\nin New York', '.', 
        ' It', "'", 's inexpensive', '.', ' Free', '-', 'for', '-', 'all', '.', 
        ' Please buy me\ntwo of them', '.', '\n\nThanks', '.', '']
        ''' 
        
        for token in patterns.RE_PUNCT.split(text): 
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)

    
    
    @staticmethod
    def nonalpha_tokenize(text, strip=None):
        '''
        >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
        >>> RE_NONALPHA.split(s)
        ['Good', 'muffins', 'cost', '', '3', '88', 'in', 'New', 'York', 
        '', '', 'Please', 'buy', 'me', 'two', 'of', 'them', '', '', 'Thanks', '']
        '''
        
        for token in patterns.RE_NONALPHA.split(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def whitespace_tokenize2(text, strip=None):
        '''>>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
        >>> RE_WHITESPACE.split(s)
        ['Good', ' ', 'muffins', ' ', 'cost', ' ', '$3.88', '\n', 'in', ' ', 
        'New', ' ', 'York.', ' ', 'Please', ' ', 'buy', ' ', 'me', '\n', 
        'two', ' ', 'of', ' ', 'them.', '\n', 'Thanks.']'''
         
        for token in patterns.RE_WHITESPACE.split(text):
            if token not in patterns.PUNCTUATION and not token.isspace():
                yield token.strip(strip)
    
    
    @staticmethod
    def tags_tokenize(text, strip=None):
        
        for token in patterns.RE_TAGS.split(text):
            if token and not token.isspace():
                yield token.strip(strip)
    

class StripperMixin():
    '''
    
    RE_PUNCT - Regexp for search an punctuation.
    RE_TAGS - Regexp for search an tags.
    RE_NUMERIC - Regexp for search an numbers.
    RE_NONALPHA - Regexp for search an non-alphabetic character.
    RE_NONASCII - Regexp for search an non-ascii character.
    RE_AL_NUM - Regexp for search a position between letters and digits.
    RE_NUM_AL - Regexp for search a position between digits and letters .
    RE_SPACES - Regexp for search space characters.
    '''
    
    @staticmethod
    def strip_accent(text):   
        '''Remove letter accents from the given string.'''
        norm = unicodedata.normalize("NFD", text)
        result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
        return unicodedata.normalize("NFC", result)
        #return gensim.utils.deaccent(text) 
    
    @staticmethod
    def strip_quotes(text):
        '''Removes a variety of quotes from the text'''
        return patterns.RE_QOUTES.sub('',text) 
    
    @staticmethod
    def strip_hyphenation(text):
        """Removing hyphenation."""
        text = patterns.RE_HYPHENATED_WORD.sub(r"\1\2",text)
        return text
    
    @staticmethod
    def strip_punctuation(text,marker=' '):
        '''Replace punctuation characters with spaces in s using RE_PUNCT
        >>> strip_punctuation(string.punctuation)
        ' '
        >>> 
        '''
        return patterns.RE_PUNCT2.sub(marker, text)
        #return gensim.parsing.preprocessing.strip_punctuation(text)
    
    @staticmethod
    def strip_tags(text,marker=''):
        '''Remove tags from s using RE_TAGS.
        >>> strip_tags('<href="http://google.com">')
        ''
        >>> 
        '''
        return patterns.RE_TAGS.sub(marker, text)
        #return gensim.parsing.preprocessing.strip_tags(text)

    @staticmethod  
    def strip_urls(text, marker=''):
        '''Remove URL's'''
        return patterns.RE_URLS.sub(marker, text)
    
    @staticmethod
    def strip_multiple_whitespaces(text):
        '''Remove repeating whitespace characters (spaces, tabs, line breaks) 
        from s and turns tabs & line breaks into spaces using RE_WHITESPACE'''
        return patterns.RE_WHITESPACE.sub(" ", text)
        #return gensim.parsing.preprocessing.strip_multiple_whitespaces(text)    
    
    @staticmethod
    def strip_digit(text,marker=''):
        '''  Remove digits (0..9 + some others) from s using RE_DIGIT'''
        return patterns.RE_DIGIT.sub(marker, text)
        #return gensim.parsing.preprocessing.strip_numeric(text)    
    
    @staticmethod
    def strip_decimal(text,marker=''):
        '''  Remove decimal from s using RE_DECIMAL'''
        return patterns.RE_DECIMAL.sub(marker, text)
        #return gensim.parsing.preprocessing.strip_numeric(text)  
    
    @staticmethod
    def strip_numeric(text,marker=''):
        '''  Remove numeric  from s using RE_NUMERIC'''
        return patterns.RE_NUMERIC.sub(marker, text)
        #return gensim.parsing.preprocessing.strip_numeric(text)  
    
    @staticmethod
    def strip_roman_numerals(text, marker=''):
        '''  Remove digits from s using RE_ROMAN_NUMERALS'''
        return patterns.RE_ROMAN_NUMERALS.sub(marker, text)
    
    
    @staticmethod
    def strip_nonletter_sequences(text, marker=' '):
        ''' Remove non-letter sequences'''
        return patterns.RE_NONLETTER.sub(marker, text)
    
    @staticmethod
    def strip_contractions(text): 
        '''Replacing common contractions'''
        return RegexReplacer(contractions.CONTRACTIONS).replace(text)
    
    @staticmethod
    def strip_possessive_endings(text, marker=''): 
        '''Replacing common contractions'''
        return patterns.RE_POSSESSIVE_ENDINGS.sub(marker, text)
    
    #--------------------------------------------------
    @staticmethod    
    def strip_non_alphanum(text):
        ''' Remove non-alphabetic characters from s using RE_NONALPHA'''
        return patterns.RE_NONALPHA.sub(" ", text)
        #return gensim.parsing.preprocessing.strip_non_alphanum(text) 
    
    @staticmethod
    def strip_nonasci(text):
         ''' Remove non-ASCII characters from s using RE_NONASCII'''
         return patterns.RE_NONASCII.sub('',text) 
    
    @staticmethod
    def strip_stopwords(text):
        '''Remove STOPWORDS from s'''
        return gensim.parsing.preprocessing.remove_stopwords(text)
    
    @staticmethod
    def strip_short(text,minsize=3):
        '''Remove words with length lesser than minsize from s.'''
        return " ".join(e for e in text.split() if len(e) >= minsize)
        #return gensim.parsing.preprocessing.strip_short(text,minsize=minsize)
   
    @staticmethod
    def split_alphanum(text):
        '''Add spaces between digits & letters in s using RE_AL_NUM''' 
        s = patterns.RE_AL_NUM.sub(r"\1 \2", text)
        return patterns.RE_NUM_AL.sub(r"\1 \2", text)
        #return gensim.parsing.preprocessing.split_alphanum(text)
        

    @staticmethod
    def strip_chars(text, chars=string.punctuation):   
        '''Removes the beginning and ending punctuation marks from this line.'''
        return text.strip(chars)
    
    
class RemoverMixin():    
    
    #--------------------------------------------------
    
    @staticmethod
    def remove_short(tokens, minsize=3):
        '''Remove tokens length less than min_word_len characters'''
        result = []
        if not minsize: return tokens
        for token in tokens:
            tok = token.lemma if hasattr(token,'lemma') else token
            if len(tok) >= minsize:
                result.append(token)
            
        return result
    
    @staticmethod
    def remove_stopwords(tokens, stopwords=[]):
        '''Remove STOPWORDS from list tokens'''
        result = []
        if not stopwords: return tokens
        for token in tokens:
            tok = token.lemma if hasattr(token,'lemma') else token
            if tok.lower() not in stopwords:
                result.append(token)
            
        return result
    
    @staticmethod       
    def remove_ifnotin_lexicon(tokens, lexicons=[]):
        '''Lexicons is a list of lists of words'''
        result = []
        if not lexicons: return tokens
        for token in tokens:
            tok = token.lemma if hasattr(token,'lemma') else token
            for lex in lexicons:
                if any(t in lex for t in [tok,tok.capitalize()]):
                    result.append(token)
        return result
    
    @staticmethod  
    def remove_if_proper_name(tokens, names=[]):
        '''Names is a list of words'''
        result = []
        if not names: return tokens
        for token in tokens:
            tok = token.lemma.lower() if hasattr(token,'lemma') else token.lower()
            if tok.capitalize() not in names:
                result.append(token)     
        return result
    
    @staticmethod
    def remove_by_tagpos(tokens, allowed_tags=[],disallowed_tags=[]):
        '''
        allowed_tags is a list or set of allowed tags for parts of speech
        disallowed_tags is a list or set of disallowed tags for parts of speech
        '''
      
        def allowed():
            for token in tokens:
                pos = token.pos if hasattr(token,'pos') else ''
                if pos in allowed_tags:
                    result.append(token)
               
        
        def disallowed():
            for token in tokens:
                pos = token.pos if hasattr(token,'pos') else ''
                if pos not in disallowed_tags:
                    result.append(token)
                
        
        result = []
        if (allowed_tags and disallowed_tags):
            allowed_tags = set(allowed_tags) - set(disallowed_tags)
            allowed()
            
        elif (allowed_tags and not disallowed_tags):
            allowed()
           
        elif not allowed_tags and disallowed_tags:
            disallowed()
            
        else:
            result = tokens
            
        return result

    @staticmethod
    def remove_punctuation(tokens,chars=string.punctuation):
        '''Removes tokens that represent punctuation characters'''
        result = []
        
        for token in tokens: 
            tok = token.word if hasattr(token,'word') else token
            if tok not in chars:
                result.append(token)  
        return result
    
    @staticmethod
    def remove_case(tokens,*args):
        '''Removes the case of words'''
        result = []
        
        for token in tokens: 
            if hasattr(token,'token'):
                token.lemma = token.lemma.lower()
            else:
                token = token.lower()
            result.append(token)  
        return result
    
    
    @staticmethod
    def remove_trailing_chars(tokens,chars=string.punctuation):
        '''Removes the start and end characters from each token'''
        result = []
        
        for token in tokens: 
            if hasattr(token,'word'):
                token.word = token.word.strip(chars)
            else:
                token = token.strip(chars)
            result.append(token)  
        return result
       
    #-----------------------------------------------
    # НЕ ИСПОЛЬЗУЮТСЯ, ПОЭТОМУ НЕ РАБОТАЕТ С КЛАССОМ TOKEN
    
    @staticmethod
    def remove_quotes(tokens, *args):
        '''Removes a variety of quotes from the token'''
        result = [patterns.RE_QOUTES.sub('',token) for token in tokens]
        return result 
    
    
    @staticmethod
    def remove_nonasci(tokens, *args):
        '''Removes tokens that contain non-ascii characters'''
        def is_ascii(s):
            return all(ord(c) < 128 for c in s)
        
        return list(filter(lambda token: is_ascii(token),tokens))       
    
    @staticmethod
    def remove_nonalphabetic(tokens,other=''):
        '''Removes tokens that contain something other than Latin letters'''
        letters = set(string.ascii_letters + other)
        def ascii_letters(s):
            nonlocal letters
            return all(c in letters for c in s)
        
        return filter(lambda token: ascii_letters(token),tokens)   
        
    @staticmethod
    def remove_empty(tokens, *args):
        return filter(
            lambda token: token in ('\r\n','\r','\n','\t','',' '),
            tokens)
    
    @staticmethod       
    def remove_numeric(tokens, *args):
        '''Removes numeric'''
        def is_numeric(token):
            return token.isnumeric()
        
        return filter(lambda token: not is_numeric(token),tokens)
    
    @staticmethod       
    def remove_roman_numerals(tokens, *args):
        '''Removes Roman numerals'''
        def is_roman(token):
            return patterns.ROMAN_NUMERALS.match(token)
        
        return filter(lambda token: not is_roman(token),tokens)
     
    
    @staticmethod
    def remove_stopwords2(tokens, stopwords):
        '''Remove STOPWORDS from list tokens'''
        return gensim.corpora.textcorpus.remove_stopwords(tokens, stopwords)
    
    
    @staticmethod
    def remove_short2(tokens, minsize=3):
         '''Remove tokens shorter than `minsize` chars'''
         return gensim.corpora.textcorpus.remove_short(tokens, minsize=minsize) 
    
    
    @staticmethod
    def make_lower(tokens, *args):
        return list(map(str.lower,tokens))
    

# НЕ ИСПОЛЬЗУЕТСЯ
class FilterMixin():    
    #----------------------------------------------------------
    # фильтры постобработки токенов
    # должны возвращать True для допустимого токена
    #----------------------------------------------------------
    # фильтр на непустую лемму
    @staticmethod
    def in_nonempty_lemmas(token):
        token = token.lemma if hasattr(token,'lemma') else token
        return token not in ('',None)
     

    # фильтр на разрешенные теги
    @staticmethod
    def in_allowed_tags(token, allowed_tags=[]):
        pos = token.pos if hasattr(token,'pos') else ''
        if not allowed_tags:
            result = True
        else:
            result = pos in allowed_tags
        return result

    # фильтр на неразрешенные теги
    @staticmethod
    def in_nondisallowed_tags(token, disallowed_tags=[]):
        pos = token.pos if hasattr(token,'pos') else ''
        return pos not in disallowed_tags


    @staticmethod  
    def isnot_proper_name(token, lexicon=[]):
        token = token.lemma.lower() if hasattr(token,'lemma') else token.lower()
        return token.capitalize() not in lexicon 
    
    @staticmethod  
    def isnot_proper_name2(token):
        indexes = token.indexes
        token_ = token.word if hasattr(token,'word') else token
        result = not (token_[0].isupper() and  0 not in indexes)
        #if not result:
        #    print(repr(token), result)
        return result
    
    #----------------------------------------------
    # фильтр на принадлежность леммы определенному лексикону
    @staticmethod
    def in_lexicon1(token, lexs):
        return any(tok in lex  
            for tok in  [token.lemma,token.capitalize()]
         for lex in lexs
    )
    # фильтр на принадлежность леммы определенному лексикону
    @staticmethod
    def in_lexicon(token, lexicons):
        result = False
        token = token.lemma.lower() if hasattr(token,'lemma') else token.lower()
        for lex in lexicons:
            if any(tok in lex for tok  in [token,token.capitalize()]): 
                result = True
                break

        return result    
    
   

class PreprocessorMixin():        
    
    @staticmethod
    def stem_text(text,**kwargs):
        '''Transform s into lowercase and stem it'''
        return gensim.parsing.preprocessing.stem_text(text)
    
    
    @staticmethod
    def simple_preprocess(text, **kwargs):
        '''
        Convert a document into a list of lowercase tokens, 
        ignoring tokens that are too short or too long.
        deacc=False, min_len=2, max_len=15
        Uses gensim.utils.tokenize => gensim.utils.simple_tokenize
        
        '''
        
        return gensim.utils.simple_preprocess(text, **kwargs)
    
    @staticmethod   
    def preprocess_string(text,filters):
        '''Apply list of chosen filters to `s`.
        
        Default list of filters:
        strip_tags(),
        strip_punctuation(),
        strip_multiple_whitespaces(),
        strip_numeric(),
        remove_stopwords(),
        strip_short(),
        stem_text().'''
        
        return gensim.parsing.preprocessing.preprocess_string(text,filters=filters)

'''        
gensim.corpora.textcorpus.TextCorpus        
preprocess_text() использует:        
lower_to_unicode() - lowercase and convert to unicode (assumes utf8 encoding)
deaccent()- deaccent (asciifolding)
strip_multiple_whitespaces() - collapse multiple whitespaces into a single one
simple_tokenize() - tokenize by splitting on whitespace
remove_short() - remove words less than 3 characters long
remove_stopwords()
'''     




if __name__ =="__main__":
    
    text = """I am in your team, aren’t I?
    I’m not gonna play tennis with you"""    
    replacer = RegexReplacer(contractions.CONTRACTIONS)
    print(replacer.replace(text)) 
         
                    
        
    
    
