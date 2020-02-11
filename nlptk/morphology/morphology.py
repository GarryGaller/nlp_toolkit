from functools import partial
import sys,os
from pymorphy2 import MorphAnalyzer
from nltk import WordNetLemmatizer,pos_tag
from nltk.corpus import wordnet
#from pattern.en import lemma as pattern_lemma, tag as pattern_tag
#from treetaggerwrapper import TreeTagger,make_tags



class TreeTaggerLemmatizer():
    '''Тэггер Хелмута Шмида из института компьютерной лингвистики университета Штутгарта.
    Доступен только в виде исполняемого файла; имеет свой набор тэгов,
    которые не совпадают с тэгами NLTK
    https://treetaggerwrapper.readthedocs.io/en/latest/
    '''
    
    def __init__(self,lang="en",
                tagdir='TreeTagger',
                tagparfile='english.par',
                **kwargs):   
        
        self.__dict__.update(kwargs)
        
        if not os.path.isabs(tagdir):
            tagdir = os.path.join(sys.exec_prefix,
                r'lib\site-packages',tagdir
            )
        
        if not os.path.isabs(tagparfile):
            tagparfile = os.path.join(tagdir,
                r'lib\english.par',tagparfile
            )
        
        self.morph = TreeTagger(
            TAGLANG=lang, TAGDIR=tagdir, TAGPARFILE=tagparfile
        ).tag_text
        
    
    def lemmatize(self, tokens, **kwargs):
        tags = self.morph(' '.join(tokens))
        tags = make_tags(tags)
        
        result = [] 
        
        for tag in tags:
            lemma = tag.lemma
            pos = tag.pos
            if pos == "''":
                pos = '<UNK>'
            
            res = lemma
            if kwargs.get('pos'):
                res = lemma, pos
            result.append(res) 
        
        return result
       
        
class PatternLemmatizer():
    '''PatternLemmatizer
    N.B. Функция lemma автоматически меняет регистр букв на нижний!
    
    Ошибается в определении части речи. Присваивает NNP почти всем словам с капитализацией.
    Неверно лемматизирует слова с окончаниями на "s"  и т.д.
    
    lemma(verb, parse=True) method of pattern.text.en.inflect.Verbs instance
    Returns the infinitive form of the given verb, or None.
    
    tag(s, tokenize=True, encoding='utf-8', **kwargs)
    Returns a list of (token, tag)-tuples from the given string.
    '''
    
    def __init__(self, lang="en", **kwargs):   
        self.morph = pattern_lemma
        self.tagger = pattern_tag
        self.__dict__.update(kwargs)
    
    
    def __lemmatize__(self, token, **kwargs):
        lemma = self.morph(token)
        return lemma
    
    
    def lemmatize(self, tokens, **kwargs):
        
        result = [] 
        tagger = kwargs.get('tagger',self.tagger) 
        tags = tagger(' '.join(tokens))
        for term,pos in tags:
            lemma = self.__lemmatize__(term,**kwargs)
            res = lemma
            if pos == "''":
                pos = 'UNKN'
            if kwargs.get('pos'):
                res = lemma, pos
            result.append(res) 
        
        return result


class NLTKLemmatizer():
    '''
    >>> from nltk import WordNetLemmatizer,pos_tag
    >>> from nltk.corpus import wordnet
    >>> from pattern.en import lemma as pattern_lemma, tag as pattern_tag
    >>> pos_tag(['Darcy'])
    [('Darcy', 'NN')]
    >>>
    >>> pos_tag(['Mary'])
    [('Mary', 'NNP')]
    >>>
    >>> pos_tag(["Mary's"])
    [("Mary's", 'NN')]
    >>> pos_tag(["n't"])
    [("n't", 'RB')]
    >>> pos_tag(["'"])
    [("'", "''")]
    >>> pos_tag([" "])
    [(' ', 'NN')]
    >>> pos_tag(['Abraham'])
    [('Abraham', 'NNP')]
    >>> pos_tag(['John'])
    [('John', 'NNP')]
    >>> pos_tag(['Mary'])
    [('Mary', 'NNP')]
    >>> pos_tag(["Mary's"])
    [("Mary's", 'NN')]
    >>>
    >>> pattern_tag('Darcy')
    [('Darcy', 'NNP')]
    >>> 
    >>> pattern_tag("Mary")
    [('Mary', 'NNP')]
    >>> 
    >>> pattern_tag("Mary's")
    [('Mary', 'NNP'), ("'s", 'POS')]
 
    >>> pos_tag(['wordless'])
    [('wordless', 'NN')]
    >>> WordNetLemmatizer().lemmatize('worldless',wordnet.NOUN)
    'worldless'
    >>> pattern_tag('wordless')
    [('wordless', 'NN')]
    >>> pattern_lemma('wordless')  #  неверная лемматизация
    'wordles'
    '''

    def __init__(self, lang='eng', **kwargs):
        
        self.morph = WordNetLemmatizer()  
        self.tagger = partial(pos_tag,lang=lang)
        self.__dict__.update(kwargs)
       
        
    def __lemmatize__(self, token, pos, **kwargs):
        lemma = self.morph.lemmatize(token,pos=pos)    
        return lemma
    
    
    def lemmatize(self, tokens, **kwargs):
        tagset = kwargs.get('tagset') or self.__dict__.get('tagset')
        result = [] 
        tagger = kwargs.get('tagger',self.tagger)
        for term,pos in tagger(tokens,tagset=tagset):
            pos_wordnet = self.get_wordnet_pos(pos)
            lemma = self.__lemmatize__(term,pos=pos_wordnet) 
            if pos == "''":
                pos = 'UNKN'
            if kwargs.get('pos'):
               lemma = lemma, pos
            result.append(lemma)
             
        return result         
    
    
    def get_wordnet_pos(self, pos):
        '''nltk are translating tags into tags wordnet
        In addition to this set of verb tags, the various forms
        of the verb to be have special tags:
        be/BE, being/BEG, am/BEM, are/BER, is/BEZ, been/BEN, were/BED and was/BEDZ
        '''
        
        tag = pos[0].upper() # извлекаем первую букву из аббревиатуры части речи
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "B": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
        
        

class PymorphyLemmatizer():    
    
    def __init__(self, lang='ru', **kwargs):
        self.morph = MorphAnalyzer(lang=lang)  
  
    def lemmatize(self, tokens, **kwargs):
        
        result = [] 
        
        for term in tokens:
            res = self.__lemmatize__(term,**kwargs)
            result.append(res) 
       
        return result
    
       
    def __lemmatize__(self, token,**kwargs):
        '''
        Слово преобразуется к нормальной форме
        http://opencorpora.org/dict.php?act=gram. 
        '''
            
        # первый объект Parse из всех возможных грамматических разборов слова
        parse = self.morph.parse(token)[0] 
        tag = parse.tag  # OpencorporaTag
        
        if tag.POS is None or tag.grammemes == {'UNKN'}:
            lemma = token
            pos = 'UNKN'
        else:
            pos = tag.POS
            # приводим женские фамилии с окончанием на -ой 
            # (род., дат., творит., предл. падежи ) к правильной форме
            try:
                # укр. морфогенератор не знает тега Surn
                if self.morph.lang == 'ru' and {'femn','Surn'} in tag:
                    # склоняем в именительный падеж един. число женского рода
                    try:
                        word = parse.inflect({'nomn', 'sing' ,'femn'})
                        if word:
                            lemma = word.word
                        else:
                            lemma = parse.normal_form   
                    except Exception:
                        lemma = parse.normal_form    
                else:
                    # у прочих слов просто получаем нормальную форму: 
                    # имен. падеж ед. число
                    lemma = parse.normal_form
            except Exception as err:
                lemma = parse.normal_form
        
            result = lemma    
        
        if kwargs.get('pos'):
            result = lemma, pos
        
        return result 
                
                
if __name__ == "__main__":
    if sys.version_info < (3,6):
        import win_unicode_console
        win_unicode_console.enable()
    
    from pprint import pprint
    import nltk
    from nlptk.misc.mixins import TaggerMixin
  
     
    sent = 'This was invitation enough'.split()
    print(NLTKLemmatizer().lemmatize(sent,pos=True))
    print(NLTKLemmatizer().lemmatize(['was'],pos=True))
    print(NLTKLemmatizer(tagger=TaggerMixin().tagger_4ngram).lemmatize(['was'],pos=True))
    print(NLTKLemmatizer().lemmatize(['was'],pos=True,tagger=TaggerMixin().tagger_4ngram))
    
    print(list(TaggerMixin().tagger_4ngram(sent)))
    quit()
    
    print(PatternLemmatizer().lemmatize(['lotus-covered'],pos=True))
    print(PatternLemmatizer().lemmatize(['lotus-flowers'],pos=True))
    '''
    [('lotus-cover', 'VBN')]
    [('lotus-flower', 'NNS')]
    '''
    from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, ToktokTokenizer
    from nltk import word_tokenize
    
    example = 'I don’t care what it’s founded on.'
    example_en  = "I want you to explain to me why you won't exhibit Dorian Gray's picture"
    example_ru = 'В конце 1811 года, в эпоху нам достопамятную, жил в своем поместье Ненарадове добрый Гаврила Гаврилович Р **'
    
    sent  = WhitespaceTokenizer().tokenize(example_en)
    sent2 = WordPunctTokenizer().tokenize(example_en)
    sent3 = ToktokTokenizer().tokenize(example_en)
    sent4 = word_tokenize(example_en)
    
    print('--------------PATTERN-----------------------')
    print(PatternLemmatizer().lemmatize(sent,pos=True))
    print(PatternLemmatizer().lemmatize(sent2,pos=True))
    print(PatternLemmatizer().lemmatize(sent3,pos=True))
    print(PatternLemmatizer().lemmatize(sent4,pos=True))
    
    print('--------------NLTK------------------------')
    print(NLTKLemmatizer().lemmatize(sent,pos=True))
    print(NLTKLemmatizer().lemmatize(sent2,pos=True))
    print(NLTKLemmatizer().lemmatize(sent3,pos=True))
    print(NLTKLemmatizer().lemmatize(sent4,pos=True))
    
    
    print('--------------TREETAGGER------------------------')
    print(TreeTaggerLemmatizer().lemmatize(sent,pos=True))
    print(TreeTaggerLemmatizer().lemmatize(sent2,pos=True))
    print(TreeTaggerLemmatizer().lemmatize(sent3,pos=True))
    print(TreeTaggerLemmatizer().lemmatize(sent4,pos=True))
    
    
    sent  = WhitespaceTokenizer().tokenize(example_ru)
    sent2 = WordPunctTokenizer().tokenize(example_ru)
    sent3 = ToktokTokenizer().tokenize(example_ru)
    sent4 = word_tokenize(example_ru)
    
    print('--------------PymorphyLemmatizer------------------------')
    print(PymorphyLemmatizer().lemmatize(sent,pos=True)) 
    print(PymorphyLemmatizer().lemmatize(sent2,pos=True)) 
    print(PymorphyLemmatizer().lemmatize(sent3,pos=True)) 
    print(PymorphyLemmatizer().lemmatize(sent4,pos=True))         
    
    print('--------------NLTK RU------------------------') 
    # модель для русского языка только тегирует, но не лемматизирует
    print(NLTKLemmatizer(lang='rus').lemmatize(sent,pos=True))
    print(NLTKLemmatizer(lang='rus').lemmatize(sent2,pos=True))
    print(NLTKLemmatizer(lang='rus').lemmatize(sent3,pos=True))
    print(NLTKLemmatizer(lang='rus').lemmatize(sent4,pos=True))
    
    '''
    --------------PATTERN-----------------------
    [('i', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('wo', 'MD'), ('be', 'RB'), ('exhibit', 'JJ'), ('dorian', 'NNP'), ('gray', 'NNP'), ('be', 'POS'), ('picture', 'NN')]
    [('i', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('win', 'VBD'), ('have', '<UNK>'), ('t', 'NN'), ('exhibit', 'NN'), ('dorian', 'NNP'), ('gray', 'NNP'), ('have', 'POS'), ('', 'JJ'), ('picture', 'NN')]
    [('i', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('win', 'VBD'), ('have', '<UNK>'), ('t', 'NN'), ('exhibit', 'NN'), ('dorian', 'NNP'), ('gray', 'NNP'), ('have', 'POS'), ('', 'JJ'), ('picture', 'NN')]
    [('i', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('wo', 'MD'), ('be', 'RB'), ('exhibit', 'JJ'), ('dorian', 'NNP'), ('gray', 'NNP'), ('be', 'POS'), ('picture', 'NN')]
    --------------NLTK------------------------
    [('I', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ("won't", 'VBP'), ('exhibit', 'VB'), ('Dorian', 'JJ'), ("Gray's", 'NNP'), ('picture', 'NN')]
    [('I', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('won', 'VBD'), ("'", '<UNK>'), ('t', 'JJ'), ('exhibit', 'NN'), ('Dorian', 'JJ'), ('Gray', 'NNP'), ("'", 'POS'), ('s', 'JJ'), ('picture', 'NN')]
    [('I', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('won', 'VBD'), ("'", '<UNK>'), ('t', 'JJ'), ('exhibit', 'NN'), ('Dorian', 'JJ'), ('Gray', 'NNP'), ("'", 'POS'), ('s', 'JJ'), ('picture', 'NN')]
    [('I', 'PRP'), ('want', 'VBP'), ('you', 'PRP'), ('to', 'TO'), ('explain', 'VB'), ('to', 'TO'), ('me', 'PRP'), ('why', 'WRB'), ('you', 'PRP'), ('wo', 'MD'), ("n't", 'RB'), ('exhibit', 'VB'), ('Dorian', 'JJ'), ('Gray', 'NNP'), ("'s", 'POS'), ('picture', 'NN')]
    --------------TREETAGGER------------------------
    [('I', 'PP'), ('want', 'VVP'), ('you', 'PP'), ('to', 'TO'), ('explain', 'VV'), ('to', 'TO'), ('me', 'PP'), ('why', 'WRB'), ('you', 'PP'), ('wo', 'MD'), ("n't", 'RB'), ('exhibit', 'VV'), ('Dorian', 'NP'), ('Gray', 'NP'), ("'s", 'POS'), ('picture', 'NN')]
    [('I', 'PP'), ('want', 'VVP'), ('you', 'PP'), ('to', 'TO'), ('explain', 'VV'), ('to', 'TO'), ('me', 'PP'), ('why', 'WRB'), ('you', 'PP'), ('win', 'VVD'), ("'", '<UNK>'), ('t', 'NN'), ('exhibit', 'NN'), ('Dorian', 'NP'), ('Gray', 'NP'), ("'", 'POS'), ('S', 'NN'), ('picture', 'NN')]
    [('I', 'PP'), ('want', 'VVP'), ('you', 'PP'), ('to', 'TO'), ('explain', 'VV'), ('to', 'TO'), ('me', 'PP'), ('why', 'WRB'), ('you', 'PP'), ('win', 'VVD'), ("'", '<UNK>'), ('t', 'NN'), ('exhibit', 'NN'), ('Dorian', 'NP'), ('Gray', 'NP'), ("'", 'POS'), ('S', 'NN'), ('picture', 'NN')]
    [('I', 'PP'), ('want', 'VVP'), ('you', 'PP'), ('to', 'TO'), ('explain', 'VV'), ('to', 'TO'), ('me', 'PP'), ('why', 'WRB'), ('you', 'PP'), ('wo', 'MD'), ("n't", 'RB'), ('exhibit', 'VV'), ('Dorian', 'NP'), ('Gray', 'NP'), ("'", 'POS'), ('S', 'NN'), ('picture', 'NN')]
    --------------PymorphyLemmatizer------------------------
    [('в', 'PREP'), ('конец', 'NOUN'), ('1811', '<UNK>'), ('года,', '<UNK>'), ('в', 'PREP'), ('эпоха', 'NOUN'), ('мы', 'NPRO'), ('достопамятную,', '<UNK>'), ('жить', 'VERB'), ('в', 'PREP'), ('свой', 'ADJF'), ('поместье', 'NOUN'), ('ненарадов', 'NOUN'), ('добрый', 'ADJF'), ('гаврил', 'NOUN'), ('гаврилович', 'NOUN'), ('р', 'NOUN'), ('**', '<UNK>')]
    [('в', 'PREP'), ('конец', 'NOUN'), ('1811', '<UNK>'), ('год', 'NOUN'), (',', '<UNK>'), ('в', 'PREP'), ('эпоха', 'NOUN'), ('мы', 'NPRO'), ('достопамятный', 'ADJF'), (',', '<UNK>'), ('жить', 'VERB'), ('в', 'PREP'), ('свой', 'ADJF'), ('поместье', 'NOUN'), ('ненарадов', 'NOUN'), ('добрый', 'ADJF'), ('гаврил', 'NOUN'), ('гаврилович', 'NOUN'), ('р', 'NOUN'), ('**', '<UNK>')]
    [('в', 'PREP'), ('конец', 'NOUN'), ('1811', '<UNK>'), ('год', 'NOUN'), (',', '<UNK>'), ('в', 'PREP'), ('эпоха', 'NOUN'), ('мы', 'NPRO'), ('достопамятный', 'ADJF'), (',', '<UNK>'), ('жить', 'VERB'), ('в', 'PREP'), ('свой', 'ADJF'), ('поместье', 'NOUN'), ('ненарадов', 'NOUN'), ('добрый', 'ADJF'), ('гаврил', 'NOUN'), ('гаврилович', 'NOUN'), ('р', 'NOUN'), ('**', '<UNK>')]
    [('в', 'PREP'), ('конец', 'NOUN'), ('1811', '<UNK>'), ('год', 'NOUN'), (',', '<UNK>'), ('в', 'PREP'), ('эпоха', 'NOUN'), ('мы', 'NPRO'), ('достопамятный', 'ADJF'), (',', '<UNK>'), ('жить', 'VERB'), ('в', 'PREP'), ('свой', 'ADJF'), ('поместье', 'NOUN'), ('ненарадов', 'NOUN'), ('добрый', 'ADJF'), ('гаврил', 'NOUN'), ('гаврилович', 'NOUN'), ('р', 'NOUN'), ('**', '<UNK>')]
    --------------NLTK RU------------------------
    [('В', 'PR'), ('конце', 'S'), ('1811', 'NUM=ciph'), ('года,', 'S'), ('в', 'PR'), ('эпоху', 'S'), ('нам', 'S-PRO'), ('достопамятную,', 'S'), ('жил', 'V'), ('в', 'PR'), ('своем', 'S'), ('поместье', 'S'), ('Ненарадове', 'S'), ('добрый', 'A=m'), ('Гаврила', 'S'), ('Гаврилович', 'S'), ('Р', 'INIT=abbr'), ('**', 'NONLEX')]
    [('В', 'PR'), ('конце', 'S'), ('1811', 'NUM=ciph'), ('года', 'S'), (',', 'NONLEX'), ('в', 'PR'), ('эпоху', 'S'), ('нам', 'S-PRO'), ('достопамятную', 'A=f'), (',', 'NONLEX'), ('жил', 'V'), ('в', 'PR'), ('своем', 'S'), ('поместье', 'S'), ('Ненарадове', 'S'), ('добрый', 'A=m'), ('Гаврила', 'S'), ('Гаврилович', 'S'), ('Р', 'INIT=abbr'), ('**', 'NONLEX')]
    [('В', 'PR'), ('конце', 'S'), ('1811', 'NUM=ciph'), ('года', 'S'), (',', 'NONLEX'), ('в', 'PR'), ('эпоху', 'S'), ('нам', 'S-PRO'), ('достопамятную', 'A=f'), (',', 'NONLEX'), ('жил', 'V'), ('в', 'PR'), ('своем', 'S'), ('поместье', 'S'), ('Ненарадове', 'S'), ('добрый', 'A=m'), ('Гаврила', 'S'), ('Гаврилович', 'S'), ('Р', 'INIT=abbr'), ('**', 'NONLEX')]
    [('В', 'PR'), ('конце', 'S'), ('1811', 'NUM=ciph'), ('года', 'S'), (',', 'NONLEX'), ('в', 'PR'), ('эпоху', 'S'), ('нам', 'S-PRO'), ('достопамятную', 'A=f'), (',', 'NONLEX'), ('жил', 'V'), ('в', 'PR'), ('своем', 'S'), ('поместье', 'S'), ('Ненарадове', 'S'), ('добрый', 'A=m'), ('Гаврила', 'S'), ('Гаврилович', 'S'), ('Р', 'INIT=abbr'), ('**', 'NONLEX')]
    '''
