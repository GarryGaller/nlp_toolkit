import os,sys
import pickle
from pprint import pprint
import nltk
from nltk.corpus import brown
from nltk.tag import _get_tagger
from nltk.corpus import brown
from nltk import NgramTagger
from nltk import TrigramTagger
from nltk import BigramTagger
from nltk import UnigramTagger
from nltk import RegexpTagger
from nltk import DefaultTagger
from nltk.tag import HunposTagger # exe
from nltk.tag import SennaTagger  # exe
from nltk.tag import CRFTagger
from nltk.tag import BrillTagger
from nltk.tag import HiddenMarkovModelTagger


PATTERNS = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN')                      # nouns (default)
]


def create_tagger(sents,patterns=PATTERNS,maxngram=4):
    '''Обучение Backoff tagger на каком-либо корпусе предложений'''
    
    train = sents
    def_tagger = DefaultTagger('NN')
    re_tagger = RegexpTagger(patterns, backoff=def_tagger)
    uni_tagger = UnigramTagger(train, backoff=re_tagger) 
    bi_tagger = BigramTagger(train, backoff=uni_tagger) 
    tri_tagger = TrigramTagger(train, backoff=bi_tagger) 
    ngram_tagger = NgramTagger(maxngram, train, backoff=tri_tagger)
    return ngram_tagger


def get_tagger(path='data/4-ngram_tagger.pickle',sents=None,ret='func'):
    '''4-ngram_tagger.pickle - тэггер обученный на брауновском корпусе'''
    
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__),'data',path + '.pickle')
    
    if not os.path.exists(path): 
        if sents is None: 
            raise ValueError('Parameter sent not passed')
        tagger = create_tagger(sents)
        with open(path, 'wb') as f:
            pickle.dump(tagger, f)
    else:
        with open(path, 'rb') as f:
            tagger = pickle.load(f)
    if ret == "func":
        tagger = tagger.tag
    elif ret == 'class':
        tagger = tagger    
    
    return tagger


#--------------------------------------------

class Hunpos():
    '''HunposTagger'''
    
    def __init__(self,
                path_to_bin='hunpos',
                path_to_model='en_wsj.model', 
                **kwargs):   
        
        self.__dict__.update(kwargs)
        
        if not os.path.isabs(path_to_bin):
            path_to_bin = os.path.join(sys.exec_prefix,
                r'lib\site-packages', path_to_bin
            )
        
        if not os.path.isabs(path_to_model):
            path_to_model = os.path.join(path_to_bin,
                r'en_wsj.model',path_to_model
            )
        
        self.tagger = HunposTagger(
            path_to_model, path_to_bin, **kwargs
            ).tag
    
    def __call__(self,tokens):
        return  self.tagger(tokens)  


class Senna():
    '''SennaTagger'''
    
    def __init__(self,
                path='senna',
                **kwargs):   
        
        self.__dict__.update(kwargs)
        
        if not os.path.isabs(path_):
            path_to_bin = os.path.join(sys.exec_prefix,
                r'lib\site-packages', path
            )
        
        self.tagger = SennaTagger(path, **kwargs).tag
    
    def __call__(self,tokens):
        return  self.tagger(tokens)  





if __name__ == "__main__":
    APPDIR = os.path.dirname(__file__)
    path = os.path.join(APPDIR,'data','4-ngram_tagger.pickle')

    tagged_sents = brown.tagged_sents()
    i = int(len(tagged_sents) * 0.9)
    train = tagged_sents[:i]
    test = tagged_sents[i:]
    
    backoff_tagger = get_tagger(path,train)
    #print(backoff_tagger(''))    # []
    #print(backoff_tagger(['']))  # [('', 'NN')]
    print(backoff_tagger(['was']))    # [('was', 'BEDZ')]  ???
    print(backoff_tagger(['a']))      # [('a', 'AT')]
    print(backoff_tagger(['some']))   # [('some', 'DTI')]
    print(backoff_tagger(['no']))     # [('no', 'AT')]
    print(nltk.pos_tag(['was']))      # [('was', 'VBD')]
    print(nltk.pos_tag(['a']))        # [('a', 'DT')]
    print(nltk.pos_tag(['some']))     # [('some', 'DT')]
    print(nltk.pos_tag(['no']))       # [('no', 'DT')]
    
    sent = '''Emma Woodhouse, handsome, clever, and rich, with a comfortable home
and happy disposition, seemed to unite some of the best blessings
of existence; and had lived nearly twenty-one years in the world
with very little to distress or vex her.'''
    
    sent2 = "The quick brown fox jumps over the lazy dog"
    
    tokens = nltk.word_tokenize(sent2)
   #-----------------------------------------
    text = nltk.word_tokenize()
    #pprint(nltk.pos_tag(tokens))
    [('The', 'DT'),    
     ('quick', 'JJ'),
     ('brown', 'NN'),  # ошибка
     ('fox', 'NN'),
     ('jumps', 'VBZ'),
     ('over', 'IN'),
     ('the', 'DT'),    
     ('lazy', 'JJ'),
     ('dog', 'NN')]
    
    #pprint(backoff_tagger.tag(tokens))
    
    [('The', 'AT'),
     ('quick', 'JJ'),
     ('brown', 'JJ'),
     ('fox', 'NN'),
     ('jumps', 'VBZ'),
     ('over', 'IN'),
     ('the', 'AT'),
     ('lazy', 'JJ'),
     ('dog', 'NN')]
    
    backoff_tagger = get_tagger(path,ret='class')
    print(backoff_tagger.evaluate(test))   # 0.9181597200393956
    
    # неверное сравнение так как системы тегирования отличаются 
    # у nltk_tag и тэггера обученного на корпусе Брауна
    standart_tagger = _get_tagger()   # загружает nltk.tag.PerceptronTagger с английской моделью
    print(standart_tagger.evaluate(test))  # 0.6028582804216173
    #--------------------------------
    # оценки с помощью sklearn
    from sklearn import metrics
    
    
    untagged_sentences = [[token for token,_ in sent] for sent in test]
    tagged_test_sentences = nltk.pos_tag_sents(untagged_sentences)
    gold = [str(tag) for sentence in test for _,tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences 
                        for _,tag in sentence]
    
    '''
    print(metrics.classification_report(gold, pred))
    micro avg       0.60      0.60      0.60     95442
    macro avg       0.10      0.11      0.10     95442
    weighted avg    0.54      0.60      0.57     95442
    
    '''
     

    
    untagged_sentences = [[token for token,_ in sent] for sent in test]
    tagged_test_sentences = tagger.tag_sents(untagged_sentences)
    gold = [str(tag) for sentence in test for _,tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences 
                        for _,tag in sentence]
    
    print(metrics.classification_report(gold, pred))
    '''
    micro avg       0.92      0.92      0.92     95442
    macro avg       0.54      0.51      0.51     95442
    weighted avg    0.92      0.92      0.92     95442
    '''
    
    





'''
def_tagger = DefaultTagger('NN')
re_tagger = RegexpTagger(PATTERNS, backoff=def_tagger)
uni_tagger = UnigramTagger(train, backoff=re_tagger) 
bi_tagger = BigramTagger(train, backoff=uni_tagger) 
tri_tagger = TrigramTagger(train, backoff=bi_tagger) 
ngram_tagger = NgramTagger(4, train, backoff=tri_tagger)


print(def_tagger.tag(test_sent))
print(re_tagger.tag(test_sent))
print(uni_tagger.tag(test_sent))
print(bi_tagger.tag(test_sent))
print(tri_tagger.tag(test_sent))
print(ngram_tagger.tag(test_sent))

[('hi', 'NN'), ('how', 'NN'), ('are', 'NN'), ('you', 'NN')]
[('hi', 'NN'), ('how', 'NN'), ('are', 'NN'), ('you', 'NN')]
[('hi', 'NN'), ('how', 'WRB'), ('are', 'BER'), ('you', 'PPSS')]
[('hi', 'NN'), ('how', 'WRB'), ('are', 'BER'), ('you', 'PPSS')]
[('hi', 'NN'), ('how', 'WRB'), ('are', 'BER'), ('you', 'PPSS')]
[('hi', 'NN'), ('how', 'WRB'), ('are', 'BER'), ('you', 'PPSS')]
'''
#------------------------------------------

'''
for n in range(1,4):
    tagger = nltk.NgramTagger(n, train, backoff=default_tagger)
    print(n,tagger.tag('hi how are you'.split()))  
'''    
'''    
1 [('hi', 'NN'), ('how', 'WRB'), ('are', 'BER'), ('you', 'PPSS')]
2 [('hi', 'NN'), ('how', 'WRB'), ('are', 'NN'), ('you', 'PPSS')]
3 [('hi', 'NN'), ('how', 'NN'), ('are', 'BER'), ('you', 'NN')]    
'''
