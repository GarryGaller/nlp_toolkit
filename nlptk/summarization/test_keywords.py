import sys
from io import StringIO
from pprint import pprint
from nltk.corpus import reuters


if __name__ == "__main__":
    if sys.version_info < (3,6):
        import win_unicode_console
        win_unicode_console.enable())
    
    from nlptk.postagging.taggers import get_tagger
    from nlptk.junk.vocabulary import Text,chardetector
    from nlptk.junk.preprocess import Preprocessor
    from nlptk.junk.configs import Config
    
    
    config = Config()
    cfg = config.to_dict() 
 
 
    print('#------Rake keywords---------------')
    from nlptk.rake.rake import Rake
    import nltk
    
    stopwords = nltk.corpus.stopwords.words('english')
    raw_text = (
        'Compatibility of systems of linear constraints over the set of '
        'natural numbers. Criteria of compatibility of a system of linear '
        'Diophantine equations, strict inequations, and nonstrict inequations '
        'are considered. Upper bounds for components of a minimal set of '
        'solutions and algorithms of construction of minimal generating sets '
        'of solutions for all types of systems are given. These criteria and '
        'the corresponding algorithms for constructing a minimal supporting '
        'set of solutions can be used in solving all the considered types of '
        'systems and systems of mixed types.'
    )     
    prp = Preprocessor(
        cfg,
        stopwords=[],   # не фильтровать по словам
        allowed_tags=[],# не фильтровать по частям речи
        #tagger=get_tagger(),
        pf={'lemmatizer':None}
    )
    '''
    prp = Preprocessor(
        cfg,
        pf={},
        replace=True
    )
    pprint(prp.preprocess_funcs)
    '''
    
    inputs = StringIO(raw_text)
    text = Text(inputs,prp,inplace=True)
    sents_tokenized = text.sents()
    rake = Rake(sents_tokenized,max_words=4,stopwords=stopwords)
    top = rake.topn(n=-1)
    
    for i,phrase in enumerate(top):
        print(i,phrase) 
   
    print('#---------------------')
    top = rake.topn(n=-1,phrase=False)
    print(top)
    print('#---------------------')
    top = rake.phrases()
    pprint(top)
    print('#---------------------')
    top = rake.get_token_weights()
    pprint(top)
    print("----degree----")
    pprint(sorted(rake.degree.items(),key=lambda t:-t[1]))
    
    quit()
    print('#------gensim keywords---------------')
    import gensim
    text = gensim.summarization.keywords(raw_text, ratio=0.8, pos_filter=[], split=True)
    pprint(text)     
