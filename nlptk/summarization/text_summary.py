import sys,os
from pprint import pprint
from io import StringIO
from pprint import pprint
from nltk.corpus import reuters





if __name__ == "__main__":
    

    from nlptk.summarization.summarizer import Summarize
    from nlptk.postagging.taggers import get_tagger
    from nlptk.junk.vocabulary import Text,chardetector
    from nlptk.junk.preprocess import Preprocessor
    from nlptk.junk.configs import Config
    
    
    config = Config()
    cfg = config.to_dict()
    
    prp = Preprocessor(cfg)  # disallowed_tags={'NNP','NNPS'}
    #print(prp.allowed_tags)
    inputs = os.path.join(config.SOURCEDIR, 
        "txt\Edgar Allan Poe The Cask of Amontillado.txt")
    inpust = os.path.join(config.SOURCEDIR, 
        "txt\Edgar Allan Poe The Masque of the Red Death.txt")
    
    fi = reuters.fileids()[0]
    #fd = nltk.corpus.reuters.open(fi) 
    # fd.name  полный путь до файла
    raw_text = reuters.raw(fi)
    inputs = StringIO(raw_text)
    text = Text(inputs,prp,inplace=True)
    sents_tokenized = text.sents()
    sents_raw = text.sents(raw_sents=True)
    
    sm = Summarize(sents_tokenized)
    top = sm.topn()
    
    sents = [sents_raw[idx] for idx in top ]
    for i,sent in enumerate(sents[:7]):
        print(i,sent)    
    
    
    print('#-------gensim--------------')
    
    import gensim
    #raw_text = open(filepath,encoding=chardetector(filepath)).read()
    
    sents = gensim.summarization.summarize(raw_text,split=True)
    print(sents)
    for i,sent in enumerate(sents[:7]):
        print(i,sent)   
    
    
   
