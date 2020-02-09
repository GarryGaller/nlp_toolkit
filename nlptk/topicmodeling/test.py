import sys,os
from functools import partial
from pprint import pprint
import glob

from nltk.corpus import stopwords

from nlptk.vocab.vocabulary import chardetector
from nlptk.vocab.vocabulary import Lexicon
from nlptk.misc.mixins import LemmatizerMixin
from nlptk.misc.mixins import TokenizerMixin
from nlptk.misc.mixins import SentencizerMixin
from nlptk.misc.mixins import StripperMixin
from utils import *

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel



def gensim_test(model,source):  
    
    
    '''
    mallet_path = os.path.join(APPDIR,r'models\mallet-2.0.8\bin\mallet')
    genism_lda_mallet = partial(
        LdaMallet,
        mallet_path, 
        num_topics=5
    )
    #print(mallet_path)
    '''
    model = TopicModel(
        inputs=Path(source,"*.txt"),
        stream=Stream(encoding=chardetector,sentencizer=sentencizer),
        #stream=partial(open,encoding=vocabulary)
        model=genism_lda,
        path=os.path.join(PICKLEDIR,'books'),
        tfidf=True
    )        
    
    allowed_tags = {
            'NN','NNS', # существительные, кроме имен собственных 'NNP', 'NNPS'
            'RB','RBR','RBS', # прилагательные
            'JJ','JJR''JJS',  # наречия
            #'VB','VBZ','VBP','VBD','VBN','VBG', # глаголы
            #'FW' # иностранные слова
        }
        
    disallowed_tags = set()
    STOPWORDS = stopwords.words('english')   + [
        'mr', 'mrs', 
        'st','sir', 'Miss',
        'www','htm','html',
        'shall','must'
    ]
    

    PROPER_NAMES = Lexicon(os.path.join(LEXPATH,'names')).load()
    LEXICON = Lexicon(os.path.join(LEXPATH ,'lexicon')).load()
    
    model.add_pipes([
            CharCleaner( [
                StripperMixin.strip_tags,
                StripperMixin.strip_accent,
                StripperMixin.strip_quotes,
                StripperMixin.strip_multiple_whitespaces
            ]),
            
            Tokenizer(tokenizer=tokenizer), 
            
            TokenCleaner([
                #StripperMixin.remove_quotes,
                StripperMixin.remove_trailing_chars,
                StripperMixin.remove_numeric,
                StripperMixin.remove_roman_numerals,
                partial(StripperMixin.remove_nonalphabetic),
                partial(StripperMixin.remove_short,minsize=3), 
                partial(StripperMixin.remove_stopwords,stopwords=STOPWORDS)
            ]),
            Lemmatizer(lemmatizer=lemmatizer,allowed_tags=allowed_tags),
            LemmaCleaner([
                #partial(StripperMixin.remove_if_proper_name,lexicon=PROPER_NAMES),
                #partial(StripperMixin.remove_ifnotin_lexicon, lexicons=[LEXICON]),
                #lambda lst:(s.lower() for s in lst)
                StripperMixin.make_lower
            ])       
            
        ]
    )
    model.fit()
    
    pprint(model.corpus[0][10]) 
    '''
    [(0, 1),  # id слова, частота
     (1, 1),
     (2, 1),
     (3, 1),
     (4, 1),
     (5, 1),
     (6, 2),
     (7, 1),
     (8, 1),
     (9, 1),
     (10, 16),
     ...]'''
    
    
        
    #-----------------------------------------------------
    for top in enumerate(model.print_topics()):
        pprint(top)
    
    pprint(model.word2id[0][:20])
    
    print('--------topics by documents------------')
    for idx,doc in enumerate(model.corpus):
        print(model.topic_for_doc(doc,idx)) 
    
    # отсортированное распределение тем для данного документа
    #pprint(model.get_document_topics(doc))
    
    
    '''
    #print('\nPerplexity: ', model.log_perplexity(model.corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=model, 
        texts=model.texts, 
        dictionary=model.id2word, 
        coherence='c_v'
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    '''
    
    


def sklearn_test(model,SOURCE):
    pass


    


if __name__ == "__main__":        
    '''
    import wikipedia

    global_warming = wikipedia.page("Global Warming")
    artificial_intelligence = wikipedia.page("Artificial Intelligence")
    mona_lisa = wikipedia.page("Mona Lisa")
    eiffel_tower = wikipedia.page("Eiffel Tower")
    corpus = [
        global_warming.content, 
        artificial_intelligence.content, 
        mona_lisa.content, 
        eiffel_tower.content]
    '''
    
    
    
    from gensim_model import TopicModel
    
    APPDIR = os.path.abspath(os.path.dirname(__file__))
    LEXPATH = os.path.abspath(os.path.join(APPDIR,'..','lexicons'))
    
    SOURCE = os.path.join(os.path.join(APPDIR,'..',r'CORPUS\en'))
    PICKLEDIR = os.path.join(os.path.join(APPDIR,r'DATA\books'))   
   
   
    print(LEXPATH)
    
    
    lemmatizer = LemmatizerMixin.lemmatize_nltk
    tokenizer =  TokenizerMixin.treebank_word_tokenize # wordpunct_tokenize
    sentencizer = SentencizerMixin.sentencize_nltk
    
    '''
    genism_lda = partial(
            #LdaModel,
            LdaMulticore,
            workers=4,
            num_topics=100, 
            random_state=1,
            #update_every=1,
            #chunksize=100,
            passes=20,
            #alpha='auto',
            per_word_topics=True
        )
    '''
     
    lemmatizer = LemmatizerMixin.lemmatize_ru
    sentencizer = SentencizerMixin.sentencize_nltk
    genism_lda = partial(
            LdaModel,
            num_topics=30, 
            random_state=10,
            #update_every=1,
            #chunksize=100,
            passes=10,
            #alpha='auto',
            #per_word_topics=True
        )
    
    
    SOURCE = os.path.join(os.path.join(APPDIR,'..',r'CORPUS\ru'))
    PICKLEDIR = os.path.join(os.path.join(APPDIR,r'DATA\books2')) 
    
    
    gensim_test(genism_lda,SOURCE)
    
    quit()
    
    
    sklearn_lda = partial( 
                LatentDirichletAllocation,
                n_components=10,
                max_iter=10,
                random_state=1,
                learning_method="online",
                learning_offset=10.0,
                batch_size= 128,
                n_jobs=-1,
            )
            
    sklearn_test(sklearn_lda,SOURCE)            
            
    
    
    
    
    
    

    
    
#https://fooobar.com/questions/443277/how-to-print-the-lda-topics-models-from-gensim-python 

#https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn   
