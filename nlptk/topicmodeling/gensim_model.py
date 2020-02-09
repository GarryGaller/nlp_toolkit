import sys
from gensim.corpora import Dictionary
from gensim.corpora import mmcorpus 
from gensim.models import TfidfModel

import os,time,curses
import typing

if sys.version_info < (3,6):
    import win_unicode_console
    win_unicode_console.enable())


def gettime():
    return time.strftime("%d.%m.%Y %H:%M:%S",time.localtime())

class TopicModel():

    def __init__(self,
              inputs,
              stream=None, 
              model=None,
              path=None,
              tfidf=False,
              verbose=True
        ):
        self.inputs = inputs
        self.stream = stream
        self.model = model
        self.pipes = []    
        self.corpus = []
        self.path = path
        self.tfidf = tfidf
        self.verbose = verbose
        self.id_to_path = []
        
    def add_pipes(self,pipes):
        self.pipes.extend(pipes)
     
    
    def get_texts(self):
        
        if self.verbose:
            cnt = 0
            stdscr = curses.initscr()
            stdscr.clear()
            stdscr.refresh()
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            stdscr.addstr(0,0,"СТАРТ: {}".format(gettime()),curses.color_pair(2))
            stdscr.refresh()
        
        texts = []
        
    
        
        for ix,path in enumerate(self.inputs): 
            text = []
            data = None
            basename = os.path.basename(path)
            if self.verbose:
                s = '{} {} {}'.format(ix,basename,gettime())
                stdscr.addstr(1,0,s,curses.color_pair(1))
                stdscr.refresh()
            
            for iy,sent in enumerate(self.stream(path)):
                data = sent
                for pipe in self.pipes:
                    
                    if self.verbose:
                        s = '===< {} >===TIME:{}'.format(
                            pipe.__class__.__name__,gettime()
                        )
                        stdscr.addstr(3,0,s,curses.color_pair(1))
                        stdscr.refresh()
                    data = (
                        list(pipe(data)) 
                        if isinstance(data, typing.Generator) 
                        else pipe(data)
                    )
                        
                    if self.verbose:
                        s = "{} {}".format(iy,str(data)[:100])
                        stdscr.addstr(4,0,s,curses.color_pair(1))
                        stdscr.refresh()
                text.extend(data)
                
            if self.verbose:
                s = '{} {} {}'.format(basename,len(text),gettime())
                stdscr.addstr(2,0,s,curses.color_pair(2))
                stdscr.refresh()
                #cnt+=5
            
            texts.append(text)
            self.id_to_path.append(basename)
            
        #curses.endwin()
        return texts    
            
    def fit(self,verbose=True):
        self.verbose = verbose
        # Term Document Frequency
        self.id2word = self.get_dict(self.path)
        # Create Corpus
        self.corpus = self.get_corpus(self.path) 
        if self.tfidf:
            self.tfidf_model = self.get_tfidf(self.path)
        # Create Model
        self.model = self.get_model(self.path)
        
        
    def get_dict(self,path):
        path = path + '.dict'   
        if not os.path.exists(path):
            self.texts = self.get_texts()
            dct = Dictionary(self.texts)
            dct.save_as_text(path)
        else:
            dct = Dictionary()
            dct = dct.load_from_text(path)
            for path in self.inputs:
                self.id_to_path.append(os.path.basename(path))
        return dct
        
    
    def get_corpus(self,path):
        path = path + '.corpus'   
        if not os.path.exists(path):
            corpus = [self.id2word.doc2bow(text,allow_update=True) 
                for text in self.texts 
            ]
            # save
            mmcorpus.MmCorpus.serialize(path,corpus)
           
        else:
            # load
            corpus = mmcorpus.MmCorpus(path)
        return corpus    
    
    
    def get_tfidf(self,path):
        path = path + '.tfidf'   
        if not os.path.exists(path):
            tfidf_model = TfidfModel(self.corpus, smartirs='ntc')
            tfidf_model.save(path) 
            # перевзвешивание корпуса
            self.corpus = tfidf_model[self.corpus]
        else:
             tfidf_model = TfidfModel.load(path)
        return tfidf_model    
    
    
    def get_model(self,path,load=False):
        path = path + '.model' 
        if load:
            if os.path.exists(path):
                model = self.model.load(path, mmap='r')
            else:
                raise Exception('Model not found')
        else:        
            model = self.model(corpus=self.corpus, id2word=self.id2word)
            model.save(path)
        
        return model    
    
    @property
    def word2id(self):
        '''Отсортированное отображение слов на частоту'''
        _ = [
            [(self.id2word[id], freq) for id, freq in cp] 
            for cp in self.corpus
        ]
        return sorted(_,key=lambda t:-t[1])
                                        
    
    def print_topics(self):
        return self.model.print_topics() 
        
    def get_document_topics(self,doc):
        return sorted(
            self.model.get_document_topics(doc),
            key=lambda x:-x[1]
        ) 

    def get_topic_terms(self,num_topic): 
        return self.model.get_topic_terms(num_topic)
    
    
    def topic_for_doc(self,doc,idx):
        
        if isinstance(idx,int):
            path = self.id_to_path[idx]
        
        topics_for_doc = self.get_document_topics(doc)
        # выбираем самый  вероятный топик
        num_topic_for_doc = topics_for_doc[0][0] 
        # получаем термы для данного топика
        terms = self.get_topic_terms(num_topic_for_doc)
        # заменяем номера термов на слова
        return (path,[self.id2word[term[0]] for term in terms])
          
        
        
        
    
    
  
