import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures
from nltk.text import TextCollection, Text
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from collections import defaultdict
from functools import partial
from pprint import pprint
from collections import Counter
import os
# Символы дополнения

UNKNOWN = "<UNK>"
LPAD = "<s>"
RPAD = "</s>"


def ngrams(words, n = 2):
    for idx in range(len(words)-n+1):
        yield tuple(words[idx:idx+n])


LPAD_SYMBOL = "<s>"
RPAD_SYMBOL = "</s>"

nltk_ngrams = partial(
    nltk.ngrams,
    pad_right = True, right_pad_symbol = RPAD_SYMBOL,
    pad_left = True, left_pad_symbol = LPAD_SYMBOL
)
    


texts = TextCollection(nltk.corpus.gutenberg)
print(type(texts))  # <class 'nltk.text.TextCollection'>
print(type(nltk.corpus.gutenberg)) #<class 'nltk.corpus.reader.plaintext.PlaintextCorpusReader'
#pprint(dir(nltk.corpus.gutenberg))
'''[ 
 'abspath',
 'abspaths',
 'citation',
 'encoding',
 'ensure_loaded'
 'fileids',
 'license',
 'open',
 'paras',
 'raw',
 'readme',
 'root',
 'sents',
 'unicode_repr',
 'words']
''' 




#pprint(dir(texts))
'''
[ 'collocation_list',
 'collocations',
 'common_contexts',
 'concordance',
 'concordance_list',
 'count',
 'dispersion_plot',
 'findall',
 'generate',
 'idf',
 'index',
 'name',
 'plot',
 'readability',
 'similar',
 'tf',
 'tf_idf',
 'tokens',
 'unicode_repr',
 'vocab']
'''


'''
for text in texts: # итератор по всем словам из текстов корпуса
    print(text) #
    break
'''


moby = Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
print(moby)  #<Text: Moby Dick by Herman Melville 1851>
#print(list(moby)) # список слова
#pprint(dir(moby))
'''
['collocation_list'
'collocations',
'common_contexts',
'concordance',
'concordance_list'
'count',
'dispersion_plot',
'findall',
'generate',
'index',
'name',
'plot',
'readability',
'similar',
'tokens',
'unicode_repr',
'vocab']'''


#pprint(moby.collocation_list())
['Sperm Whale',
 'Moby Dick',
 'White Whale',
 'old man',
 'Captain Ahab',
 'sperm whale',
 'Right Whale',
 'Captain Peleg',
 'New Bedford',
 'Cape Horn',
 'cried Ahab',
 'years ago',
 'lower jaw',
 'never mind',
 'Father Mapple',
 'cried Stubb',
 'chief mate',
 'white whale',
 'ivory leg',
 'one hand']



#list(nltk.skipgrams(sent, 2, 2))

  
    
    
'''    
    def ngrams(self, n = 2, fileids = None, categories = None):
        for sent in self.sents(fileids = fileids, categories = categories):
            for ngram in nltk_ngrams(sent, n):
                yield ngram

'''


def rank_quadgrams(corpus, metric):
    """
    Находит и оценивает тетраграммы в указанном корпусе с применением
    заданной метрики. Записывает тетраграммы в файл, если указан,
    иначе возвращает список в памяти.
    """
    # Создать объект оценки словосочетаний из слов в корпусе.
    ngrams = QuadgramCollocationFinder.from_words(corpus.words())
    # Оценить словосочетания в соответствии с заданной метрикой
    scored = ngrams.score_ngrams(metric)
    return scored



class TextCorpus(PlaintextCorpusReader):
    
    def __init__(self,args,**kwargs):
        super().__init__(args,**kwargs)
   



class NgramCounter(object):
    """
    Класс NgramCounter подсчитывает n-граммы для заданного словаря и размера  
    окна.
    """

    def __init__(self, n, vocabulary, unknown = UNKNOWN):
        """
        n -- размер n-грамм
        """
        if n < 1:
            raise ValueError("ngram size must be greater than or equal to 1")
        self.n = n
        self.unknown = unknown
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": LPAD,
            "right_pad_symbol": RPAD,
        }
        self.vocabulary = vocabulary
        self.allgrams = defaultdict(ConditionalFreqDist)
        self.ngrams = FreqDist()
        self.unigrams = FreqDist()

    '''
    Далее добавим в класс NgramCounter метод, который позволит систематически 
    вычислять распределение частот и условное распределение частот для задан­
    ного размера n­грамм.
    '''

    def train_counts(self, training_text):
        for sent in training_text:
            checked_sent = (self.check_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                self.ngrams[ngram] += 1
                context, word = tuple(ngram[:-1]), ngram[-1]
                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False
                for window, ngram_order in enumerate(range(self.n, 1, -1)):
                    context = context[window:]
                    self.allgrams[ngram_order][context][word] += 1
                self.unigrams[word] += 1
    
    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unknown

    def to_ngrams(self, sequence):
        """
        Обертка для метода ngrams из библиотеки NLTK
        """
        return nltk_ngrams(sequence, self.n, **self.padding)

'''
Модели языка n-грамм 
Теперь определим функцию (за пределами класса NgramCounter), которая создает 
экземпляр класса и вычисляет релевантные частоты. Функция count_ngrams 
принимает параметр с желаемым размером n­грамм, словарь и список пред­
ложений в виде строк, разделенных запятыми.
'''

def count_ngrams(n, vocabulary, texts):
    counter = NgramCounter(n, vocabulary)
    counter.train_counts(texts)
    return counter


if __name__ == '__main__':
    #from nltk.book import text1, text2, text3,text4,text5   
    
    #print(type(text1))
    #print(dir(text1))
    
    
    #corpus = PickledCorpusReader('../corpus')
    #corpus = nltk.corpus.gutenberg
    #mytexts  = TextCollection([text1])
    
    APPDIR = os.path.dirname(__file__)
    corpus_root = 'D:\\INSTALL\\Python3\\PROJECTS\\SCRIPTS\\TEXTS\\corpora\\'
    corpus = PlaintextCorpusReader(corpus_root, '*.txt')
    print(corpus.words())
    
    
    pprint(rank_quadgrams(corpus, QuadgramAssocMeasures.likelihood_ratio))
   

    
    tokens = [''.join(word[0]) for word in corpus.words()]
    vocab = Counter(tokens)
    sents = list([word[0] for word in sent] for sent in corpus.sents())
    trigram_counts = count_ngrams(3, vocab, sents)


    #Распределение частоты для униграмм можно получить из атрибута unigrams.
    print(trigram_counts.unigrams)
    #Для n­грамм более высокого порядка условное распределение частот можно 
    #получить из атрибута ngrams.
    print(trigram_counts.ngrams[3])    # <FreqDist with 88 samples and 3015993 outcomes>
    
    #Ключи условного распределения частот показывают возможные контексты, 
    #предшествующие каждому слову.
    #print(sorted(trigram_counts.ngrams[3].conditions())) # неверно
    #Наша модель также способна возвращать список возможных следующих слов:
    print(list(trigram_counts.ngrams[3][('the', 'President')]))
