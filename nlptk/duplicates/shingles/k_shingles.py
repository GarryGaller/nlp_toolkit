# Сегалович, Зеленков Сравнительный анализ методов определения нечетких дубликатов для Web-документов 
# http://rcdl2007.pereslavl.ru/papers/paper_65_v1.pdf


from pprint import pprint
from collections import Counter
import binascii
import math

def compare_dice(a,b):
    '''мера Дайса 2nt/na + nb.'''
    a = set(a)
    b = set(b)
    common = a & b
    dice = (len(common) * 2.0)/(len(a) + len(b))
    return  dice * 100  # переводим меру в процентное отношение


def compare_equal(a,b):
    return a == b



def calc_tfidf(tokens,corpus):
    
    tf_dict = Counter(tokens)
    
    dl = len(tokens)
    n_samples = len(corpus)
    dl_avg = sum(map(len,corpus))/n_samples
    for term in tf_dict:
        #для каждого слова считаем TF путём деления
        # встречаемости слова на общее количество слов в тексте
        #tf_dict[w] /= dl
        # или  по формуле из статьи Сегаловича
        tf_dict[term] /= 2 * (0.25 + 0.75 * (dl / dl_avg )) + tf_dict[term] 
        
    
    for term in tf_dict:
        df = sum(1.0 for tokens in corpus if term in tokens)  or 1.0     
        #idf = math.log(n_samples / df ) + 1 
        idf = math.log((n_samples - df + 0.5) / (df + 0.5))
        tf_dict[term] *= idf
    
    return tf_dict

        
def k_shingles(tokens,k=2):
    """
    Генератор шинглов указанный длины.
    
    >>> text = '''Белеет парус одинокой В тумане моря голубом!.. 
    Что ищет он в стране далекой? Что кинул он в краю родном?'''
    
    >>> list(k_shingles(text.split(),k=3)))
    ['Белеет парус одинокой',
     'парус одинокой В',
     'одинокой В тумане',
     'В тумане моря',
     'тумане моря голубом!..',
     'моря голубом!.. Что',
     'голубом!.. Что ищет',
     'Что ищет он',
     'ищет он в',
     'он в стране',
     'в стране далекой?',
     'стране далекой? Что',
     'далекой? Что кинул',
     'Что кинул он',
     'кинул он в',
     'он в краю',
     'в краю родном?']
    """
    for i in range(len(tokens) - (k-1)):
        yield ' '.join(tokens[i:i + k])
    

def k_shingles_hashing(tokens,k):
    '''шинглирование + хэширование каждого шингла'''
    for shingle in k_shingles(tokens,k=k):
        yield binascii.crc32(shingle.encode('utf-8'))    


def shingle_log(tokens,k=5):
    for shingle in k_shingles_hashing(tokens,k=k):
        if shingle % 4 == 0:
            yield shingle


def shingle_long_sent(sents):
    '''Вычисление сигнатуры от двух наиболее длинных (по числу слов) предложений'''
    
    sents = sorted(sents,
        key=lambda x:len(x.split()),
        reverse=True
    )
    shingle = ' '.join(sents[:2])
    
    return binascii.crc32(shingle.encode('utf-8'))   
 

def shingle_heavy_sent(sents,corpus):
    '''Вычисление сигнатуры от двух наиболее тяжелых по весу предложений'''
    wt_list = []
    tokens = []
    for sent in sents:
        tokens.extend(sent.split())
        
    tf = calc_tfidf(tokens,corpus)
    calc_wt = lambda tok: tf[tok]
    
    for idx,sent in enumerate(sents):
        wt_list.append((
            idx,sum(map(calc_wt,sent.split())))
        )
        
    wt_list.sort(key=lambda x:x[1],reverse=True)    
    
    shingle = ' '.join(sents[:2])
    
    return binascii.crc32(shingle.encode('utf-8'))    


def shingle_tf(tokens):
    '''Вычисление сигнатуры от шести наиболее тяжелых по весу TF слов'''
    tf = Counter(tokens)
    
    length = len(tokens)
    for term in tf:
        #для каждого слова считаем tf путём деления
        #встречаемости слова на общее количество слов в тексте
         tf[term] /= length
    
    top =  [k for k,v in tf.most_common(6)]
    shingle = ' '.join(top)
    
    return binascii.crc32(shingle.encode('utf-8')) 
   
    
def shingle_tfidf(tokens,corpus):
    '''Вычисление сигнатуры от шести наиболее тяжелых по весу TF-IDF слов'''
    
    tf = calc_tfidf(tokens,corpus)
    
    top =  [k for k,v in tf.most_common(6)]
    shingle = ' '.join(top)
    
    return binascii.crc32(shingle.encode('utf-8'))


def shingle_opt_freq(tokens,corpus):
    '''Вычисление сигнатуры от шести наиболее тяжелых по весу TF-IDF_opt слов'''
    
    tf = Counter(tokens)
    tf_max = tf.most_common(1)[0][1]
    dl = len(tokens)
    n_samples = len(corpus)
    dl_avg = sum(map(len,corpus)) / n_samples
    
    for term in tf:
        #для каждого слова считаем TF по формуле из статьи Сегаловича
        tf[term] = 0.5 + 0.5 * tf[term] / tf_max
        
    
    for term in tf:
        df = sum(1.0 for tokens in corpus if term in tokens)  or 1.0     
        idf = -math.log(df / n_samples)
        if idf < 11.5:
            idf_opt = math.sqrt(idf / 11.5)
        else:
            idf_opt = 11.5 / idf
        
        tf[term] *= idf_opt
    
    
    top =  [k for k,v in tf.most_common(6)]
    shingle = ' '.join(top)
    
    return binascii.crc32(shingle.encode('utf-8'))



if __name__ == "__main__":
    pass
    
   
    
    
    
        
    
    
    
    
    
    
    
    

    
    
                      
        
