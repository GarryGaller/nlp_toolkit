from collections import Counter
from nlptk.measures.metrics import MeasuresAssociation

class Collocate():
    
    method = {
            'mi':MeasuresAssociation.mi,
            'student':MeasuresAssociation.t_score,
            't-score':MeasuresAssociation.t_score,
            'z_score':MeasuresAssociation.z_score,
            'dice':MeasuresAssociation.dice,
            'logdice':MeasuresAssociation.log_dice,
            'log_dice':MeasuresAssociation.log_dice,
            'mi3':MeasuresAssociation.mi3,
            'salience':MeasuresAssociation.salience,
            'min_sens':MeasuresAssociation.min_sens
            #'likelihood':MeasuresAssociation.likelihood
    }
    
    def __init__(self):
        pass
    
    def collocations(self,tokens,bigrams,measure='student'):
        # словарь методов реализующих меры ассоциаций
        
        
        self.dict_bigrams = Counter(bigrams)           # частотный словарь биграмм
        dict_freq = self.freq(tokens)                  # частотный словарь (Counter) всех слов в выборке
        total = len(tokens)                            # общее число всех слов в выборке
        collocat_dict = dict()                         # словарь биграмм\коллокаций, где каждому ключу присваивается значение меры ассоциации в зависимости от метода
        bigrams = set(bigrams)                         # уникализируем список биграмм
        
        for bigram in bigrams:
            n_req  = dict_freq.get(bigram[0],0)        # частота ключевого слово с биграмме
            c_freq = dict_freq.get(bigram[1],0)        # частота коллоката в биграмме
            cn_freq = self.dict_bigrams.get(bigram)    # частота самой биграммы
            collocat_dict[bigram] = type(self).method[measure](cn_freq,n_req,c_freq,total)
        
        return Counter(collocat_dict)