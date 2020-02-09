import math

#==============================================
class MeasuresAssociation():    
    #----------------------------------------#
    #
    #----------------------------------------#
    
    #----------------------------------------#
    # MI поднимает наверх сочетания с редкими словами (т.е чувствительна к низкочастотным словам), опечатками, иностранными словами -> необходим порог снизу по частотности
    # МI описывает предметную область: выделяются имена собственные, специальные термины,сложные номинации, отражающие предметную область
    # статистически значимыми являются биграммы со значением больше 3
    #----------------------------------------#
    '''
    @staticmethod
    
    def pearson(cn_freq,n_freq,c_freq,total):
        # c_freq  -  частота встречаемости коллоката
        # n_freq  -  частота встречаемости главного слова
        # cn_freq -  частота совместной встречаемости ключевого слова в паре с коллокатом: то есть это число повторов биграммы в списке всех биграмм
        # total   -  общее число слов в тексте
        O_f = cn_freq                         # наблюдаемая частота
        E_f = (n_freq * c_freq) / total       # ожидаемая частота ?
        #E_f = (n_freq/total) * (c_freq/total) # ожидаемая частота ?
        res = (O_f - E_f) / E_f
        
        
        res = (cn_freq * total) / (n_freq * c_freq) 
        return math.log(res,2.0)
        '''
    
    @staticmethod
    def mi(cn_freq,n_freq,c_freq,total):
        # c_freq  -  частота встречаемости коллоката
        # n_freq  -  частота встречаемости главного слова
        # cn_freq -  частота совместной встречаемости ключевого слова в паре с коллокатом: то есть это число повторов биграммы в списке всех биграмм
        # total   -  общее число слов в тексте
        res = (cn_freq * total) / (n_freq * c_freq) 
        return math.log(res,2.0)
    
    #----------------------------------------#
    # Эвристическая мера MI3 увеличивает вес частоты совместной встречаемости в числителе, что не даёт MI завышать значения для низкочастотных сочетаний. 
    # Таким образом, MI3 должна показать лучшие результаты при выявлении коллокаций на практике, чем обычная мера MI.
    #----------------------------------------#
    @staticmethod
    def mi3(cn_freq,n_freq,c_freq,total):
        res = ((cn_freq ** 3) * total) / (n_freq * c_freq) 
        return math.log(res,2.0)
    
    #----------------------------------------#
    # ? Эвристический вариант меры MI ,также обозначается как MI.log-f
    # Эта мера увеличивает вес частоты совместной встречаемости ключевого слова и коллоката по сравнению с MI. 
    # Таким образом, эффективность salience также должна быть выше, чем у MI.
    #----------------------------------------#
    @staticmethod
    def salience(cn_freq,n_freq,c_freq,total):
        res = Metrics.mi(cn_freq,n_freq,c_freq,total) * math.log(cn_freq + 1)
        return res  
    
    #----------------------------------------#
    # возможна переоценка некоторых случайных результатов, в частности, 
    # сочетаний высокочастотного элемента с низкочастотным - 
    # поэтому необходимо задавать список stop слов;
    # t-score описывает жанровые характеристики, выделяются:
    # предложные группы и обстоятельства (например, времени),
    # числа,
    # суммы,
    # общие для текстов выборки именные сочетания,
    # коллокации со служебными словами,
    # общеязыковые устойчивые сочетания

    #----------------------------------------#
    @staticmethod
    def t_score(cn_freq,n_freq,c_freq,total):
        O_f = cn_freq                         # наблюдаемая частота
        E_f = (n_freq * c_freq) / total       # ожидаемая частота ?
        #E_f = (n_freq/total) * (c_freq/total) # ожидаемая частота ?
        res = (O_f - E_f) / math.sqrt(O_f)
        return res
    
    #----------------------------------------#
    # Критерий t-student
    # 
    #----------------------------------------#
    """
    @staticmethod
    def t_student(cn_freq,n_freq,c_freq,total):
        # вероятность совместной встречаемости слов равна произведению вероятностей каждого слова в биграмме
        p_joint_occurrence  = (c_freq/total)  * (n_freq/total)
        x_mean = cn_freq/total # выборочное среднее  
        res = (x_mean - p_joint_occurrence)/math.sqrt(x_mean/total) 
        return res
    """
    
    #----------------------------------------#
    @staticmethod
    def z_score(cn_freq,n_freq,c_freq,total):
        O_f = cn_freq                     # наблюдаемая частота
        E_f = (n_freq * c_freq) / total   # ожидаемая частота
        res = (O_f - E_f) / math.sqrt(E_f)
        return res
    
    #----------------------------------------#
    # Мера Дайса не зависит от размера корпуса, она учитывает только частоту
    # совместной встречаемости и независимые частоты
    # Kак и MI, эта мера дает завышенную оценку низкочастотных словосочетаний,
    # но завышение у меры Dice гораздо менее критично, чем у меры MI.
    #----------------------------------------#
    @staticmethod
    def dice(cn_freq,n_freq,c_freq,total):
        #https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
        #https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Dice%27s_coefficient
        
        res = (2 * cn_freq) / (n_freq + c_freq) 
        return res       
    
    #----------------------------------------#    
    # нормализованный вариант меры Dice
    #----------------------------------------# 
    @staticmethod
    def log_dice(cn_freq,n_freq,c_freq,total):
        res = (2 * cn_freq) / (n_freq + c_freq) 
        return 14 + math.log(res,2.0) 
    
    #----------------------------------------#    
    # мера Жаккарда
    #----------------------------------------# 
    @staticmethod
    def jaccard(cn_freq,n_freq):
        '''Computes Jaccard similatiry coefficient'''
        #https://ru.wikipedia.org/wiki/Коэффициент_Жаккара
         
        return cn_freq / (n_freq + c_frec - cn_freq)
    
    
    #----------------------------------------#
    # minimum sensitivity – ещё одна точечная оценка силы ассоциации. 
    # Как и мера Dice, minimum sensitivity не учитывает размер корпуса (текста). 
    # Опять же, возникает возможность переоценки низкочастотных сочетаний и элементов   
    #----------------------------------------#
    @staticmethod    
    def min_sens(cn_freq,n_freq,c_freq,total):
        return min(cn_freq / n_freq, cn_freq / c_freq) 
        
 



def entropy(self, text):
        """
        Вычисляет приближенную перекрестную энтропию модели n-грамм 
        для заданного текста в форме списка строк, разделенных запятыми. 
        Это средний логарифм вероятности всех слов в тексте.
        """
        normed_text = (self._check_against_vocab(word) for word in text)
        entropy = 0.0
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            entropy += self.logscore(word, context)
            processed_ngrams += 1
        return - (entropy / processed_ngrams)

       
        
def perplexity(self, text):
    """
    Для заданного списка строк, разделенных запятыми, вычисляет
    неопределенность текста.
    """
    return pow(2.0, self.entropy(text))
        
