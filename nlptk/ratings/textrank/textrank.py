from typing import List
from itertools import combinations
import networkx as nx
import heapq



class TextRank():

    def __init__(self, words:List[List[str]], nsents):
        self.pr = None
        self._rank(words,nsents)   
    
    
    def similarity(self,s1, s2):
        '''Мера сходства - коэффициент Сёренсена - 
        https://ru.wikipedia.org/wiki/Коэффициент_Сёренсена
        отношение количества одинаковых слов в 
        предложениях к суммарной длине предложений.
        ''' 
        if not len(s1) or not len(s2):
            return 0.0
        
        return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))


    def _rank(self, words, nsents):
        '''Простая реализация алгоритма TextRank'''

        # создаем все возможные комбинации (без повторов) из двух предложений
        pairs = combinations(range(nsents), 2)
        # вычисляем меру похожести между предложениями в паре (1s, 2s, оценка_сходства)
        scores = []
        for i, j in pairs:
            sim = self.similarity(words[i], words[j])
            # отфильтруем пары у которых похожесть равна нулю
            if sim:
                scores.append((i, j, sim)) 
        
        # создаем взвешенный граф
        g = nx.Graph()
        g.add_weighted_edges_from(scores)
        del scores
        self.pr = nx.pagerank(g)  # словарь вида {индекс_предложения: значение PageRank}
        
    
    def topn(self,n=7):        
        result = heapq.nlargest(n, 
                self.pr.items(), 
                key=lambda s:s[1])
        
        return result
        
