from itertools import combinations
import networkx as nx



def similarity(s1, s2):
    '''Мера сходства - коэффициент Сёренсена - 
    https://ru.wikipedia.org/wiki/Коэффициент_Сёренсена
    отношение количества одинаковых слов в 
    предложениях к суммарной длине предложений.
    ''' 
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))


def textrank(words,sentences,sort_by_pr=True):
    '''Простая реализация алгоритма TextRank'''

    # создаем все возможные комбинации (без повторов) из двух предложений
    pairs = combinations(range(len(sentences)), 2)
    # вычисляем меру похожести между предложениями в паре (1s, 2s, оценка_сходства)
    scores = [
        (i, j, similarity(words[i], words[j])) 
        for i, j in pairs
    ]
    
    # отфильтруем пары у которых похожесть равна нулю
    scores = list(filter(lambda x: x[2], scores))
    
    # создаем взвешенный граф
    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    pr = nx.pagerank(g)  # словарь вида {индекс_предложения: значение PageRank}
    
    # получаем кортежи данных вида (индекс_предложения, оценка, предложение)
    data = (
        (idx, pr[idx], s) 
        for idx, s in enumerate(sentences) if idx in pr
    )
    # сортируем по убыванию оценки
    #return sorted(data,key=lambda x: pr[x[0]], reverse=True)
    if sort_by_pr:
        data = sorted(data,key=lambda x: x[1], reverse=True)
    return data, scores, pr
