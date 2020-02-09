import sys,os
from typing import List
from collections import defaultdict, Counter
from itertools import groupby, chain, product
import heapq
from pprint import pprint
import string 

class Rake():
    def __init__(
            self,
            text:List[List[str]], 
            stopwords=[],
            max_words=100, 
            min_chars=3
        ):
        
        self.text = text
        self.stopwords = stopwords
        self.blacklist = set(chain(stopwords, string.punctuation))
        self._phrases = set()
        # Частота (freq(w)) определяется как количество фраз, 
        # в которые входит рассматриваемое слово
        self.freq = Counter()
        # Степень (deg(w)) определяется как суммарное количество слов, 
        # из которых состоят фразы, в которых оно содержится.
        self.degree = Counter()
        # Вес слова определим как отношение степени слова к его частоте: 
        # s(w) = deg(w)/freq(w)
        self.token_weights = Counter()
        self.phrase_scores = Counter()
        self.min_chars = min_chars
        self.max_words = max_words
      
        self._generate_phrases()
        self._calc_frequencies()
        self._calc_weights()
        self._calc_scores()
      
        
    def _generate_phrases(self):
        '''Create contender phrases from sentences.'''
        
        for sent in self.text:
            self._phrases.update(self._get_phrase_list(sent))
        
    
    def _get_phrase_list(self,sent):
        '''Grouping the left words into phrases'''
        
        groups = groupby(sent, lambda x: x not in self.blacklist)
        phrases = [tuple(group[1]) for group in groups if group[0]]
        result = []
        
        for phrase in phrases:
            if (
                    phrase
                    and len(' '.join(phrase)) >= self.min_chars
                    and len(phrase) <= self.max_words
            ):
                result.append(phrase)
        #print('_get_phrase_list')
        #pprint(result)
        return result
    
    
    
    def _calc_frequencies(self):
        '''Calculation of frequencies of words'''
        for phrase in self._phrases:
            for token in phrase:
                self.freq[token] += 1
                self.degree[token] += len(phrase) - 1 # 1 вычитается не везде; смысл?
        
        # не во всех примерах Rake используется добавление частоты к degree ; смысл?
        for token in self.freq:
            self.degree[token] += self.freq[token]
        
    
    def _calc_frequencies2(self):
        
        self.freq = Counter(chain.from_iterable(self._phrases))
        def build_occurance_graph():       
        
            graph = defaultdict(lambda: defaultdict(int))
            for phrase in self._phrases:
                # For each phrase in the phrase list, count co-occurances of the
                # word with other words in the phrase.
                #
                # Note: Keep the co-occurances graph as is, to help facilitate its
                # use in other creative ways if required later.
                for (word, coword) in product(phrase, phrase):
                    graph[word][coword] += 1                                
            return graph
        
        graph = build_occurance_graph()
        self.degree = defaultdict(int)
        for token in graph:
            self.degree[token] = sum(graph[token].values())
        
        pprint(graph )
      
    
    def _calc_weights(self):     
        # веса слов s(w) = deg(w)/freq(w)
        for token in self.freq:
            score = self.degree[token] / (self.freq[token] * 1.0)
            self.token_weights[token] += score 
    
   
    def _calc_scores(self):
        
        for phrase in self._phrases:
            #print(phrase,self._phrases.count(phrase))
            score = sum(self.token_weights.get(token,0) for token in phrase)
            self.phrase_scores[' '.join(phrase)] += score  


    def topn(self,n=7,phrase=True):
        '''Get top phrases with ratings'''
        
        if phrase:
            scores = self.phrase_scores
        else:
            scores = self.token_weights
        if n < 0:
            n = len(scores)
        
        return heapq.nlargest(n, 
            scores, 
            key=scores.get
        )     
        
    def phrases(self,scores=True):
        if scores:
            result = sorted(
                self.phrase_scores.items(), 
                key=lambda t:t[1], 
                reverse=True
            )     
        
        else:
            result = sorted(
                self.phrase_scores, 
                key=self.phrase_scores.get, 
                reverse=True
            )
        return result 
    
    def get_token_weights(self,scores=True):
        if scores:
            result = sorted(
                self.token_weights.items(), 
                key=lambda t:t[1], 
                reverse=True
            )     
        
        else:
            result = sorted(
                self.token_weights, 
                key=self.token_weights.get, 
                reverse=True
            )
        return result 
         
        
        
