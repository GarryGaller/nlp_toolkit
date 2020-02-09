import sys,os
from typing import List
from collections import defaultdict, Counter
import heapq
from pprint import pprint


class Summarize():
    def __init__(self,text:List[List[str]],max_words_per_sent=30):
        self.text = text
        self.tokens = (token for sent in text for token in sent)
        self.token_weights = Counter(self.tokens)    
        self.sent_scores = Counter()
        self.max_words_per_sent = max_words_per_sent
        self._summarize()
    
    def _summarize(self):
        
        max_frequency = max(self.token_weights.values())
        for token in self.token_weights.keys():
            self.token_weights[token] = round(
                    self.token_weights[token]/max_frequency,
                    2
                )
        
        for idx,sent in enumerate(self.text):
            if self.max_words_per_sent != -1:
                if len(sent) > self.max_words_per_sent:
                    continue
            for token in sent:
                if token in self.token_weights:
                    self.sent_scores[idx] += self.token_weights[token]  
        
        return self.sent_scores

    
    def topn(self,n=7,sent=True):
        if sent:
            scores = self.sent_scores
        else:
            scores = self.token_weights
        if n < 0:
            n = len(scores)
        
        return heapq.nlargest(n, 
            scores, 
            key=scores.get
        )  

            
    def sents(self,scores=False):
        if scores:
            result = sorted(
                self.sent_scores.items(), 
                key=lambda t:t[1], 
                reverse=True
            )     
        
        else:
            result = sorted(
                self.sent_scores, 
                key=self.sent_scores.get, 
                reverse=True
            )
        return result 
    
    def keywords(self,scores=False):
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