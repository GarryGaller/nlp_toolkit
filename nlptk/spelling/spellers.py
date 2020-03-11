import os, sys
from functools import partial
from hunspell  import Hunspell
import enchant
from nltk.metrics import edit_distance



class Speller():
    
    def __init__(self, backend='CyHunspell', lang='en', **kwargs):
        main = sys.modules[__name__]
        if hasattr(main,backend): 
            self.speller = getattr(main,backend)(lang=lang,**kwargs)
        else:
            raise Exception("Backend <%s> not found" % backend)
     
    def __call__(self):
        return self.speller
       
        

class CyHunspell():
    
    '''
    Спеллер на основе cython версии hunspell
    
    >>> word_en = 'cookbok'
    >>> word_ru = 'поваринная'
    >>> speller_en = CyHunspell(lang="en")
    >>> speller_en.spell(word_en)
    False
    >>> speller_en.suggest(word_en)
    ('cookbook', 'copybook', 'codebook', 'Cook', 'cook')
    >>> speller_en.replace(word_en)
    'cookbook'
    >>> speller_ru = CyHunspell(lang="ru")
    >>> speller_ru.spell(word_ru)
    False
    >>> speller_ru.suggest(word_ru)
    ('поваренная',)
    >>> speller_ru.replace(word_ru)
    'поваренная'
    '''
    
    langs = {
        'ru':'ru_RU',
        'en':'en_US'
    }
    
    
    def __init__(self, 
            lang='en', 
            max_dist=2, 
            cpu=os.cpu_count(), 
            # cache_manager="hunspell",disk_cache_dir=None, 
            # hunspell_data_dir=None,system_encoding=None
            spell_kwargs={} 
            ):
        
        self.lang = self.langs.get(lang,lang)
        self.spell_dict = Hunspell(self.lang, **spell_kwargs)
        self.max_dist = max_dist
        self.spell_dict.set_concurrency(cpu)
        
    
    def spell(self, word):
        
        try:
            result = self.spell_dict.spell(word)
        except UnicodeEncodeError as err:
            result = None    
        return result
    
    def suggest(self, word):
        
        try:
            result = self.spell_dict.suggest(word)
        except UnicodeEncodeError as err:
            result = tuple()
        return result 
    
    def replace(self, word, max_dist=None):
        max_dist = max_dist if max_dist is not None else self.max_dist
        
        if self.spell(word):
            return word
        suggestions = self.suggest(word)
        if (
            suggestions and edit_distance(word, suggestions[0]) <= 
            max_dist):
            return suggestions[0]
        else:
            return word

    

class Enchant():
    
    '''
    Спеллер на основе ispell\myspell
    
    >>> word_en = 'cookbok'
    >>> word_ru = 'поваринная'
    >>> speller_en = Enchant(lang="en")
    >>> speller_en.spell(word_en)
    False
    >>> speller_en.suggest(word_en)
    ['cookbook', 'copybook', 'Cook', 'cook']
    >>> speller_en.replace(word_en)
    'cookbook'
    >>> speller_ru = Enchant(lang="ru")
    >>> speller_ru.spell(word_ru)
    False
    >>> speller_ru.suggest(word_ru)
    ['поваренная']
    >>> speller_ru.replace(word_ru)
    'поваренная'
    '''
    
    def __init__(self, lang='en', max_dist=2, spell_kwargs={}):
        self.spell_dict = enchant.Dict(lang, **spell_kwargs)
        self.max_dist = max_dist
        self.lang = lang
    
    def spell(self, word):
        
        try:
            result = self.spell_dict.check(word)
        except UnicodeEncodeError as err:
            result = None    
        return result
    
 
    def suggest(self, word):
        
        try:
            result = self.spell_dict.suggest(word)
        except UnicodeEncodeError as err:
            result = []
        return result 
    
    def replace(self, word):
        
        if self.spell(word):
            return word
        suggestions = self.suggest(word)
        if (
            suggestions and edit_distance(word, suggestions[0]) <= 
            self.max_dist):
            return suggestions[0]
        else:
            return word


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    speller = Speller(lang="en", backend='Enchant')
    print(speller().spell('превет'))
    
    
    speller = Speller(lang="ru")
    print(speller().suggest('превет'))
    
    speller = Speller(backend='Enchant',lang="ru")
    print(speller().suggest('превет'))
    
    from nlptk.misc.mixins import SpellerMixin
    
    speller_ru = SpellerMixin.speller_ru()
    print(speller_ru)
    print(speller_ru().suggest('превет'))
    
    speller_ru = SpellerMixin.speller_ru(backend="Enchant")
    print(speller_ru)
    print(speller_ru().suggest('превет'))
    print(speller_ru().suggest('вечор'))
    
    
    word_en = 'cookbok'
    word_ru = 'поваринная'
    
    speller_en = CyHunspell(lang="en")
    print(speller_en.spell(word_en))
    print(speller_en.suggest(word_en))
    print(speller_en.replace(word_en))
    
    speller_ru = CyHunspell(lang="ru")
    print(speller_ru.spell(word_ru))
    print(speller_ru.suggest(word_ru))
    print(speller_ru.replace(word_ru))
    
    
    speller_en = Enchant('en')
    print(speller_en.spell(word_en))
    print(speller_en.suggest(word_en))
    print(speller_en.replace(word_en))
    
    speller_ru = Enchant('ru')
    print(speller_ru.spell(word_ru))
    print(speller_ru.suggest(word_ru))
    print(speller_ru.replace(word_ru))
        
