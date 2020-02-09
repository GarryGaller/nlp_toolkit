# -*- coding: utf-8 -*-
#http://linguis.ru/art/spell/
#http://norvig.com/spell-correct.html
#http://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb
# http://harisankar-krishnaswamy.blogspot.com/2011/10/spell-suggestion-like-google-microsoft.html
#http://theyougen.blogspot.com/2010/02/faster-spelling-corrector.html
#http://theyougen.blogspot.com/2010/02/producing-candidates-for-spelling.html
#https://code-examples.net/ru/q/230483

#https://socialnetwork.readthedocs.io/en/latest/index.html
#https://socialnetwork.readthedocs.io/en/latest/spell-check.html

import time
import re, collections

def get_words(text): 
    return re.findall(
        r'[a-z]+|[а-я]+', 
        text.lower()
    )

def train(features):
    model = collections.defaultdict(int)
    for f in features:
        model[f] += 1
    return model


alphabets = {
            'en':'abcdefghijklmnopqrstuvwxyz',
            'ru':'абвгдежзийклмнопрстуфхцчшщъыьэюя'
}

ttable = {
            'en':"qwertyuiop[]asdfghjkl;'zxcvbnm,",
            'ru':'йцукенгшщзхъфывапролджэячсмитьбю'            
}

# P(слово | опечатка) = P(опечатка | слово) * P(cлово) / const

def P(word, tokens): 
    '''вероятность слова'''
    N = sum(tokens.values())
    return tokens[word] / N


def get_lang(word):
    return max(
        alphabets, 
        key=lambda lang: sum(1 for l in word if l in alphabets[lang])
    )


def translate(word, from_lang):
    '''исправляет ошибки раскладки'''
    
    words = []
    for to_lang in alphabets:
        if to_lang == from_lang: 
            continue
        to, frm = ttable[to_lang], ttable[from_lang]
        try:
            words.append(''.join(to[frm.index(char)] for char in word))
        except ValueError: 
            pass
    return words


def edits1(word, lang):
    alphabet = alphabets[lang]
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word,lang): 
    return set(e2 for e1 in edits1(word,lang) for e2 in edits1(e1,lang))


def known_edits2(word, lang):
    return set(e2 
        for e1 in edits1(word, lang) 
        for e2 in edits1(e1, lang) if e2 in NWORDS
    )


def known(words): 
    return set(w for w in words if w in NWORDS)


def norvig_correct(word,top=1):
    lang = get_lang(word)
    candidates = (
        known([word]) 
        or known(edits1(word, lang)) 
        or known_edits2(word, lang) 
        or known(translate(word, lang)) 
        or [word]
        )
    
    #result = max(candidates, key=NWORDS.get)
    result = sorted(candidates,key=NWORDS.get,reverse=True)[:top]
    
    return result

#-------------------------------------------------
# SpellChecker на основе триграмм
#https://socialnetwork.readthedocs.io/en/latest/spell-check.html
#-------------------------------------------------
def nltk_correct(vocabulary,word_to_check,restrict=False,ngram=3,top=1):
    if restrict:
        vocabulary = [i for w in vocabulary if w[0] == word_to_check[0]]
    # calculate the distance of each word with entry and link both together
    possible_fixes = [(nltk.jaccard_distance(
                            set(nltk.ngrams(word_to_check, n=ngram)), \
                            set(nltk.ngrams(token, n=ngram))), 
                        token)  # возвращаем кортеж (дистанция, исходное слово)
                        for token in vocabulary]

    result = [fixes[1] for fixes in sorted(possible_fixes)[:top]]
    return result



if __name__ == "__main__":
    import nltk
    from nltk.corpus import words
    
    #print(nltk_correct(words.words(),'incendenece',top=10))
    #['indecence', 'impendence', 'incendiary', 'intendence', 'indene', 'incendivity', 'independence', 'incense', 'incendiarism', 'incandescence']
    
    filepath = r".\RUSSIAN\1grams-3\1grams-3.txt"           # 875111
    #filepath = r'.\RUSSIAN\efremova-wintxt\efremova.txt'   # всего 3357031,  уникальных 198299
    
    text = open(filepath,encoding='utf-8').read()
    vocabulary = set(get_words(text))
    print(len(vocabulary))  # 
    NWORDS = train(get_words(text)) # частотный словарь
    
    words = ['серибро','превет', 'очепятка','стикло','мышъ']
    print('nltk_correct')
    for word in words:
        start = time.time()
        print(
            word,
            '=>', 
            *nltk_correct(
                vocabulary,
                word,
                ngram=3,
                restrict=True, # ограничить поиск словами начинающимися с той же буквы
                top=10),
            time.time() - start
        ) 
    
    print('norvig_correct')
    
    for word in words:
        start = time.time()
        print(
            word, 
            '=>',
            *norvig_correct(word,top=10),
            time.time() - start
        )
    
    
    quit()
    #----------------------------------
    #
   
    words = ['сиребро','ашибка','ашипка', 'грокодил', 'скупитса','памагите','женщына','вичеслеть']
    print('norvig_correct')
    
    for word in words:
        start = time.time()
        print(
            word, 
            '=>',
            *norvig_correct(word),
            time.time() - start
        ) 
        
    #----------------------------------
    print('nltk_correct')
    for word in words:
        start = time.time()
        print(
            word, 
            '=>',
            *nltk_correct(vocabulary,word,ngram=4),
            time.time() - start
        ) 
        
           
    
