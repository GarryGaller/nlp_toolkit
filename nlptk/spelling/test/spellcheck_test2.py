import time
import re
from pprint import pprint
from collections import defaultdict

#http://linguis.ru/art/spell/
#http://norvig.com/spell-correct.html

'''
For a word of length n, 
there will be 
n deletions, 
n-1 transpositions, 
26n alterations, 
26(n+1) insertions, 
for a total of 54n+25 

для русского:
33-1
33n
33(n+1)
67n + 31
'''

r_alphabet = re.compile(r'[а-яА-ЯЁё0-9-]+|[.,:;?!]+')


def gen_lines(corpus):
    data = open(corpus)
    for line in data:
        yield line.lower()

def gen_tokens(lines):
    for line in lines:
        for token in r_alphabet.findall(line):
            yield token

def bigrams(tokens):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    for tu in zip(tokens[:],tokens[1:]):
        yield tu 


alphabets = {
            'en':'abcdefghijklmnopqrstuvwxyz',
            'ru':'абвгдежзийклмнопрстуфхцчшщъыьэюя'
}

ttable = {
            'en':"qwertyuiop[]asdfghjkl;'zxcvbnm,",
            'ru':"йцукенгшщзхъфывапролджэячсмитьбю"            
}

def get_lang(word):
    return max(
        alphabets, 
        key=lambda lang: sum(1 for l in word if l in alphabets[lang])
    )

def edits1(word, lang):
    alphabet = alphabets[lang]
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    
    return set(deletes + transposes + replaces + inserts)

def edits2(word,lang): 
    return set(e2 for e1 in edits1(word,lang) 
                    for e2 in edits1(e1,lang)
            ) 
            
def edits3(word,lang): 
    return set(e3 for e1 in edits1(word,lang) 
                    for e2 in edits1(e1,lang)
                    for e3 in edits1(e2,lang)
            ) 

         
def known_edits2(word, lang):
    return set(e2 
        for e1 in edits1(word, lang) 
        for e2 in edits1(e1, lang) if e2 in NWORDS
    )


def translate(word, from_lang):
    words = []
    for to_lang in alphabets:
        if to_lang == from_lang: continue
        to, frm = ttable[to_lang], ttable[from_lang]
        try:
            result = ''.join(to[frm.index(char)] for char in word)
            words.append(result)
            
        except ValueError as err: 
            pass
    return words


def keyboard_bigrams():
    '''формирование списка всех рядом стоящих клавиатурных сочетаний'''
    # три линии клавиатурной раскладки 
    keyboard = ['йцукенгшщзхъ','фывапролджэ','ячсмитьбю']
    lst = []
    for tokens in keyboard:
        lst.extend(
            map(''.join,
            bigrams(tokens))
        )
    # дополнительно переворачиваем каждое сочетание и добавляем в список
    lst += list(map(lambda x:x[::-1],lst))
    return lst



def keyboard_transpositions(tokens,ngrams,invert=False,index=False):
    '''
    Генерация ошибок клавиатурных транспозиций
    Результат - словарь вида: 
    dict[правильное_написание] = {ошибочное_написание1,ошибочное_написание2,...}
    
    >>> tokens = ['клавиатура', 'лаборатория','проложить','синхрофазатрон','привет','ячневый']
    >>> transpositions = keyboard_transpositions(tokens,ngrams)
    >>> pprint(transpositions) 
    defaultdict(<class 'set'>,
            {'клавиатура': {'клваиатура'},
             'лаборатория': {'лаброатроия', 'лаброатория'},
             'привет': {'рпивет'},
             'проложить': {'прлоожтиь',
                           'проложтиь',
                           'рплоожить',
                           'рплоожтиь',
                           'рположить'},
             'синхрофазатрон': {'синхорфазаторн',
                                'синхорфазатрон',
                                'синхрофазаторн'},
             'ячневый': {'чяенывй', 'чяневый', 'чяенвый', 'яченывй'}})
    
    :invert=True сгенерировать инвентированный вариант словаря
    [ошибочное_написание] = правильное_написание
    :index=True - использовать индекс правильного слова вместо него самого 
    [ошибочное_написание] = индекс_слова_в_исходном_списке
    
    >>>transpositions = keyboard_transpositions(tokens,ngrams,invert=True,index=True)
    >>> pprint(transpositions)  
    {'клваиатура': 0,
     'лаброатория': 1,
     'лаброатроия': 1,
     'прлоожтиь': 2,
     'проложтиь': 2,
     'рпивет': 4,
     'рплоожить': 2,
     'рплоожтиь': 2,
     'рположить': 2,
     'синхорфазаторн': 3,
     'синхорфазатрон': 3,
     'синхрофазаторн': 3,
     'чяенвый': 5,
     'чяенывй': 5,
     'чяневый': 5,
     'яченывй': 5}

    '''
    
    if not invert:
        word_dict = defaultdict(set)
    else:
        word_dict = {}   
    
    for idx,token in enumerate(tokens): 
        orig_token = token
        for ng2 in ngrams:
            for _ in range(token.count(ng2)):
                # делаем замену сочетания на реверс, чтобы получить ошибку транспозиции
                token = token.replace(ng2,ng2[::-1],1)
                if token != orig_token:
                    if not invert:
                        word_dict[orig_token].add(token)
                    else:
                         word_dict[token] = orig_token if not index else idx     
    return word_dict



def invert_dict(d):
    inverted_dict = defaultdict(list)
    for key in d:
        if hasattr(d[key], '__iter__'):
            for term in d[key]:
                inverted_dict[term].append(key)
        else:
            inverted_dict[d[key]] = key
    return inverted_dict



if __name__ == "__main__":
    
    words = ['ghbdtn','знерщт','jib,rf']
    # определяем ошибки раскладки
    for word in words:
        print(translate(word, get_lang(word)))
    
    words = ['привет','somthing']
    '''
    for word in words:
        variants = edits1(word, get_lang(word))
        pprint(variants)
        print(len(variants))
    '''
    
    ngrams = keyboard_bigrams()
    print(ngrams )

    '''
    ['йц', 'цу', 'ук', 'ке', 'ен', 'нг', 'гш', 'шщ', 
    'щз', 'зх', 'хъ', 'фы', 'ыв', 'ва', 'ап', 'пр', 
    'ро', 'ол', 'лд', 'дж', 'жэ', 'яч', 'чс', 'см', 
    'ми', 'ит', 'ть', 'ьб', 'бю', 'цй', 'уц', 'ку', 
    'ек', 'не', 'гн', 'шг', 'щш', 'зщ', 'хз', 'ъх', 
    'ыф', 'вы', 'ав', 'па', 'рп', 'ор', 'ло', 'дл', 
    'жд', 'эж', 'чя', 'сч', 'мс', 'им', 'ти', 'ьт', 'бь', 'юб'
    ]'''
    
    '''
    tokens = ['клавиатура', 'лаборатория','проложить','синхрофазатрон','привет','ячневый']
    transpositions = keyboard_transpositions(tokens,ngrams)
    pprint(transpositions) 
    
    transpositions = keyboard_transpositions(tokens,ngrams,invert=True,index=True)
    pprint(transpositions)  
    '''
    
    
    #--------------------------------------------------
    corpus = r'.\RUSSIAN\Зализняк\zdf-win\zdf-win.txt'
    lines = gen_lines(corpus)
    tokens = gen_tokens(lines) 
    
    '''
    direct_index = keyboard_transpositions(
        tokens,
        ngrams,
    )
    pprint(direct_index)  
    print(len(direct_index)) # 68782
    quit()
    
    invert_index = keyboard_transpositions(
        tokens,
        ngrams,
        invert=True,
    )
    
    
    #NWORDS = invert_index 
    #pprint(edits1('привет', 'ru'))
    
    pprint(invert_index)  
    print(len(invert_index))  # 147501
    print(invert_index.get('рпивет'))
    quit()
    '''
    start = time.time()
    direct_index = edits1('привет', 'ru')   # 416 вариантов
    #pprint(direct_index)
    
    
    quit()
    
    print(len( direct_index),time.time() - start)
    print('привед'  in direct_index)
    print('рпивет'  in direct_index)
    print('преивет' in direct_index)
    print('превет'  in direct_index)
    print('превед'  in direct_index) # 2 ошибки False
    
    print('*' * 20) 
    start = time.time()
    direct_index =  edits2('привет', 'ru') # 79380 0.18720030784606934
    print(len( direct_index),time.time() - start)
    print('привед'   in direct_index)
    print('рпивет'   in direct_index)
    print('преивет'  in direct_index)
    print('превет'   in direct_index)
    print('превед'   in direct_index) 
    print('перевед'  in direct_index) # 3 ошибки False

    
    print('*' * 20) 
    start = time.time()
    direct_index =  edits3('привет', 'ru') #   9488979 106.18507266044617
    print(len( direct_index),time.time() - start)
    print('привед'   in direct_index)
    print('рпивет'   in direct_index)
    print('преивет'  in direct_index)
    print('превет'   in direct_index)
    print('превед'   in direct_index) 
    print('перевед'  in direct_index) # все три ошибки
    
    quit()
    
    pprint(invert_index)
    
    quit()
    for  word in edits1('привет', 'ru'):
        print(word,invert_index.get(word))  
        
        
        
'''        
{'апривет',
 'аривет',
 'бпривет',
 'бривет',
 'впривет',
 'вривет',
 'гпривет',
 'гривет',
 'дпривет',
 'дривет',
 'епривет',
 'еривет',
 'жпривет',
 'жривет',
 'зпривет',
 'зривет',
 'ипривет',
 'иривет',
 'йпривет',
 'йривет',
 'кпривет',
 'кривет',
 'лпривет',
 'лривет',
 'мпривет',
 'мривет',
 'нпривет',
 'нривет',
 'опривет',
 'оривет',
 'паивет',
 'паривет',
 'пбивет',
 'пбривет',
 'пвивет',
 'пвривет',
 'пгивет',
 'пгривет',
 'пдивет',
 'пдривет',
 'пеивет',
 'перивет',
 'пживет',
 'пжривет',
 'пзивет',
 'пзривет',
 'пивет',
 'пиивет',
 'пирвет',
 'пиривет',
 'пйивет',
 'пйривет',
 'пкивет',
 'пкривет',
 'пливет',
 'плривет',
 'пмивет',
 'пмривет',
 'пнивет',
 'пнривет',
 'поивет',
 'поривет',
 'ппивет',
 'ппривет',
 'правет',
 'праивет',
 'прбвет',
 'прбивет',
 'прввет',
 'првет',
 'првивет',
 'првиет',
 'пргвет',
 'пргивет',
 'прдвет',
 'прдивет',
 'превет',
 'преивет',
 'пржвет',
 'прживет',
 'прзвет',
 'прзивет',
 'приавет',
 'приает',
 'прибвет',
 'прибет',
 'привает',
 'приват',
 'привбет',
 'привбт',
 'приввет',
 'приввт',
 'привгет',
 'привгт',
 'привдет',
 'привдт',
 'приве',
 'привеа',
 'привеат',
 'привеб',
 'привебт',
 'привев',
 'привевт',
 'привег',
 'привегт',
 'привед',
 'приведт',
 'привее',
 'привеет',
 'привеж',
 'привежт',
 'привез',
 'привезт',
 'привеи',
 'привеит',
 'привей',
 'привейт',
 'привек',
 'привект',
 'привел',
 'привелт',
 'привем',
 'привемт',
 'привен',
 'привент',
 'привео',
 'привеот',
 'привеп',
 'привепт',
 'привер',
 'приверт',
 'привес',
 'привест',
 'привет',
 'привета',
 'приветб',
 'приветв',
 'приветг',
 'приветд',
 'привете',
 'приветж',
 'приветз',
 'привети',
 'приветй',
 'приветк',
 'приветл',
 'приветм',
 'приветн',
 'привето',
 'приветп',
 'приветр',
 'приветс',
 'приветт',
 'привету',
 'приветф',
 'приветх',
 'приветц',
 'приветч',
 'приветш',
 'приветщ',
 'приветъ',
 'приветы',
 'приветь',
 'приветэ',
 'приветю',
 'приветя',
 'привеу',
 'привеут',
 'привеф',
 'привефт',
 'привех',
 'привехт',
 'привец',
 'привецт',
 'привеч',
 'привечт',
 'привеш',
 'привешт',
 'привещ',
 'привещт',
 'привеъ',
 'привеът',
 'привеы',
 'привеыт',
 'привеь',
 'привеьт',
 'привеэ',
 'привеэт',
 'привею',
 'привеют',
 'привея',
 'привеят',
 'привжет',
 'привжт',
 'привзет',
 'привзт',
 'привиет',
 'привит',
 'привйет',
 'привйт',
 'привкет',
 'привкт',
 'привлет',
 'привлт',
 'привмет',
 'привмт',
 'привнет',
 'привнт',
 'привоет',
 'привот',
 'привпет',
 'привпт',
 'приврет',
 'приврт',
 'привсет',
 'привст',
 'привт',
 'привте',
 'привтет',
 'привтт',
 'привует',
 'привут',
 'привфет',
 'привфт',
 'привхет',
 'привхт',
 'привцет',
 'привцт',
 'привчет',
 'привчт',
 'прившет',
 'прившт',
 'привщет',
 'привщт',
 'привъет',
 'привът',
 'привыет',
 'привыт',
 'привьет',
 'привьт',
 'привэет',
 'привэт',
 'привюет',
 'привют',
 'привяет',
 'привят',
 'пригвет',
 'пригет',
 'придвет',
 'придет',
 'приевет',
 'приевт',
 'приеет',
 'приет',
 'прижвет',
 'прижет',
 'призвет',
 'призет',
 'приивет',
 'прииет',
 'прийвет',
 'прийет',
 'приквет',
 'прикет',
 'прилвет',
 'прилет',
 'примвет',
 'примет',
 'принвет',
 'принет',
 'приовет',
 'приоет',
 'припвет',
 'припет',
 'прирвет',
 'прирет',
 'присвет',
 'присет',
 'притвет',
 'притет',
 'приувет',
 'приует',
 'прифвет',
 'прифет',
 'прихвет',
 'прихет',
 'прицвет',
 'прицет',
 'причвет',
 'причет',
 'пришвет',
 'пришет',
 'прищвет',
 'прищет',
 'приъвет',
 'приъет',
 'приывет',
 'приыет',
 'приьвет',
 'приьет',
 'приэвет',
 'приэет',
 'приювет',
 'приюет',
 'приявет',
 'прияет',
 'прйвет',
 'прйивет',
 'прквет',
 'пркивет',
 'прлвет',
 'прливет',
 'прмвет',
 'прмивет',
 'прнвет',
 'прнивет',
 'провет',
 'проивет',
 'прпвет',
 'прпивет',
 'пррвет',
 'прривет',
 'прсвет',
 'прсивет',
 'пртвет',
 'пртивет',
 'прувет',
 'пруивет',
 'прфвет',
 'прфивет',
 'прхвет',
 'прхивет',
 'прцвет',
 'прцивет',
 'прчвет',
 'прчивет',
 'пршвет',
 'пршивет',
 'прщвет',
 'прщивет',
 'пръвет',
 'пръивет',
 'прывет',
 'прыивет',
 'прьвет',
 'прьивет',
 'прэвет',
 'прэивет',
 'прювет',
 'прюивет',
 'прявет',
 'пряивет',
 'псивет',
 'псривет',
 'птивет',
 'птривет',
 'пуивет',
 'пуривет',
 'пфивет',
 'пфривет',
 'пхивет',
 'пхривет',
 'пцивет',
 'пцривет',
 'пчивет',
 'пчривет',
 'пшивет',
 'пшривет',
 'пщивет',
 'пщривет',
 'пъивет',
 'пъривет',
 'пыивет',
 'пыривет',
 'пьивет',
 'пьривет',
 'пэивет',
 'пэривет',
 'пюивет',
 'пюривет',
 'пяивет',
 'пяривет',
 'ривет',
 'рпивет',
 'рпривет',
 'рривет',
 'спривет',
 'сривет',
 'тпривет',
 'тривет',
 'упривет',
 'уривет',
 'фпривет',
 'фривет',
 'хпривет',
 'хривет',
 'цпривет',
 'цривет',
 'чпривет',
 'чривет',
 'шпривет',
 'шривет',
 'щпривет',
 'щривет',
 'ъпривет',
 'ъривет',
 'ыпривет',
 'ыривет',
 'ьпривет',
 'ьривет',
 'эпривет',
 'эривет',
 'юпривет',
 'юривет',
 'япривет',
 'яривет'}
'''        
        
        
