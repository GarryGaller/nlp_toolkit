import string
import re


'''
Полный список граммем здесь: http://opencorpora.org/dict.php?act=gram
NOUN    имя существительное хомяк
ADJF    имя прилагательное (полное) хороший
ADJS    имя прилагательное (краткое)    хорош
COMP    компаратив  лучше, получше, выше
VERB    глагол (личная форма)   говорю, говорит, говорил
INFN    глагол (инфинитив)  говорить, сказать
PRTF    причастие (полное)  прочитавший, прочитанная
PRTS    причастие (краткое) прочитана
GRND    деепричастие    прочитав, рассказывая
NUMR    числительное    три, пятьдесят
ADVB    наречие круто
NPRO    местоимение-существительное он
PRED    предикатив  некогда
PREP    предлог в
CONJ    союз    и
PRCL    частица бы, же, лишь
INTJ    междометие  ой
'''
#https://www.regular-expressions.info/posixbrackets.html

# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'  32 символа
#\u2026 -  многоточие  :          …
#\u2014 -  длинное тире:          —
#\u2013    cреднее тире:          –
#\u2012    цифровое тире:         ‒
#\u2010    дефис(настоящий):      ‐
#\u2212    знак минус(настоящий): −

PUNCTUATION = string.punctuation + '\u2026\u2014\u2013\u2012\u2010\u2212' + '«»‹›‘’“”„'

RE_PUNCT_EXTENDED = re.compile(re.escape(PUNCTUATION))
RE_HYPHENATION = re.compile(r'[-]+[\x20]*\r?\n\s*')
# from textacy
RE_HYPHENATED_WORD = re.compile(
    r"(\w{2,}(?<!\d))\-\s+((?!\d)\w{2,})",
    flags=re.UNICODE | re.IGNORECASE)
re_hyphenated_word=lambda text:RE_HYPHENATED_WORD.sub(r"\1\2", text)


# апостроф-кавычка\гравис\дефисоминус
SURROGATE_SUBSTITUTES  = '\u0027\u0060\u002D' 
# юникодный дефис\косая черта\знак ударения\знак типографского апострофа\модификатор буквы апостроф
NON_ALPHABETIC_ORTHOGRAPHIC_SIGNS = '\u2010\u002F\u0301\u2019\u02BC' 
EXCLUDE_CHARS = '$#@%:.,' # набор символов по которым не нужно делить токены: 
"""
:  = web-адреса и файловые пути в стиле Unix  (для windows путей не работает)
@  = электронные адреса, 
#  = хэштеги, 
$  = денежные единицы, 
%  = числа со знаком процента
., = числа включающие точку или запятую в качестве разделителя целой и дробной частей
"""

RE_TOKEN = re.compile(
    r"(?m)([^\w_" + 
    EXCLUDE_CHARS + 
    NON_ALPHABETIC_ORTHOGRAPHIC_SIGNS + 
    SURROGATE_SUBSTITUTES + 
    "]|[+]|(:|,|[.]{3})(?=\s+?|$|\u0022)|([.]{1,3})(?=[)]|\s+?[^a-zа-яё]|$))"
)


RE_WORD = re.compile(r'\b\w+?\b',re.UNICODE)
RE_WORD2 = re.compile(r'\w+|\$[\d\.]+|\S+',re.UNICODE)
RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_PUNCT2 = re.compile(r'([\s%s])+' % re.escape(''.join(set(string.punctuation) - {"'","`"})), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)                     # html
RE_URLS = re.compile(r"(www|http:|https:)+[^\s]+[\w]", re.UNICODE) # urls
RE_DIGIT = re.compile(r"[0-9]+", re.UNICODE)     # все арабско-индийские цифры (изменить)
RE_DECIMAL = re.compile(r"[0-9]+", re.UNICODE)   # все арабско-индийские цифры (изменить)
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)   # все арабско-индийские цифры (изменить)
RE_NONALPHA = re.compile(r"[\W]", re.UNICODE)    # все небуквенные символы
RE_NONLETTER2 = re.compile(r'(\W)\1', re.UNICODE)        # все повторяющиеся двухсимвольные наборы небуквенных символов
RE_NONLETTER = re.compile(r'(?=(\W))\1{2,}', re.UNICODE) # все наборы из небуквенных символов длиной от 2-х символов
RE_NONASCII= re.compile(r'([^a-z]+)', re.UNICODE|re.I) # все не латинские буквы
RE_AL_NUM = re.compile(r'([a-z]+)([0-9]+)', flags=re.UNICODE|re.I) # все сочетания из латинские буквы и последующих цифр
RE_NUM_AL = re.compile(r'([0-9]+)([a-z]+)', flags=re.UNICODE|re.I) # все сочетания из цифр и последующих латинскихе букв
RE_ASCII = re.compile(r"[\x00-\x7F]+", flags=re.UNICODE)  # все ASCII символы - печатные и непечатные
RE_LATIN = re.compile(r'([a-z]+)', flags=re.UNICODE|re.I) # все латинские буквы
RE_WHITESPACE = re.compile(r'(\s)+', re.UNICODE) # все  пробельные символы
RE_BLANK = re.compile(r'[ \t]+', re.UNICODE)     # только пробел и tab
RE_HYPHENATION = re.compile(r'[-]+\s*\r?\n\s*',re.UNICODE) # переносы слов

RE_QOUTES = re.compile(r'["\'«»‹›‘’“”„`]',re.UNICODE)
RE_QOUTES = re.compile(r'["«»‹›‘’“”„`]',re.UNICODE)

RE_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

ROMAN_NUMERALS = re.compile('''
      \b                        # начало слова
      M{0,3}                    # тысячи - 0 до 3 M
      (CM|CD|D?C{0,3})          # сотни — 900 (CM), 400 (CD), 0-300 (0 до 3 C),
                                # или 500-800 (D, с последующими от 0 до 3 C)
      (XC|XL|L?X{0,3})          # десятки - 90 (XC), 40 (XL), 0-30 (0 до 3 X),
                                # или 50-80 (L, с последующими от 0 до 3 X)
      (IX|IV|V?I{0,3})          # единицы - 9 (IX), 4 (IV), 0-3 (0 до 3 I),
                                # или 5-8 (V, с последующими от 0 до 3 I)
      \b                        № конец слова  
''',re.VERBOSE|re.IGNORECASE)



'''
 'POS',
 'CC'
 'UH'
 'PRP','PRP$',
 'NNP','NNPS',
 'SYM',
 'TO' ,
  'WP','WDT','WP$'
  'WRB'
  
  'NN','NNS',
  'RB','RBR','RBS',
  'JJ','JJR''JJS',
  'VB','VBZ','VBP','VBD','VBN','VBG',
  'FW'
'''  
  
'''
CC  conjunction, coordinating   and, or, but
CD  cardinal number five, three, 13%
DT  determiner  the, a, these
EX  existential there   there were six boys
FW  foreign word    mais
IN  conjunction, subordinating or preposition   of, on, before, unless
JJ  adjective   nice, easy
JJR adjective, comparative  nicer, easier
JJS adjective, superlative  nicest, easiest
LS  list item marker     
MD  verb, modal auxillary   may, should
NN  noun, singular or mass  tiger, chair, laughter
NNS noun, plural    tigers, chairs, insects
NNP noun, proper singular   Germany, God, Alice
NNPS    noun, proper plural we met two Christmases ago
PDT predeterminer   both his children
POS possessive ending   's
PRP pronoun, personal   me, you, it
PRP$    pronoun, possessive my, your, our
RB  adverb  extremely, loudly, hard 
RBR adverb, comparative better
RBS adverb, superlative best
RP  adverb, particle    about, off, up
SYM symbol  %
TO  infinitival to  what to do?
UH  interjection    oh, oops, gosh
VB  verb, base form think
VBZ verb, 3rd person singular present   she thinks
VBP verb, non-3rd person singular present   I think
VBD verb, past tense    they thought
VBN verb, past participle   a sunken ship
VBG verb, gerund or present participle  thinking is fun
WDT wh-determiner   which, whatever, whichever
WP  wh-pronoun, personal    what, who, whom
WP$ wh-pronoun, possessive  whose, whosever
WRB wh-adverb   where, when
'''
