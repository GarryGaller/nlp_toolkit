from sklearn.decomposition import LatentDirichletAllocation
from nlptk.vocab.vocabulary import vocabulary

from functools import partial
from pprint import pprint
import os
import glob
import typing

if sys.version_info < (3,6):
    import win_unicode_console
    win_unicode_console.enable())

from nltk.corpus import stopwords
from nlptk.vocab.vocabulary import Lexicon
from nlptk.misc.mixins import LemmatizerMixin
from nlptk.misc.mixins import TokenizerMixin
from nlptk.misc.mixins import SentencizerMixin
from nlptk.misc.mixins import StripperMixin

lemmatizer = LemmatizerMixin.lemmatize_nltk
tokenizer =  TokenizerMixin.treebank_word_tokenize # wordpunct_tokenize
sentencizer = SentencizerMixin.sentencize_nltk
