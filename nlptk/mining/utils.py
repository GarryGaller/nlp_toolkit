
import os,sys
import heapq
from chardet.universaldetector import UniversalDetector


def _sort(obj, typ):
    if typ == 1:
        res = sorted(obj.items(),key=lambda t:t[1])
    elif typ == -1:
        res = sorted(obj.items(),key=lambda t:-t[1])
    else:
        res = obj     
    return res


def _top(obj, n):
    if n > 0:
        res = heapq.nlargest(n, obj, key=obj.get)   
    else:
        res = heapq.nsmallest(abs(n), obj, key=obj.get)
    return res 


def get_feats(tokens):
    return dict([(token, True) for token in tokens])

def create_sample(tag, train_ratio):
    from nltk.corpus import movie_reviews
    
    ids = movie_reviews.fileids(tag)
    feats = [(get_feats(movie_reviews.words(fileids=[f])),tag) for f in ids]
    idx = int(len(feats) * train_ratio)
    train = feats[: idx]
    text = feats[idx: ]
    return train, test

#train_pos, test_pos = create_sample(’pos’, 0.75)
#train_neg, test_neg = create_sample(’neg’, 0.75)
#train = train_pos + train_neg
#test = test_pos + test_neg



def pad_sequences(
    sequences, 
    maxlen,
    padding='pre',
    truncating='pre', 
    value=0.0
    ):
    
    sequences = sequences.copy()
    
    for seq in sequences:
        if not len(seq):
            continue 
        
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq[:] = seq[-maxlen:]
            elif truncating == 'post':
                seq[:] = seq[:maxlen]
            else:
                raise ValueError('Truncating type "%s" '
                                 'not understood' % truncating)
        else:
            pad_len = maxlen - len(seq) 
            if padding == 'post':
                seq[len(seq):] =  [value] * pad_len 
            elif padding == 'pre': 
                seq[:] = [value] * pad_len  + seq[:]
            else:
                raise ValueError('Padding type "%s" '
                                'not understood' % padding)
            
    return sequences 
    
    
    
class datapath():
    
    def __init__(self, path, datadir="data", ext=".txt"):
        if not path:
            raise ValueError('Path not passed')
        self.appdir = os.path.abspath(os.path.dirname(__file__))
        self.datadir = datadir
        self.ext = ext
        self.name = os.path.basename(os.path.splitext(path)[0])
        self.short = os.path.join(self.appdir,'data',self.name)
        self.full = os.path.join(self.appdir,'data',self.name + ext)
        
 
        
def chardetector(filepath):
    default_encoding = sys.getfilesystemencoding()
    #default_enc = locale.getpreferredencoding()
    result = None
    detector = UniversalDetector()
    detector.reset()
    
    for line in open(filepath, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()
    encoding = detector.result.get('encoding') or default_encoding
    
    return encoding    
    
