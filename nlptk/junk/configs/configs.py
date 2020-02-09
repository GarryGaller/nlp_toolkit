import os
from nltk.corpus import stopwords


class Config():
    
    APPDIR = os.path.abspath(os.path.dirname(__file__))
    SOURCEDIR = os.path.abspath(os.path.join(APPDIR,'..',r'CORPUS')) # источник файлов для моделей и индексов
    PICKLEDIR = os.path.abspath(os.path.join(APPDIR,r'store'))

    LEXPATH = os.path.abspath(os.path.join(APPDIR,'..','lexicons'))
    
    STOPWORDS = stopwords.words('english') + [
        'mr', 'mrs', 
        'st','sir', 'Miss',
        'www','htm','html',
        'shall','must'
    ]
    MIN_WORD_LEN = 3
    
    WILDCARD = r"*.txt" # фильтр файлов из директории-источника
    ENCODING = 'chardetect'   # кодировка для текстовых файлов; для документов .docx не требуется
    
    VERBOSE = False
    
    if not os.path.exists(PICKLEDIR):
        os.mkdir(PICKLEDIR)

    def to_dict(self):    
        config = {}
        for key in dir(self):
            if key.isupper():
                config[key.lower()] = getattr(self, key)
        return config
        
if __name__ == "__main__":
    print(Config.APPDIR)
    print(Config.SOURCEDIR)
    print(Config.PICKLEDIR )
    print(Config.LEXPATH)        
        
     
        
