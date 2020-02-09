
#1) извлечение наиболее частотных биграм
bg = list(nltk.bigrams(prog.findall(s.lower())))
bgfd = nltk.FreqDist(bg)
bgfd.most_common(18)

#2) Извлечение биграмм на основе мер ассоциации и статистических критериев.
from nltk.collocations import *
N_best = 100 # number of bigrams to extract

# class for association measures
bm = nltk.collocations.BigramAssocMeasures()

# class for bigrams extraction and storing
f = BigramCollocationFinder.from_words(m)

# remove too seldom bigrams
f.apply_freq_filter(5)
# get top-100 bigrams using simple frequency
raw_freq_ranking = [' '.join(i) for i in
f.nbest(bm.raw_freq, N_best)]

# get top-100 bigrams using described measures
tscore_ranking = [' '.join(i) for i in
f.nbest(bm.student_t, N_best)]

pmi_ranking = [' '.join(i) for i in
f.nbest(bm.pmi, N_best)]

llr_ranking = [' '. join(i) for i in
f.nbest(bm.likelihood_ratio, N_best)]

chi2_ranking = [' '.join(i) for i in
f.nbest(bm.chi_sq, N_best)]


#3) Алгоритм TextRank для извлечения словосочетаний.
#4) Rapid Automatic Keyword Extraction.    https://dspace.spbu.ru/bitstream/11701/10796/1/SandulMV_GraduationWork.pdf
#5) Выделение ключевых слов по tf-idf.