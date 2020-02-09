import operator
import scipy.stats
import sklearn.metrics
import pandas as pd
import numpy as np
import numpy.linalg as la
import math
from math import  sqrt, fabs, log, factorial as f
import scipy.spatial
from functools import reduce

def P(word, tokens): 
    '''вероятность слова'''
    N = sum(tokens.values())
    return tokens[word] / N


#http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
# самый быстрый
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def argsort(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key=lambda x: x[1])]

def argmin(seq):
    return seq.index(min(seq))
    
def argmax(seq):
    return seq.index(max(seq))

#sorted(range(len(seq)), key = lambda x: seq[x].sort_property)
'''
seq = [1,2,4,100,0,0,0,0,0,0,0,-1,0,56]
argsort = lambda l: sorted(range(len(l)), key=l.__getitem__)
lst_idx = argsort(seq)
idx_min,idx_max = lst_idx[0],lst_idx[-1]
if idx_min > idx_max:
    idx_max,idx_min = idx_min,idx_max

print(seq[idx_min:idx_max].count(0))

seq = [1,2,4,100,0,0,0,0,0,0,0,-1,0,56]
idx_min,idx_max = seq.index(min(seq)),seq.index(max(seq))
if idx_min > idx_max:
    idx_max,idx_min = idx_min,idx_max
print(seq[idx_min:idx_max].count(0)))
'''

#=============================================================
# поэлементное произведение векторов, где каждый элемент равен 
# произведению соответствующих элементов заданных векторов.
#=============================================================
def vectorproduct(v1, v2):
    return [x * y for x, y in zip(v1, v2)]

#=============================================================
# скалярное произведение
#=============================================================
def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

#=============================================================
# угол между векторами как аркосинус от косинуса угла
#=============================================================
def angle_vect(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    res = np.dot(v1, v2)
    numerator = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    res = numerator/denom
    if res > 1.0: res = 1.0
    return np.arccos(res)
#=============================================================
# среднее арифметическое
#=============================================================
def mean(samples):
    return sum(samples)/len(samples)

#scipy.stats.gmean(a[, axis, dtype])    
#scipy.stats.hmean(a[, axis, dtype])
#=============================================================
# среднее взвешенное
#=============================================================
def weighted_average(samples):
    from collections import Counter
    cnt = Counter(samples)
    return sum(k * v for k,v in cnt.items())/sum(cnt.values())


def intergroup_average(*groups):
    '''межгрупповое средневзвешенное'''
    numerator = sum(weighted_average(g) * len(g) for g in groups)
    denom = sum(len(g) for g in groups)
    
    return numerator/denom

#=============================================================
# среднее геометрическое
#=============================================================

def geometric_average(samples):
    res = reduce(lambda x,y : x*y,samples) 
    res = res ** (1/float(len(samples)))
    return res


#=============================================================
# среднее гармоническое 
#=============================================================

def harmonic_average(samples):
    res = len(samples)/sum(1/v for v in samples)
    return res


#=============================================================
# среднее квадратическое 
#=============================================================

def square_average(samples):
    res = sqrt(sum(v**2 for v in samples)/len(samples))
    return res


#=============================================================
# среднее кубическое 
#=============================================================

def cubic_average(samples):
    res = sqrt(sum(v**(1/3) for v in samples)/len(samples))
    return res


#=============================================================
# cреднее линейное отклонение
#=============================================================
def mean_deviation(samples):
    average = mean(samples)
    result = sum((frequency - average) for frequency in samples)
    return result/len(samples) 

#=============================================================
# сумма квадратов отклонений от средней
#=============================================================
def sum_squared_deviations(samples):
    average = mean(samples)
    result = sum((frequency - average) ** 2 for frequency in samples)
    return result 


#=============================================================
# сумма произведений отклонений от средней величины
#=============================================================
def sum_products_deviations(samples_x,samples_y):
    sampling_deviation = 0
    mean_x = mean(samples_x)
    mean_y = mean(samples_y)
    
    sampling_deviation = sum(
        (x - mean_x) * (y - mean_y) for x,y in zip(samples_x,samples_y)
    )
    
    return sampling_deviation

#=============================================================
# дисперсия: отношение суммы квадратов отклонений от среднего к числу наблюдений
#=============================================================
def var(samples, ddof=0):
    '''
    Дисперсия - (сигма в квадрате, СКО в квадрате) отношение суммы квадратов 
    отклонений от среднего к числу наблюдений.
    (Средний квадрат отклонения от среднего значения.
    Среднее значение квадратов минус квадрат среднего.)
    
    samples : array_like 
        array of values
    ddof : int, optional
        delta degrees of freedom
    
    ddof=0 дисперсия популяции (всей генеральной совокупности)
    ddof=1 выборочная дисперсия
    
    В стандартной статистической практике ddof=1 
    обеспечивает несмещенную оценку дисперсии гипотетической 
    бесконечной совокупности. 
    ddof=0 предоставляет оценку максимального правдоподобия 
    дисперсии для нормально распределенных переменных.
    
    >>> samples = [168,160,168,168,168]
    >>> from statistics import variance,pvariance
    #sample variance
    >>> var(samples,ddof=1))
    12.799999999999999
    >>> import numpy as np
    >>> np.var(samples,ddof=1)
    12.799999999999999
    >>> variance(samples)
    12.799999999999999
    # population variance
    >>> var(samples)
    10.239999999999998
    >>> np.var(samples)
    10.239999999999998
    >>> pvariance(samples)
    10.24
    '''
    
    sampling_deviation = sum_squared_deviations(samples)
    disp = sampling_deviation/(len(samples) - ddof)
    
    return disp


#=============================================================
# стандартное (среднее квадратичное) отклонение
#=============================================================
def std(samples,ddof=0,return_var=False):
    '''
    Cреднеквадратическое отклонение - дает абсолютную оценку 
    рассеивания(разброса) значений случайной величины относительно 
    её математического ожидания.
    
    samples : array_like 
        array of values
    ddof : int, optional
        delta degrees of freedom
    
    >>> std([1,2,3,4,5,6,7,8,9]))
    2.581988897471611
    >>> import numpy as np
    >>> np.std([1,2,3,4,5,6,7,8,9]))
    2.581988897471611
    '''
    disp = var(samples,ddof)
    sd = sqrt(disp)
    if return_var:
        result = sd,disp
    else:
        result = sd
    return result

#=============================================================
# стандартная  ошибка среднего
#=============================================================
def sem(sample,ddof=1):
    '''Стандартная ошибка среднего (средняя погрешность среднего значения)
    
    >>> sample = [16, 18, 16, 14, 12, 12] 
    >>> import scipy.stats
    >>> scipy.stats.sem(sample)
    0.9888264649460884
    >>> sem(sample)
    0.9888264649460884
    '''
    
    n = len(sample)
    result = std(sample,ddof=ddof) / sqrt(n)
    return result

#=============================================================
# стандартная  ошибка корреляции
#=============================================================
# проверить!
def std_corrcoef(samples_x, samples_y):
     '''Стандартная ошибка корреляции Пирсона'''
     
     assert len(samples_x) == len(samples_y)
     N = len(samples_x)
     R = pearsonr(samples_x, samples_y)
     numerator = 1 - R ** 2
     denom = N - 2
     SE = numerator / denom 
     p_value = R / SE
     return SE,p_value


#=============================================================
# доверительный интервал
#=============================================================
#http://qaru.site/questions/124655/compute-a-confidence-interval-from-sample-data
#http://easydan.com/arts/2016/probability-estimation/
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a) 
    se = scipy.stats.sem(a)
    p = (1 + confidence)/2. # 0.975
    df = n - 1
    h = se * scipy.stats.t.ppf(p, df)
    #h = qt(p = p, df = df) # в R lang
    return m, (m - h, m + h)

def mean_confidence_interval2(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    df = n - 1
    confint = scipy.stats.t.interval(confidence, df, loc=m, scale=se)
    return m,confint

#(11.5, (9.445739743239121, 13.554260256760879))
#(11.5, (9.445739743239121, 13.554260256760879))


#statsmodels.stats.api.DescrStatsW(a).tconfint_mean()


#=============================================================
# автокорреляция
#=============================================================
def estimated_autocorrelation(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


#=============================================================
# коэффициент вариации
#=============================================================
def variation(samples):
    '''
    Коэффициент вариации - 
    безразмерная величина, характеризующая разброс выборочных данных.
    Отношение MSE (стандартного отклонения) случайной величины 
    к её мат. ожиданию(среднему).
    Не должен превышать 40%.
    Если значение коэффициента вариации не превышает 33%, 
    то совокупность считается однородной, 
    если больше 33%, то – неоднородной.
    
    samples:  array_like
        array of values
    
    >>> samples = [1, 2, 3, 4, 5]
    >>> print(variation(samples))
    0.47140452079103173
    >>> import scipy.stats
    >>> print(scipy.stats.variation(samples))
    0.47140452079103173
    '''
    return std(samples)  / mean(samples)


#=============================================================
# ковариация
#=============================================================
def cov(samples_x, samples_y, ddof=0):
    '''Ковариация - количественный показатель силы
    и направления связи двух элементов.
    Математическое ожидание произведения отклонений СВ от их мат. ожиданий: 
    M([X - M[X])*(Y - M[Y])]
    Диапазон: [-∞,+∞]
    samples : array_like 
        array of values
    ddof : int, optional
        delta degrees of freedom
     
    ddof=0 ковариация популяции (всей генеральной совокупности)
    ddof=1 выборочная ковариация
    
    >>> samples = [168,160,168,168,168], [168,160,168,168,100]
    >>> cov(*samples,ddof=1))
    0.01923076923076923
    # вычисляет выборочную ковариацию
    >>> import numpy as np
    >>> np.cov(*samples,ddof=1)[0][1]
    0.01923076923076923
    '''
    
    assert len(samples_x) == len(samples_y)
    
    numerator = sum_products_deviations(samples_x, samples_y)
    denom = (len(samples_x) - ddof)
    
    return numerator/denom
 
#=============================================================
# центральные моменты 
#=============================================================


#scipy.stats.moment(a, moment=1, axis=0, nan_policy='propagate')[source]
def skew(samples):
    '''Вычисляет ассиметрию выборки, 
    третий центральный момент отклонений от среднего.
    На нормально распределенных данных ассиметрия равна 0.
    
    >>> a = [-100,1,2,3,4,5,6,7,8,9,100]
    >>> print(scipy.stats.skew(a))
    -0.2856173901466769
    >>> print(skew(a))
    -0.28561739014667703
    '''
    m = mean(samples)
    n = len(samples)
    s = std(samples)
    result = sum((((x - m) ** 3) /s ** 3) for x in samples)
    result /= n
    return result 


def kurt(samples,fisher=True):
    '''Вычисляет коэффициент эксцесса,
    который является четвертым центральным моментом отклонений от среднего, 
    нормированный дисперсией.
    Если используется определение Фишера, то из результата вычитается 3.0, 
    чтобы получить 0.0 для нормального распределения. Иначе используется
    определение Пирсона, чтобы получить 3.0 для нормального распределения.
    
    >>> a = [-100,1,2,3,4,5,6,7,8,9,100]
    >>> print(scipy.stats.kurtosis(a))
    2.49963418722474
    >>> print(kurt(a))
    2.4996341872247436
    '''
    
    m = mean(samples)
    n = len(samples)
    s = std(samples)
    result = (sum((((x - m) ** 4) /s ** 4) for x in samples)/n)
    if fisher:
        result -= 3
    
    return result 


#http://medstatistic.ru/theory/hi_kvadrat.html
#=============================================================
# хи-квадрат Пирсона
#=============================================================
def chi2_test(cont,exps=None,ddof=0):
    ''''Критерий хи-квадрат
    Непараметрический односторонний критерий проверки значимости связи 
    между двумя категор. факторами
    H0 (нулевая) гипотеза - отсутствие зависимости между переменными
    '''
    _SMALL = 1e-20
    if exps is None:
        exp = mean(cont)
        chi2 = sum(
            ((obs - exp)**2) / (exp + _SMALL)  for obs in cont
        )
    else:
        chi2 = sum(
            ((obs - exp) ** 2) / (exp + _SMALL) for obs,exp in zip(cont,exps)
        )
    return chi2

# from nltk
def chi_sq(cls, *marginals):
    """Scores ngrams using Pearson's chi-square as in Manning and Schutze
    5.3.3.
    """
    cont = cls._contingency(*marginals)
    exps = cls._expected_values(cont)
    return sum((obs - exp) ** 2 / (exp + _SMALL) for obs, exp in zip(cont, exps))




#χ2 = (40-33.6)2/33.6 + (30-36.4)2/36.4 + (32-38.4)2/38.4 + (48-41.6)2/41.6 = 4.396.

# неверно!!!!
def fisher_exact(*args):
    '''Точный критерий Фишера - 
    выполняет проверку значимости связи между двумя категор. факторами
    (ранее использовался только для небольшого числа наблюдений)
    Используется для сравнения двух относительных показателей, 
    характеризующих частоту определенного признака, имеющего два значения
    Может быть односторонним и двусторонним.
    Является параметрическим аналогом хи-квадрат Пирсона, 
    при этом точный критерий Фишера 
    обладает более высокой мощностью.
    '''
    row = [sum(i)  for i in args]
    col = [sum(i)  for i in zip(*args)]
    
    print(row,col)
    
    numerator = reduce(
        lambda x,y: x * y, 
        [math.factorial(arg) for arg in row + col]
    )
    
    print(numerator)
    
    all = [] 
    
    for arg in args:
        print(arg)
        all.extend(arg)
    print('all',all)
    
    N = sum(all)
    
    denom = reduce(lambda x,y: x * y, [math.factorial(i) for i in all])
    denom *= math.factorial(N)
    return numerator/denom

'''
data = [
    [10,70],
    [2,88]
]
print(fisher_exact(*data))
print()
import scipy.stats
oddsratio, pvalue = scipy.stats.fisher_exact(data)
print(oddsratio, pvalue)
oddsratio, pvalue = scipy.stats.fisher_exact(data,alternative='greater')
print(oddsratio, pvalue)
oddsratio, pvalue = scipy.stats.fisher_exact(data,alternative='less')
print(oddsratio, pvalue)

quit()
'''


#=============================================================
#  мощность критерия
#=============================================================
# Не реализован
def power_():
    '''Вероятность отвергнуть неверную нулевую гипотезу
    Вероятность ошибки второго рода
    Вероятность того, что нулевая гипотеза будет отвергнута, когда альтернативная верна
    Площадь под кривой для альтернативной гипотезы
    '''
    
    obs = obs/n
    exp = exp/n
    sqrt(sum((exp-obs)**2/exp))

#=============================================================
# вычисление значения p_value
#=============================================================
def calc_p_value(t,df):
    '''df = n - 1 где n число наблюдений\переменных
    two-sided pvalue = Prob(abs(t)>tt)
    '''
    return scipy.stats.t.sf(np.abs(t), df) * 2

def calc_p_value2(t,df):
    '''
    t - вычисленная t-статистика
    df = n - 1 где n число наблюдений\переменных'''
    return (1 - scipy.stats.t.cdf(np.abs(t), df)) * 2
# R lang  Python
# qt      scipy.stats.t.ppf
# pt      scipy.stats.t.cdf


# rnorm  scipy.stats.norm.rvs    Генерация случайных выборок
# pnorm  scipy.stats.norm.cdf    Функция распределения вероятностей F(x)
# qnorm  scipy.stats.norm.ppf    Обратная к cdf функция процентной точки (часто используется при проверке статистических гипотез)
# dnorm  scipy.stats.norm.pdf    Плотность распределения вероятностей
#        scipy.stats.norm.sf     Survival function 
#        scipy.stats.norm.isf    Обратная функция к 1-F(x)
#        scipy.stats.norm.stats  Базовые характеристики: среднее, дисперсия, асимметрия, эксцесс

#=============================================================
# критерий T-Student's - для проверки количественных переменных
# Проверяет гипотезу о равенстве математических ожиданий в двух выборках
#=============================================================
def ttest_1samp(samples, mu, ddof=1):
    '''
    Одновыборочный t-критерий
    samples     - выборка
    mu          - мат. ожидание\среднее
    
    >>> import scipy.stats
    >>> sample = [168,160,168,168,168]
    >>> mu = 167
    >>> ttest_1samp(sample, mu,ddof=1)
    -0.3749999999999965
    >>> scipy.stats.ttest_1samp(sample, mu, ddof=1).statistic
    -0.3749999999999965
    '''
    n = len(samples)
    
    numerator = weighted_average(samples) - mu
    v = var(samples,ddof=ddof)
    denom = sqrt(v / float(n))
    
    return numerator/denom


#df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

def ttest(samples_x,samples_y,ddof=0):
    '''Двухвыборочный t-критерий (с поправкой Уэлча)??? для независимых выборок
    с неравными значениями длины и дисперсии.
    
    samples_x,samples_y : array_like 
        arrays of values
    ddof : int, optional
        delta degrees of freedom
    
    >>> samples = [168,160,168,168,168], [168,160,168,168,100]
    >>> import scipy.stats
    >>> test = scipy.stats.ttest_ind(*samples)
    >>> test.statistic
    1.0159443179342342
    >>> ttest(*samples)
    1.0159443179342342
    '''
    
    average_x = weighted_average(samples_x)
    average_y = weighted_average(samples_y)
    disp_x = var(samples_x,ddof=ddof)
    disp_y = var(samples_y,ddof=ddof)
    
    numerator = average_x - average_y
    
    vn1 = disp_x/len(samples_x)
    vn2 = disp_y/len(samples_y)
    
    denom = sqrt(vn1 + vn2)
    
    return numerator/denom

'''
#https://pythonfordatascience.org/welch-t-test-python-pandas/
def welch_dof(x,y):
    dof =( (
        x.var() / x.size + 
        y.var() / y.size
        )**2 / (
            (
                x.var() /x.size)**2 / 
                (x.size-1) + 
                (y.var()/y.size)**2 / 
                (y.size-1)
       )) 
    print(f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")

def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}"
    )
'''




def ttest_equal(samples_x,samples_y,ddof=0):
    '''Двухвыборочный t-критерий для независимых выборок
    с равными значениями длины(размеров) и дисперсии.
    '''
    
    n = len(samples_x)
    average_x = weighted_average(samples_x)
    average_y = weighted_average(samples_y)
    disp_x = var(samples_x,ddof=ddof)
    disp_y = var(samples_y,ddof=ddof)
    
    numerator = average_x -  average_y
    
    pooled_std = sqrt((disp_x + disp_y)/2)
    
    denom = pooled_std * sqrt(2/n)
    
    return numerator/denom


def ttest_equal_var(samples_x,samples_y,ddof=0):
    '''Двухвыборочный t-критерий для независимых выборок
    с равными значениями дисперсий.
    '''
    average_x = weighted_average(samples_x)
    average_y = weighted_average(samples_y)
    disp_x = var(samples_x,ddof=ddof)
    disp_y = var(samples_y,ddof=ddof)
    
    len_x,len_y = len(samples_x),len(samples_y)
    
    numerator = average_x -  average_y
    df = len_x + len_y - 2 # степени свободы
    svar = (len_x - 1) * disp_x + (len_y - 1) * disp_y
    pooled_std = sqrt(svar/df)
    
    denom = pooled_std  * sqrt(1.0/len_x + 1.0/len_y)
    
    return numerator/denom


def t_student(cn_freq,n_freq,c_freq,total):
    '''вероятность совместной встречаемости слов равна произведению 
    вероятностей каждого слова в биграмме'''
    
    p_joint_occurrence  = (c_freq/total)  * (n_freq/total)
    x_mean = cn_freq/total # выборочное среднее  
    res = (x_mean - p_joint_occurrence)/sqrt(x_mean/total)    
   
#=============================================================
# коэффициент корреляции Пирсона
#=============================================================
def pearsonr(samples_x, samples_y):
    '''Коэффициент корреляции Пирсона.
    Математическое ожидание произведения случайных величин.
    диапазон [-1,1]
    0.0  -      корреляция отсутствует
    0.29 -      слабая связь
    0.30 - 0.69 среднняя связь
    0.70 - 1.0  сильная связь
    Показывает силу линейной связи между двумя количественными показателями
    
    samples_x,samples_y : array_like 
        arrays of values
    
    >>> samples = [168,160,168,168,168], [168,160,168,168,100]
    >>> pearsonr(*samples)[
    -0.13543408472149038
    # Normalized covariance matrix
    >>> import numpy as np
    >>> np.corrcoef(*samples)[0,1]
    -0.13543408472149038
    >> import scipy.stats
    >>> coeff,p_value = scipy.stats.pearsonr(*samples)  
    >>> coeff,p_value
    0.1354340847214904 0.8280885880470833
    '''
    
    assert len(samples_x) == len(samples_y)
    # суммы квадратов отклонений от средней
    v1 = sum_squared_deviations(samples_x) 
    v2 = sum_squared_deviations(samples_y) 
    # сумма произведений отклонений от среднего
    numerator = sum_products_deviations(samples_x,samples_y)
    # корень квадратный из произведения сумм квадратов отклонений от среднего
    denom = (v1 * v2) ** 0.5   
    # result = summa/math.sqrt(sum2_x * sum2_y)
    
    return numerator / denom 


def pearsonr2(samples_x, samples_y):
    '''Коэффициент корреляции Пирсона
    диапазон [-1,1]
    
    result = (
        cov(samples_x, samples_y) /
        ((var(samples_x) * var(samples_y)) ** 0.5) 
        ) 
    
    >>> samples = [168,160,168,168,168], [168,160,168,168,100]
    >>> pearsonr(*samples)[
    -0.13543408472149038
    
    '''
    
    assert len(samples_x) == len(samples_y)
    numerator = cov(samples_x, samples_y)
    denom = std(samples_x) * std(samples_y)
    
    return numerator / denom 


#https://pastebin.com/SQpj0SUw - еще здесь вариант

def pearsonr3(x, y):
    assert len(x) == len(y)
    
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(map(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den


def pearsonr4(x, y):
   ''' Compute Pearson Correlation Coefficient. '''
   # Normalise X and Y
   x,y = np.array(x,dtype=float),np.array(y,dtype=float)
   x -= x.mean(0)
   y -= y.mean(0)
   # Standardise X and Y
   x /= x.std(0)
   y /= y.std(0)
   # Compute mean product
   return np.mean(x*y)


def pearsonr5(data1, data2):
    '''
    #http://qaru.site/questions/49623/calculating-pearson-correlation-and-significance-in-python
    '''
    
    M = len(data1)

    sum1 = 0.
    sum2 = 0.
    for i in range(M):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    
    for i in range(M):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / M) ** 0.5
    std2 = (var_sum2 / M) ** 0.5
    cross_mean = cross_sum / M

    return (cross_mean - mean1 * mean2) / (std1 * std2)


#=============================================================
# коэффициент корреляции Спирмена
#=============================================================
''' Варианты функций ранжирования:
average: average rank of group
min: lowest rank in group
max: highest rank in group
first: ranks assigned in order they appear in the array
dense: like ‘min’, but rank always increases by 1 between group
'''

def spearmanr(samples_x, samples_y,method='ordinal'):
    '''Коэффициент Спирмена
    диапазон [-1,1]
    Показывает силу монотонной взаимосвязи между количественными величинами.
    Устойчив к выбросам.
    '''
    
    assert len(samples_x) == len(samples_y)
    
    def rank(seq):
        '''функция ранжирования
        То же самое как df.rank(method='first')  
        ranks assigned in order they appear in the array
        '''
        
        #x = argsort(seq)
        #y = [x.index(idx) for idx,num in enumerate(seq)]
        
        y = scipy.stats.rankdata(seq,method=method).tolist()
        
        return y
        
    rx = rank(samples_x)
    ry = rank(samples_y)
    
    n = len(samples_x)
    dsq = sum((x - y) **2 for x,y in zip(rx, ry))
    numerator = 6 * dsq
    denom = n * (n ** 2 - 1)
    
    return 1 - (numerator/denom)

#https://stackoverflow.com/questions/47562775/scipy-spearman-correlation-coefficient-is-nan-in-some-cases/47563047#47563047
def spearmanr2(x, y): 
    """ `x`, `y` --> pd.Series""" 
    
    x, y = pd.Series(x), pd.Series(y)
    assert x.shape == y.shape 
    rx = x.rank(method='average') 
    ry = y.rank(method='average') 
    d = rx - ry 
    dsq = np.sum(np.square(d)) 
    n = x.shape[0] 
    numerator = 6. * dsq
    denom = n * (n ** 2 - 1.)
    coef = 1. - (numerator/denom)
    return coef 



#=============================================================
# Евклидова норма => длина вектора
# cкалярное произведение вектора самого на себя — это квадрат его длины
#=============================================================
def norm(v):
    return sqrt(sum(i ** 2 for i in v))

#=============================================================
# манхэттенская норма - сумма модулей координат вектора
#=============================================================
def norm2(v):
    return sqrt(sum(abs(i) for i in v))

#=============================================================
# евклидово расстояние
#=============================================================
def euclidean_distance(v1, v2):
    return sum((x - y) ** 2 for x, y in zip(v1, v2)) ** 0.5
#тоже самое
def euclidean_distance2(v1, v2):
    return sqrt(sum((x - y) ** 2 for x,y in zip(v1, v2)))

def euclidean_distance_array(v1, v2):
    #import numpy as np
    v1, v2 = np.array(v1), np.array(v2)
    return np.sqrt(np.sum((v1 - v2) ** 2))

#http://www.michurin.net/computer-science/scipy/document_distance.html
def euclidean_distance_array_norm(v1, v2):
     # Строго говоря, эту нормолизацию можно было сделать
    # эффективней, если использовать параметр axis в la.norm()
    #import numpy as np
    #import numpy.linalg as la
    v1, v2 = np.array(v1), np.array(v2)
    v1, v2 = v1/la.norm(v1),v2/la.norm(v2)
    return la.norm(v1 - v2)

#=============================================================
# манхэттенское расстояние
#=============================================================
def manhattan_distance(v1, v2):
    return sum(abs(x - y) for (x, y) in zip(v1, v2)) 

#=============================================================
# расстояние Чебышева
#=============================================================
def chebyshev_distance(v1, v2):
    return max((x - y) for (x, y) in zip(v1, v2))

def chebyshev_distance_array(v1,v2):
   
    v1, v2 = np.array(v1), np.array(v2)
    return max(v1 - v2)

#=============================================================
# расстояние минковского
#=============================================================
def minkowski(a, b, p=2):
    assert len(a) == len(b) 
    delta = pow(sum((abs(x-y)**p) for x,y in zip(a,b)), 1/p)
    return delta


def minkowski_distance_array(v1,v2,p=2):
    #import numpy as np
    #import numpy.linalg as la
    v1, v2 = np.array(v1), np.array(v2)
    delta = la.norm(v1 - v2,ord=p)
    return delta

#=============================================================
# расстояние Хэмминга
#=============================================================
def hamming_distance(a,b):
    assert len(a) == len(b)
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return dist

def hamming_distance(a,b): 
    '''hamming distance'''
    assert len(a) == len(b)
    result = sum(1 for t in zip(a,b) if t[0] != t[1])
    
    return result

#=============================================================
#косинус угла между векторами x и y вычисляется как отношение скалярного произведения к произведению их норм.
#=============================================================
# косинусная дистанция
def cosine_dist_fast(v1, v2):
     res = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
     return 1.0 - res

# косинусная дистанция
def cosine_dist(v1, v2):
    res = dot(v1, v2) / ((norm(v1) * norm(v2)))
    return 1.0 - res

# косинусная дистанция
def cosine_dist_from_scipy(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return scipy.spatial.distance.cosine(v1, v2)

#=============================================================
# расстояние Жаккарда
#=============================================================
def jaccard_distance(a,b):
    '''
    Вычисляет расстояние Жаккарда
    >>> s1='это очень похожее предложение'
    >>> s2='это очень похожее прелдожение'
    # на уровне букв
    >>> jaccard_distance(s1,s2)
    0.0   
    >>> nltk.jaccard_distance(set(s1),set(s2))
    0.0
    # на уровне слов
    >>> jaccard_distance(s1.split(),s2.split())
    0.4
    >>> nltk.jaccard_distance(set(s1.split()),set(s2.split()))
    0.4
    >>> list(nltk.ngrams(s1.split(), n=2))
    [('это', 'очень'), ('очень', 'похожее'), ('похожее', 'предложение')]
    >>> 
    '''
    
    if not isinstance(a,set) or not isinstance(b,set):
        a,b = set(a), set(b)
    
    common = a & b
    len_a,len_b,len_c = len(a), len(b), len(common)
    
    score = len_c / (len_a + len_b - len_c)
    return round(1 - score,2)



#=============================================================
# Коэффициент Танимото - описывает степень схожести двух множеств
# Вычисляется по формуле, в которой 
# Na – количество элементов в A, 
# Nb – количество элементов в B, 
# Nc – количество элементов в пересечении C
#=============================================================
def tanimoto(a,b):
    '''расширенного коэффициента Жаккара
    Исполльзуется для компонентов булевых векторов, т.е.  
    принимающих только два значения 0 и 1
    '''
    #c = [v for v in a if v in b]
    if not isinstance(a,set) or not isinstance(b,set):
        a,b = set(a), set(b)
    
    common = a & b
    len_a,len_b,len_c = len(a), len(b), len(common)
    
    score = len_c/(len_a + len_b - len_c)
    return score


#=============================================================
# мера сходства Сёренсена
#=============================================================
def sorensen(a, b):
    '''Мера сходства - коэффициент Сёренсена - 
    https://ru.wikipedia.org/wiki/Коэффициент_Сёренсена
    ''' 
    
    if not len(a) or not len(b):
        return 0.0 
        
    if not isinstance(a,set) or not isinstance(b,set):
        a,b = set(a), set(b)    
        
    common = a & b
    len_a,len_b,len_c = len(a), len(b), len(common)
    
    score = (2.0 * len_c) / (len_a + len_b)
    return score


#=============================================================
# мера Дайса
#=============================================================
def dice(a, b):
    """dice coefficient 2nt/na + nb."""
    
    if not isinstance(a,set) or not isinstance(b,set):
        a,b = set(a), set(b)    
        
    common = a & b
    len_a,len_b,len_c = len(a), len(b), len(common)
    
    score = (2.0 * len_c) / (len_a + len_b)
    return score

#dice([1,2,3,4,5,6], [1,2,3,4,5,6]) # 1.0
#dice([1,2,3,4,5,6], [1,2,3,4,4,4]) # 0.8

''''
""" more orthodox and robust implementation """
def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    if not len(a) or not len(b): return 0.0
    if len(a) == 1:  a = a + '.'
    if len(b) == 1:  b = b + '.'
    
    a_bigram_list=[]
    for i in range(len(a)-1):
      a_bigram_list.append(a[i:i+2])
    b_bigram_list=[]
    for i in range(len(b)-1):
      b_bigram_list.append(b[i:i+2])
      
    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0/(len(a_bigrams) + len(b_bigrams))
    return dice_coeff


""" duplicate bigrams in a word should be counted distinctly
(per discussion), otherwise 'AA' and 'AAAA' would have a 
dice coefficient of 1...
"""

def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
    
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
    
    a_bigram_list.sort()
    b_bigram_list.sort()
    
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
    
    score = float(matches)/float(lena + lenb)
    return score
'''


#=============================================================
# BM25
#=============================================================
k1 = 1.2
k2 = 100
b = 0.75
R = 0.0


def score_BM25(n, f, qf, r, N, dl, avdl):
    K = compute_K(dl, avdl)
    first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)
    return first * second * third


def compute_K(dl, avdl):
    return k1 * ((1-b) + b * (float(dl)/float(avdl)) )


#=============================================================
#  tfc–weighting
#=============================================================
import scipy.sparse as sparse
 
def tfc(freqs_arr):
    '''
    where Fij is the frequency of term i in document j;
    Ni is the number of documents containing the i-th term; 
    N is the total number of documents and M is the number of terms.'''
    
    if sparse.isspmatrix_csr(freqs_arr):
        v = freqs_arr
    else:
        v = sparse.csr_matrix(freqs_arr)
    l = sparse.csc_matrix(
        np.log10(v.shape[0]/(v > 0).sum(axis=0)).reshape(-1)
    )
    v = v.multiply(l)
    denom = v.multiply(v).sum() ** 0.5
    return v / denom

#=============================================================
#  Шкалирование\нормировка данных
#=============================================================
def min_max_scale(samples):
    samples = samples.copy()
    max_ = max(samples)
    min_ = min(samples)
    for i in range(len(samples)):
        samples[i] -= min_
        samples[i] /= max_ - min_  
    
    return samples


def scale(samples, with_mean=True, with_std=True,copy=True):
    '''
    Нормировка данных к нулевому среднему и единичной дисперсии
    
    >>> print(scale([1,255,46,24]))
    [-0.7936814559186198, 1.7106053739364042, -0.3500085923616274, -0.566915325656157]
    >>> import sklearn.preprocessing
    >>> print(sklearn.preprocessing.scale([1,255,46,24]))
    [-0.79368146  1.71060537 -0.35000859 -0.56691533]
    '''
    if copy:
        samples = samples.copy()
    if with_mean:
        mean_ = mean(samples)
    if with_std:
        var_ = std(samples) 
    
    for i in range(len(samples)):
        if with_mean:
            samples[i] -= mean_
        if with_std:
            samples[i] /= var_  
    
    return samples

'''
scaler = sklearn.preprocessing.StandardScaler()
print(scaler.fit([[1,255,46,24]]))
print(scaler.mean_)
print(scaler.var_)
quit()
'''

#=============================================================
#  Оценки качества модели
#=============================================================



'''
# критерий  Фишера

y_mean = mean(y_true) 
numerator = sum((t[0] - t[1]) ** 2 for t in zip(y_pred,y_true))
denom = sum((t[0] - t[1]) ** 2 for t in zip(y_true,y_mean))
'''

'''
import sklearn.feature_selection
def main():
    from sklearn import svm
    from sklearn.datasets import samples_generator
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.preprocessing import MinMaxScaler

    X, y = samples_generator.make_classification(
        n_samples=1000, 
        n_informative=5, 
        n_redundant=4, 
        random_state=_random_state
    )
    anova_filter = SelectKBest(f_regression, k=5)
    scaler = MinMaxScaler()
    clf = svm.SVC(kernel='linear')

    steps = [scaler, anova_filter, clf]
'''    

#sklearn.feature_selection.f_classif
#sklearn.feature_selection.f_regression 

def fisher_score(*args):
    '''F-тест Фишера
    определяет статистическую значимость коэффициента детерминации R2
    Если связи есть - дисперсия регресссии будет всегда больше дисперсии остатков,
    если связи нет - дисперсия регрессии и дисперсия остатков будет одинаковыми
    '''
    # n-объём выборки, k-количество параметров модели
    # F = ESS / (k-1) / RSS / (n-k)
    k = len(args)
    #alldata = np.concatenate(args)
    alldata = []
    for arg in args:
        alldata.extend(args)
    
    n = len(alldata)
    #r2 = r2_score(y_true,y_pred)
    #numerator = r2 / (k - 1)
    #denom = (1 - r2) / (n - k)
    
    sstot = _sum_of_squares(alldata) - (_square_of_sums(alldata) / n)
    ssbn = 0
    for a in args:
        ssbn += _square_of_sums(a - offset) / len(a)

    ssbn -= _square_of_sums(alldata) / n
    sswn = sstot - ssbn
    F = ESS /( k - 1) / RSS / (n - k)
    return F

#=============================================================
# регрессионные метрики
#=============================================================
def loss(y_true,y_pred):
    '''squared error'''
    loss = (y_true - y_pred) ** 2
    return loss

def _mse_by_rows(y_true,y_pred):    
    out = []
    for i in range(len(y_true)):
        row = []
        for j in range(len(y_true[0])):
            row.append(loss(y_true[i][j],y_pred[i][j]))
        out.append(sum(row)/len(row)) 
    return out 
    
    
def _mse_by_cols(y_true,y_pred):    
    out = []
    for i in range(len(y_true[0])):
        col = []
        for j in range(len(y_true)):
            col.append(loss(y_true[j][i],y_pred[j][i]))
        out.append(sum(col)/len(col)) 
    return out


 def mean_squared_error(Y_true,Y_pred):
    '''
    >>> import numpy as np
    >>> y_true = [1,2,3,4,5,6,7,8,9]
    >>> y_pred = [1,3,5,6,7,4,9,0,8]
    >>> np.square(np.array(y_true) - np.array(y_pred)).mean(axis=-1) # по всем столбцам и строкам
    9.555555555555555
    >>> 
    >>> import sklearn.metrics as sk
    >>> sk.mean_squared_error(np.array(y_true), np.array(y_pred))
    9.555555555555555
    >>> 
    '''
    
    return sum(
        map(lambda t:loss(t[0], t[1]), zip(Y_true,Y_pred))
    )/len(Y_true)




# общая сумма квадратов отклонений
# Explained sum of squares: ESS объясненная или регрессионная сумма квадратов
# Residual sum of squares:  RSS необъясненная или остаточная сумма квадрато
# Total sum of squares:     TSS = ESS + RSS  # равенство не обязательно

# tss = np.sum((y_true - y_true.mean()) ** 2)  ; df = n - 1
# rss = np.sum((y_true - y_pred) ** 2)         ; df = n - k, где n - объем выборки, k - число параметров
# ess = np.sum((y_pred - y_true.mean()) ** 2)  ; df = k - 1

def r2_score(y_true,y_pred):   
    '''Коэффициент детерминации
    это доля дисперсии зависимой переменной, объясняемая 
    рассматриваемой моделью зависимости.
    Это единица минус доля необъяснённой дисперсии: 
    1 - Residual sum of squares/Total sum of squares.
    
    Принимает значение на промежутке [-∞,1] и 
    чем ближе к 1, тем сильнее зависимость
    >>> y_true, y_pred = [3, -0.5, 2, 7],[2.5, 0.0, 2, 8]
    >>> sklearn.metrics.r2_score(y_true, y_pred)
    0.9486081370449679
    >>> r2_score(y_true,y_pred)
    0.9486081370449679
    '''
    y_true, y_pred = np.array(y_true),np.array(y_pred)
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
     
    return 1 - (rss / tss)   
    

def r2_score2(y_true,y_pred):
    '''Коэффициент детерминации
    
    >>> y_true, y_pred = [3, -0.5, 2, 7],[2.5, 0.0, 2, 8]
    >>> sklearn.metrics.r2_score(y_true, y_pred)
    0.9486081370449679
    >>> r2_score2(y_true,y_pred)
    0.9486081370449679
    '''
    
    y_mean = mean(y_true) 
    numerator = sum((t[0] - t[1]) ** 2 for t in zip(y_true,y_pred))
    denom =     sum((y - y_mean) ** 2 for y in y_true)
    return 1 - (numerator/denom)   
 

def r2_adjusted_score(y_true,y_pred):
    '''Исправленный коэффициент детерминации
    k - число параметров модели
    '''
    
    r2 = r2_score(y_true,y_pred)
    k = 2  # ???
    param1 = 1 - (1 - r2)
    param2 = (n - 1) / (n - k)
    return param1 * param2 
   


#=============================================================
# кластерные метрики
#=============================================================
def silhouette_score(x, y, metric='euclidean'): 
    # a — среднее внутрикластерное расстояние, 
    # b — среднее межкластерное расстояние.
    #s = b - a / (max(a,b)) # силуэтная метрика
    pass


def mutual_info_score():
    pass


def durbin_watson():
    '''Критерий Дарбина—Уотсона (или DW-критерий) — статистический критерий, 
    используемый для тестирования автокорреляции первого порядка элементов исследуемой последовательности'''
    pass


 

#=============================================================
# классификационые метрики
#=============================================================
'''
accuracy_score
jaccard_similarity_score
zero_one_loss
log_loss
roc_auc_score
f1_score
recall_score
precision_score
matthews_corrcoef
...
'''
#=============================================================
# коэффициент корреляции Мэтьюса
#=============================================================

# 
def matthews_corrcoef(y_true, y_pred,eps=1e-15):
    '''
    Коэффициент корреляции Мэтьюса
    Мера качества для бинарной классификации
    диапазон [-1,1]
    >>> y_true,y_pred = [+1, +1, +1, -1], [+1, -1, +1, +1]
    >>> sklearn.metrics.matthews_corrcoef(y_true,y_pred)
    -0.3333333333333333
    >> matthews_corrcoef(y_true,y_pred)
    -0.33333333333333326
    '''
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    n = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denom = np.sqrt(n)

    return numerator / (denom + eps)

# не совсем точно
def matthews_corrcoef2(y_true, y_pred,eps=1e-15):
    '''http://www.machinelearning.ru/wiki/index.php?title=Корреляция_Мэтьюса'''
    
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    
    n = tn + tp + fn + fp
    s = (tp + fn) / n
    p = (tp + fp) / n
    numerator = tp / n - (s * p)
    denom =  np.sqrt(p * s * (1 - s) * (1 - p)) 
    coef = numerator / (denom + eps)
    return coef


if __name__ == "__main__":
    np.random.seed(42)
    
    import scipy.stats
    
    samples = [168,160,168,168,168]
    
    
    print("-------ДИСТАНЦИЯ ЕВКЛИДА--------")
    
    from scipy.spatial import distance
    
    x,y = [1,2,3,4,5,6,7,8,9,0], [0,1,2,3,4,5,6,7,8,9]
    print(distance.euclidean(x,y))
    print(euclidean_distance(x,y))
    print(euclidean_distance2(x,y))
    print(euclidean_distance_array(x,y))
    print('---нормированная---')
    print(euclidean_distance_array_norm(x,y))
    
    
    print("-------ДИСТАНЦИЯ ТАНИМОТО--------")
    
    print(tanimoto(list('hello world'),list('hello worl')))
    # вычисляет расстояние только между bool array's
    #print(scipy.spatial.distance.rogerstanimoto(list('hello world'),list('hello worl')))
    
    print("-------ДИСТАНЦИЯ МИНКОВСКОГО--------")
    
    
    vectors = [1, 1, 1], [3, 3, 0]
    print(minkowski(*vectors)) 
    print(minkowski_distance_array(*vectors ))
    print(scipy.spatial.minkowski_distance(*vectors))
    print(distance.minkowski(*vectors))
    
        
    print("-------ХИ КВАДРАТ ПИРСОНА--------")
    
    print(chi2_test(samples)) 
    stat,pvalue =  scipy.stats.chisquare(samples)
    print(stat) 
    obs = [16, 18, 16, 14, 12, 12] 
    exp = [16, 16, 16, 16, 16, 8]
    print(chi2_test(obs,exp,ddof=1)) 
    stat,pvalue = scipy.stats.chisquare(obs,exp,ddof=1)
    print(stat) 
    
    print("-------СТАНДАРТНАЯ ОШИБКА СРЕДНЕГО--------")
    sample = [16, 18, 16, 14, 12, 12] 
    print(scipy.stats.sem(sample))
    print(sem(sample))
    
    
    print("-------СТАНДАРТНОЕ ОТКЛОНЕНИЕ--------")
    from statistics import stdev,pstdev
    print('=sample standard deviation=')
    print(stdev(samples))
    print(np.std(samples,ddof=1))
    print(std(samples,ddof=1))

    print('=population standard deviation=')
    print(pstdev(samples))
    print(np.std(samples))
    print(std(samples))
    
    print("-------  КОЭФФЦИЕНТ ДЕТЕРМИНАЦИИ--------") 
    y_true, y_pred = [3, -0.5, 2, 7],[2.5, 0.0, 2, 8]
    print(sklearn.metrics.r2_score(y_true, y_pred)) 
    print(r2_score(y_true, y_pred)) 
    print(r2_score2(y_true, y_pred))
    y_true,y_pred = [1, 2, 3], [3, 2, 1]
    print(sklearn.metrics.r2_score(y_true, y_pred)) 
    print(r2_score(y_true, y_pred)) 
    print(r2_score2(y_true, y_pred))
    
    print("-------ДИСПЕРСИЯ--------")
    '''
    В стандартной статистической практике ddof=1 обеспечивает несмещенную оценку дисперсии гипотетической бесконечной совокупности. 
    ddof=0 предоставляет оценку максимального правдоподобия дисперсии для нормально распределенных переменных.
    '''
    from statistics import variance,pvariance
    print('=sample variance=')
    print(var(samples,ddof=1))
    print(np.var(samples,ddof=1))
    print(variance(samples))
    print('=population variance=')
    print(var(samples))
    print(np.var(samples))
    print(pvariance(samples))
    
    print("-" * 25)
    
    samples = scipy.stats.norm.rvs(loc=5, scale=1, size=100)
    from statistics import variance,pvariance
    print('=sample variance=')
    print(var(samples,ddof=1))
    print(np.var(samples,ddof=1))
    print(variance(samples))
    print('=population variance=')
    print(var(samples))
    print(np.var(samples))
    print(pvariance(samples))
    
    print("-------МЕРЫ ЦЕНТРАЛЬНОЙ ТЕНДЕНЦИИ--------")
    x = [1,4,9,16,25,36,49,16]
    print(scipy.stats.describe(x))
    print('kurt:',scipy.stats.kurtosis(x))
    print('skew:',scipy.stats.skew(x))
    print('kurt:',kurt(x))
    print('skew:',skew(x))
    
    print("-------КОЭФФИЦИЕНТ ВАРИАЦИИ--------")
    print(scipy.stats.variation(samples))
    print(variation(samples)) 

    print("-" * 25)
    
    print("-------КОВАРИАЦИЯ--------")
    samples = [1,2,3,4,5,6,7,8], [1,4,9,16,25,36,49,16]
    print(cov(*samples,ddof=1))
    # вычисляет выборочную ковариацию
    print(np.cov(*samples,ddof=1)[0][1])
    
    print("-" * 25)
    
    print("-------КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ПИРСОНА--------")
    
    samples = [1,2,3,4,5,6,7,8], [1,4,9,16,25,36,49,16]
    coeff = pearsonr(*samples)
    print(coeff)
    
    coeff = pearsonr2(*samples)
    print(coeff)
    
    coeff = pearsonr3(*samples)
    print(coeff)
    
    coeff = pearsonr4(*samples)
    print(coeff)
     
    coeff = pearsonr5(*samples)
    print(coeff)
    # Normalized covariance matrix
    coeff = np.corrcoef(*samples)
    print(coeff) # матрица
    print(coeff[1,0])

    coeff,p_value = scipy.stats.pearsonr(*samples)  
    print(coeff,p_value)
    
    import pandas as pd
    df = pd.DataFrame(list(zip(*samples)))
    print(df.corr().iloc[1,0])  # 0.7453559924999299
    

    print("-" * 25)
    print("-------КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ СПИРМЕНА--------")
    print(spearmanr(*samples))
    print(spearmanr(*samples,method="average"))
    print(spearmanr2(*samples))
    coeff,p_value = scipy.stats.spearmanr(*samples)
    print(coeff,p_value)
    
    print("-------КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ МЭТЬЮСА--------")
    y_true,y_pred = [+1, +1, +1, -1], [+1, -1, +1, +1]
    print(sklearn.metrics.matthews_corrcoef(np.array(y_true),np.array(y_pred)))
    print(matthews_corrcoef(np.array(y_true),np.array(y_pred)))
    print(matthews_corrcoef2(np.array(y_true),np.array(y_pred)))
    
    print("-------ТЕСТ STUDENT--------")
    test = ttest_1samp([168,160,168,168,168], 167,ddof=1)
    print(test) 

    #Calculate the T-test for the mean of ONE group of scores.
    test = scipy.stats.ttest_1samp([168,160,168,168,168], 167)
    print('statistic:',test.statistic)
    print('pvalue:',test.pvalue)
    print('calc_p_value :',calc_p_value(test.statistic, 5-1))
    print('calc_p_value2:',calc_p_value2(test.statistic,5-1))
    print('для двух групп')
    # вычисляет для двух групп наблюдений
    # Calculate the T-test for the means of two independent samples of scores
    
    test = scipy.stats.ttest_ind(*samples)
    print('statistic:',test.statistic)
    print('pvalue:',test.pvalue)
    print('calc_p_value :',calc_p_value( test.statistic,(len(samples[0]) - 1) * 2 ))
    print('calc_p_value2:',calc_p_value2(test.statistic,(len(samples[0]) - 1) * 2 ))
    print(ttest_equal(*samples,ddof=1))
    print(ttest_equal_var(*samples,ddof=1))
    
    # If False, perform Welch’s t-test, which does not assume equal population variance
    print('-------С поправкой Уэлча---------')
    test = scipy.stats.ttest_ind(*samples,equal_var=False) 
    print(test.statistic)
    print(ttest(*samples,ddof=1))

    rvs1 = scipy.stats.norm.rvs(loc=5, scale=1, size=100)
    rvs2 = scipy.stats.norm.rvs(loc=5, scale=1, size=100)
    rvs3 = scipy.stats.norm.rvs(loc=5, scale=2, size=100)
    
    print("--------equal_var=True------" )
    test = scipy.stats.ttest_ind(rvs1,rvs2) 
    print(test.statistic)
    print(ttest_equal(rvs1.tolist(),rvs2.tolist(),ddof=1))
    print(ttest_equal_var(rvs1.tolist(),rvs2.tolist(),ddof=1))
    
    print("--------equal_var=False--С поправкой Уэлча----" )
    test = scipy.stats.ttest_ind(rvs1,rvs3,equal_var=False) 
    print(test.statistic)
    print(ttest(rvs1.tolist(),rvs3.tolist(),ddof=1)) 
    
    
    print("-" * 30)
    np.random.seed(42)
    rnorm1 = scipy.stats.norm.rvs(size=100)
    rnorm2 = np.concatenate([
        rnorm1[:70] **2, 
        scipy.stats.norm.rvs(size=30)]
    )
    
    rpareto1 = scipy.stats.pareto.rvs(1,size=100)
    rpareto2 = np.concatenate([
        rpareto1[:70] **2,
        scipy.stats.pareto.rvs(1,size=30)]
    )
    print('---------Тест на нормально распределенных данных------------')
    print("------pearsonr-------")
    samples = rnorm1.tolist(),rnorm2.tolist()
    coeff = pearsonr(*samples)
    p_value = calc_p_value2(coeff, (len(samples[0]) * 2) - 2)
    print(coeff)
    coeff,p_value = scipy.stats.pearsonr(*samples)
    print(coeff,p_value)
    print("------spearmanr-------")
    samples = rnorm1.tolist(),rnorm2.tolist()
    coeff = spearmanr(*samples)
    print(coeff)
    coeff,p_value = scipy.stats.spearmanr(*samples)
    print(coeff,p_value)
    
    print('---------Тест на данных распределенных по закону Парето ------------')
    print("------pearsonr-------")
    coeff = pearsonr(rpareto1.tolist(),rpareto2.tolist())
    print(coeff)
    coeff,p_value = scipy.stats.pearsonr(rpareto1,rpareto2)
    print(coeff,p_value)
    print("------spearmanr-------")
    coeff = spearmanr(rpareto1.tolist(),rpareto2.tolist())
    print(coeff)
    coeff,p_value = scipy.stats.spearmanr(rpareto1,rpareto2)
    print(coeff,p_value)
    
    print('-----Тест на нормальность----------------------')
    statistic,p_value =  scipy.stats.shapiro(rnorm1) # нулевая гипотеза не отвергнута - данные распределены по нормальному закону
    print(statistic,p_value)
    statistic,p_value = scipy.stats.kstest(rnorm1,'norm')
    print(statistic,p_value)
   
    print('-------------------------')
    statistic,p_value =  scipy.stats.shapiro(rpareto1) # нулевая гипотеза отвергнута - данные распределены не по нормальному закону
    print(statistic,p_value)
    statistic,p_value = scipy.stats.kstest(rpareto1,'norm')
    print(statistic,p_value)
   
    
    #print(scipy.stats.kstest(scipy.stats.norm.rvs(size=100), 'norm'))

    print('-----------ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ----------------------')
    a = range(10,14)
    print(mean_confidence_interval(a))
    print(mean_confidence_interval2(a))
