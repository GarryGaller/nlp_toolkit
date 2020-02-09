with open(r'D:\INSTALL\Python3\PROJECTS\tri.txt') as f:
    s = f.read()

trigrams = []
count = 0

for l in s.splitlines():
    print(l)
    n, w1, w2, w3 = l.split('\t')
    n = int(n)
    count += n
    trigrams.append((count, (w1, w2, w3)))

from collections import namedtuple

#----------------------------
# копипаста 

def generate_bigrams(trigrams):
    # Dictionary will contain tuple of first (inclusive) and last (exclusive) index in trigrams of that bigram
    bigrams = {}
    last_bigram = None
    first_index = None

    for i, (c, (w1, w2, w3)) in enumerate(trigrams):
        bigram = (w1, w2)
        if bigram != last_bigram or i == len(trigrams) - 1:
            if last_bigram is not None:
                # if this isn't the first bigram, record details of last bigram
                bigrams[last_bigram] = (first_index, i)
            first_index = i
            last_bigram = bigram
    return bigrams
bigrams = generate_bigrams(trigrams)

#iteratively prune dead-ends
while True:
    oldlen = len(trigrams)
    def is_not_dead_end(t):
        try:
            bigram_data = bigrams[(t[1][1], t[1][2])]
        except KeyError:
            return False
        return True # bigram_data[1] > bigram_data[0] + 1
    trigrams = [t for t in trigrams if is_not_dead_end(t)]
    bigrams = generate_bigrams(trigrams)
    if len(trigrams) == oldlen:
        break

# Draws randomly from a discrete cumulative CDF.
# the function cumulative_cdf_func takes in integer i in
# [0, N] and returns an integer representing the unnormalized
# probability that X is less than or equal to i.
def randomWithPDF(cumulative_cdf_func, N):
    K = cumulative_cdf_func(N - 1)
    from random import randint
    Z = randint(0, K - 1)
    # find largest i such that Z < cumulative_cdf_func(i)
    if Z < cumulative_cdf_func(0):
        return (0, float(cumulative_cdf_func(0)) / float(K))
    j = 0
    i = N - 1
    while j + 1 < i:
        k = (j + i) / 2
        if Z < cumulative_cdf_func(k):
            i = k
        else:
            j = k
    return (i, float(cumulative_cdf_func(i) - cumulative_cdf_func(i - 1)) / float(K))

def getRandomTrigram():
    i, p = randomWithPDF(lambda i: trigrams[i][0], len(trigrams))
    return (trigrams[i][1], p)

def getNextRandomTrigram(trigram):
    bigram = (trigram[1], trigram[2])
    try:
        first, last = bigrams[bigram]
    except KeyError:
        raise ValueError
    if first == 0:
        c0 = 0
    else:
        c0 = trigrams[first - 1][0]
    i, p = randomWithPDF(lambda i: trigrams[first + i][0] - c0, last - first)
    return (trigrams[first + i][1], p)

def getRandomSequence(target_entropy, max_iters=1000):
    import math
    trigram, p = getRandomTrigram()
    
    sequence = list(trigram)
    entropy = -math.log(p)
    iters = 1
    
    while entropy < target_entropy and iters < max_iters:
        try:
            trigram, p = getNextRandomTrigram(trigram)
        except ValueError:
            print('Warning: sequence terminated before desired entropy was reached')
            break
        
        sequence.append(trigram[2])
        entropy -= math.log(p)
        iters += 1
    return sequence, entropy

sequence, entropy = getRandomSequence(40)
print("Random sequence: %s" % ' '.join(sequence))
print("Entropy (in bits): %f" % entropy)
