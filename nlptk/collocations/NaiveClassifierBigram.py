# копипаста

def build_topn_best_words(self):
        word_fd = FreqDist()
        label_word_fd = ConditionalFreqDist()
        positivecount = 0
        negativecount = 0
        with open(r"..\polarityData\TweetCorpus\training.1600000.processed.noemoticon.csv", "rb") as f:
            reader = csv.reader(f)
            for row in reader:
                # Positive sentiment tweets
                if row[0] == "4" and positivecount < self.corpuslength:
                    tweet = row[5]
                    tokens = WhitespaceTokenizer().tokenize(tweet)
                    # print tweet
                    for token in tokens:
                        word_fd.inc(token.lower())
                        label_word_fd["pos"].inc(token.lower())
                    positivecount += 1
                # Negative sentiment tweets
                if row[0] == "0" and negativecount < self.corpuslength:
                    tweet = row[5]
                    tokens = WhitespaceTokenizer().tokenize(tweet)
                    # print tweet
                    for token in tokens:
                        word_fd.inc(token.lower())
                        label_word_fd["neg"].inc(token.lower())
                    negativecount += 1

        # print word_fd
        # print label_word_fd

        pos_word_count = label_word_fd["pos"].N()
        neg_word_count = label_word_fd["neg"].N()
        total_word_count = pos_word_count + neg_word_count
        print "Positive Word Count:", pos_word_count, "Negative Word Count:", neg_word_count, "Total Word count:", total_word_count

        word_scores = {}
        for word, freq in word_fd.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(label_word_fd["pos"][word], (freq, pos_word_count), total_word_count)
            neg_score = BigramAssocMeasures.chi_sq(label_word_fd["neg"][word], (freq, neg_word_count), total_word_count)
            word_scores[word] = pos_score + neg_score

        best = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:10000]
        self.bestwords = set([w for w, s in best])
        print "Best Words Count:", len(self.bestwords)  # , 'Best Words Set:', self.bestwords
