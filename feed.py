import feedparser
import re


def read(feed, classifier):
    f = feedparser.parse(feed)
    for entry in f['entries']:
        print
        print '----'
        print 'Title:   %s' % (entry['title'].encode('utf-8'))
        print 'Publisher: %s' % (entry['publisher'].encode('utf-8'))
        print
        print entry['summary'].encode('utf-8')

        fulltext = '%s\n%s\n%s' % (entry['title'], entry['publisher'],entry['summary'])
        print 'Guess: %s' % str(classifier.classify(fulltext))

        cl = raw_input('Enter Category: ')
        classifier.train(fulltext, cl)
