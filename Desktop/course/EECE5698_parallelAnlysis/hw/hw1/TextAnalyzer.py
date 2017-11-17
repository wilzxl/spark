import sys
import argparse
import numpy as np
from pyspark import SparkContext
import math

def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')


    if args.mode=='TF':
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed
    	myrdd = sc.textFile(args.input)
    	myrdd.flatMap(lambda s : s.split())\
    	.map(lambda word: (toLowerCase(stripNonAlpha(word)), 1))\
    	.reduceByKey(lambda x, y : x + y)\
    	.filter(lambda (x, y) : x != "").saveAsTextFile(args.output)
        '''
        map(toLowerCase).map(stripNonAlpha)
        '''








    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        myrdd = sc.textFile(args.input)
    	top_20 = myrdd.map(lambda s : eval(s))\
    	.takeOrdered(20, key = lambda (x, y) : -y)
    	sc.parallelize(top_20).saveAsTextFile(args.output)





       
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        myrdd = sc.wholeTextFiles(args.input).cache()
    	doc_num = myrdd.count()
    	myrdd.flatMapValues(lambda s : s.split())\
    	.map(lambda (x, y) : (x, toLowerCase(stripNonAlpha(y))))\
    	.distinct().map(lambda (x, y) : (y, 1))\
    	.reduceByKey(lambda x, y : x + y)\
    	.filter(lambda (x, y) : x != "")\
    	.map(lambda (x, y) : (x, math.log(1.0 * doc_num / y)))\
    	.saveAsTextFile(args.output)






    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        TF = sc.textFile(args.input).map(lambda s : eval(s))
    	IDF = sc.textFile(args.idfvalues).map(lambda s: eval(s))
    	TFIDF = TF.join(IDF)
    	score = TFIDF.map(lambda (x, y) : (x, y[0] * y[1]))
    	score.saveAsTextFile(args.output)
    	#score_20 = score.takeOrdered(20, key = lambda (x, y) : -y)
    	#sc.parallelize(score_20).saveAsTextFile(args.output)





        
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        F_1 = sc.textFile(args.input).map(lambda s : eval(s))
    	F_2 = sc.textFile(args.other).map(lambda s : eval(s))
    	intersect = F_1.join(F_2)
    	sum_1 = intersect.map(lambda (x, y) : y[0] * y[1])\
    	.reduce(lambda x, y : x + y)
    	sum_2 = F_1.map(lambda (x, y) : y ** 2)\
    	.reduce(lambda x, y : x + y)
    	sum_3 = F_2.map(lambda (x, y) : y ** 2)\
    	.reduce(lambda x, y : x + y)
    	# Make it as list, or error
    	cosine = [(1.0 * sum_1) / math.sqrt(sum_2 * sum_3)]
    	sc.parallelize(cosine, 1).saveAsTextFile(args.output)
        




