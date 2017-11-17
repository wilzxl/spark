import sys
import argparse
import numpy as np
import math
from pyspark import SparkContext

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
        rdd = sc.textFile(args.input)
        rdd.flatMap(lambda line: line.split())\
           .map(lambda word : (toLowerCase(stripNonAlpha(word)), 1))\
           .reduceByKey(lambda x,y: x+y)\
           .filter(lambda (x,y): x != "")\
           .saveAsTextFile(args.output)
	
    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        rdd = sc.textFile(args.input)
        _20 = rdd.map(lambda line: eval(line))\
                 .takeOrdered(20, key=lambda (x,y): -y)
        sc.parallelize(_20, 1)\
          .saveAsTextFile(args.output)





       
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        rdd = sc.wholeTextFiles(args.input)
        count = rdd.count()
        wordnumber = rdd.flatMapValues(lambda x : x.split())\
                        .map(lambda (x,y): (x,toLowerCase(stripNonAlpha(y))))\
                        .distinct()\
                        .map(lambda (x,y): (y,1))\
                        .reduceByKey(lambda x,y : x+y)\
                        .filter(lambda (x,y): x!="")\
                        .map(lambda (x,y) : (x,math.log(1.0*count/y)))\
                        .saveAsTextFile(args.output)
       





    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        TF = sc.textFile(args.input)\
               .map(lambda line: eval(line))
        IDF = sc.textFile(args.idfvalues)\
                .map(lambda line: eval(line))
        TFIDF = TF.join(IDF)
        Score = TFIDF.map(lambda (x,y): (x,1.0 * y[0] * y[1]))\
                     .saveAsTextFile(args.output)
                   #.takeOrdered(20, key=lambda (x,y): -y)
        #sc.parallelize(_20, 1)\
          #.saveAsTextFile(args.output)





        
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        TFIDF1 = sc.textFile(args.input)\
                   .map(lambda line: eval(line))
        SUM1 = TFIDF1.map(lambda x : x[1]**2)\
                     .reduce(lambda x,y: x + y)
        TFIDF2 = sc.textFile(args.other)\
                   .map(lambda line :eval(line))
        SUM2 = TFIDF2.map(lambda x : x[1]**2)\
                     .reduce(lambda x,y: x + y)
        COMB = TFIDF1.join(TFIDF2)
        SUM3 = COMB.map(lambda (x,y):1.0 * y[0] * y[1])\
                   .reduce(lambda x,y: x + y )
        COS = [(1.0 * SUM3)/math.sqrt(1.0 * SUM1 * SUM2)]
        sc.parallelize(COS, 1)\
          .saveAsTextFile(args.output)
        

          


