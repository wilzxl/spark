import numpy as np
import argparse
from time import time
import math
from SparseVector import SparseVector


def readBeta(input):
    beta = SparseVector({})
    with open(input,'r') as fh:
        for  line in fh:
            (feat,val) = eval(line.strip())
            beta[feat] = val
    return beta




def writeBeta(output,beta):
    with open(output,'w') as fh:
        for key in beta:
            fh.write('(%s,%f)\n' % (key,beta[key]))






def readData(input_file):
    listSoFar = []
    with open(input_file,'r') as fh:
        for line in fh:
                (x,y) = eval(line)
                x = SparseVector(x)
                listSoFar.append((x,y))

    return listSoFar




def getAllFeatures(data):
    features = SparseVector({})
    for (x,y) in data:
        features = features + x
    return features.keys() 



def logisticLoss(beta,x,y):
    l=math.log(1.0+math.exp(-y*beta.dot(x)))
    return l
    pass



def gradLogisticLoss(beta,x,y):
    deltaL=x*(-y/(1.0+math.exp(y*(beta.dot(x)))))
    return deltaL
    pass
  


def totalLoss(data,beta,lam = 0.0):
    loss = 0.0 
    for (x,y) in data:
        loss += logisticLoss(beta,x,y)
    return loss + lam * beta.dot(beta) 



def gradTotalLoss(data,beta, lam = 0.0):
    TotalG=beta*(2*lam)
    for (x,y) in data:
        TotalG+=gradLogisticLoss(beta,x,y)
    return TotalG
    pass	


def lineSearch(fun,x,grad,fx,gradNormSq, a=0.2,b=0.6):
    t = 1.0
    while fun(x-t*grad) > fx- a * t * gradNormSq:
        t = b * t
    return t 
 
    
def test(data,beta):
    P=[]
    N=[]
    TP=[]
    FP=[]
    TN=[]
    FN=[]
    for (x,y) in data:
        if x.dot(beta)>0:
            P.append((x,y))
            if y==+1:
                TP.append((x,y))
            else:
                FP.append((x,y))
        else:
            N.append((x,y))
            if y==-1:
                TN.append((x,y))
            else:
                FN.append((x,y))
    ACC=1.*(len(TN)+len(TP))/(len(N)+len(P))
    PRE=1.*(len(TP))/(len(TP)+len(FP))
    REC=1.*(len(TP))/(len(TP)+len(FN))
    return ACC,PRE,REC
    pass


def train(data,beta_0, lam,max_iter,eps,test_data=None):
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        obj = totalLoss(data,beta,lam)   

        grad = gradTotalLoss(data,beta,lam)  
	gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        fun = lambda x: totalLoss(data,x,lam)
        gamma = lineSearch(fun,beta,grad,obj,gradNormSq)
        
        beta = beta - gamma * grad
        if test_data == None:
            print 'k = ',k,'\tt = ',time()-start,'\tL(ß_k) = ',obj,'\t||?L(ß_k)||_2 = ',gradNorm,'\tgamma = ',gamma
        else:
            acc,pre,rec = test(test_data,beta)
            print 'k = ',k,'\tt = ',time()-start,'\tL(ß_k) = ',obj,'\t||?L(ß_k)||_2 = ',gradNorm,'\tgamma = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec
        k = k + 1

    return beta,gradNorm,k         


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter ?')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='e-tolerance. If the l2_norm gradient is smaller than e, gradient descent terminates.') 

    
    args = parser.parse_args()
    

    print 'Reading training data from',args.traindata
    traindata = readData(args.traindata)
    print 'Read',len(traindata),'data points with',len(getAllFeatures(traindata)),'features in total'
    
    if args.testdata is not None:
        print 'Reading test data from',args.testdata
        testdata = readData(args.testdata)
        print 'Read',len(testdata),'data points with',len(getAllFeatures(testdata)),'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    print 'Training on data from',args.traindata,'with ? =',args.lam,', e =',args.eps,', max iter = ',args.max_iter
    beta, gradNorm, k = train(traindata,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps,test_data=testdata) 
    print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
    print 'Saving trained ß in',args.beta
    writeBeta(args.beta,beta)