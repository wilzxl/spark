import LogisticRegression as LR
train = LR.readData("newsgroups/news.train")
test = LR.readData("newsgroups/news.test")
beta0 = LR.SparseVector({})
a,b,c = LR.train(train,beta0,0,20,0.001,test)


