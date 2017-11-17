import matplotlib.pyplot as plt
import numpy as np

lrtime = []
lrgradNorm = []
lracc =[]
lrpre =[]
lrrec =[]
plrtime = []
plrgradNorm = []
plracc =[]
plrpre =[]
plrrec =[]
plrtime5 = []
plrtime10 = []
plracc5 = []
plracc10 = []

# with open('lr5a','r') as file_object:
#     for line in file_object.readlines():
# 		lrtime.append(eval(line.split("\t")[0]))
# 		lrgradNorm.append(eval(line.split("\t")[1]))
# 		lracc.append(eval(line.split("\t")[2]))
# 		lrpre.append(eval(line.split("\t")[3]))
# 		lrrec.append(eval(line.split("\t")[4]))
# file_object.close()

with open('plr5a','r') as file_object:
    for line in file_object.readlines():
		plrtime.append(eval(line.split("\t")[0]))
		plrgradNorm.append(eval(line.split("\t")[1]))
		plracc.append(eval(line.split("\t")[2]))
		plrpre.append(eval(line.split("\t")[3]))
		plrrec.append(eval(line.split("\t")[4]))
file_object.close()

with open('plr5b5','r') as file_object:
    for line in file_object.readlines():
		plrtime5.append(eval(line.split("\t")[0]))
		plracc5.append(eval(line.split("\t")[2]))
file_object.close()

with open('plr5b10','r') as file_object:
    for line in file_object.readlines():
		plrtime10.append(eval(line.split("\t")[0]))
		plracc10.append(eval(line.split("\t")[2]))
file_object.close()

# plt.xlim(0, 600)
# plt.ylim(0, 400)
"""
(a)
"""
# plt.figure(1)
# plt.xlabel('$time$', fontsize=12)
# plt.ylabel('$gradient$', fontsize=12)
# plt.plot(lrtime , lrgradNorm, 'b-', linewidth=2, label='$LogiticRegression$')
# plt.plot(plrtime , plrgradNorm, 'r-', linewidth=2, label='$ParallelLogisticRegression$')
# plt.legend(edgecolor='black')
# plt.show()


# plt.figure(2)
# plt.xlabel('$time$', fontsize=12)
# plt.ylabel('$accuracy$', fontsize=12)
# plt.plot(lrtime , lracc, 'b-', linewidth=2, label='$LogiticRegression$')
# plt.plot(plrtime , plracc, 'r-', linewidth=2, label='$ParallelLogisticRegression$')
# plt.legend(edgecolor='black')
# plt.show()

# plt.figure(3)
# plt.xlabel('$time$', fontsize=12)
# plt.ylabel('$precision$', fontsize=12)
# plt.plot(lrtime , lrpre, 'b-', linewidth=2, label='$LogiticRegression$')
# plt.plot(plrtime , plrpre, 'r-', linewidth=2, label='$ParallelLogisticRegression$')
# plt.legend(edgecolor='black')
# plt.show()

# plt.figure(4)
# plt.xlabel('$time$', fontsize=12)
# plt.ylabel('$rec$', fontsize=12)
# plt.plot(lrtime , lrrec, 'b-', linewidth=2, label='$LogiticRegression$')
# plt.plot(plrtime , plrrec, 'r-', linewidth=2, label='$ParallelLogisticRegression$')
# plt.legend(edgecolor='black')
# plt.show()


"""
(b)
"""
#lamda = 0
plt.figure(1)
plt.xlabel('$time$', fontsize=12)
plt.ylabel('$accuracy$', fontsize=12)
plt.plot(plrtime , plracc, 'b-', linewidth=2, label='$\\lambda = 0$')
plt.plot(plrtime5 , plracc5, 'r-*', linewidth=2, label='$\\lambda = 5$')
plt.plot(plrtime10 , plracc10, 'c--', linewidth=2, label='$\\lambda = 10$')
plt.legend(edgecolor='black')
plt.show()