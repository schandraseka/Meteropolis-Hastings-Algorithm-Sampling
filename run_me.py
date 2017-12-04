# Import python modules
import numpy as np
import kaggle
import random
import matplotlib.pyplot as plt

def onea(J, alpha):
    if J[0] != 0:
        return 1e-100
    res = 1.0
    for i,val in enumerate(J):
        if i!=0 and J[i] == J[i-1]:
            res*=alpha
        elif i!=0 and J[i] != J[i-1]:
            res*=(1-alpha)
    if res != 0.0:
        return res
    else:
        return 1e-100

inputs = [[np.array([0,1,1,0,1]), 0.75], [np.array([0,0,1,0,1]), 0.2], [np.array([1,1,0,1,0,1]), 0.2], [np.array([0,1,0,1,0,0]), 0.2]]

for i,inp in enumerate(inputs):
    print("Result for 1a for input" + str(i + 1) + " " + str(inp[0])  + " "+ str(inp[1])  +": " + str(onea(inp[0], inp[1])))
print("--------------------------------------------------------------------------------------------------------")


def pbj(b,j):
    if b == 0 and j == 0:
        return 0.20
    elif b == 0 and j == 1:
        return 0.90
    elif b ==1 and j == 0:
        return 0.80
    elif b == 1 and j == 1:
        return 0.10
    else:
        return 0

def oneb(J, B):
    res = 1.0
    for i, val in enumerate(J):
        res *= pbj(B[i], J[i])
    return res

inputs = [[np.array([0,1,1,0,1]), np.array([1,0,0,1,1])], [np.array([0,1,0,0,1]), np.array([0,0,1,0,1])], [np.array([0,1,1,0,0,1]), np.array([1,0,1,1,1,0])], [np.array([1,1,0,0,1,1]), np.array([0,1,1,0,1,1])]]

for i,inp in enumerate(inputs):
    print("Result for 1b for input" + str(i + 1) + " " + str(inp[0])  + " "+ str(inp[1])  +": " + str(oneb(inp[0], inp[1])))

print("--------------------------------------------------------------------------------------------------------")

def onec(alpha):
    if alpha >=0 and alpha <=1:
        return 1
    else:
        return 0

def oned(J, B, alpha):
    return onec(alpha)*oneb(J,B)*onea(J,alpha)

inputs = [[np.array([0,1,1,0,1]), np.array([1,0,0,1,1]), 0.75], [np.array([0,1,0,0,1]), np.array([0,0,1,0,1]), 0.3], [np.array([0,0,0,0,0,1]), np.array([0,1,1,1,0,1]), 0.63], [np.array([0,0,1,0,0,1,1]), np.array([1,1,0,0,1,1,1]), 0.23]]

for i,inp in enumerate(inputs):
    print("Result for 1d for input" + str(i + 1) + " " + str(inp[0])  + " "+ str(inp[1])  + " "+ str(inp[2]) +": " + str(oned(inp[0], inp[1], inp[2])))



print("--------------------------------------------------------------------------------------------------------")

def onee(J):
    ind = random.randint(0,len(J)-1)
    J_new = np.copy(J)
    J_new[ind] ^= 1
    return J_new

inputs = [[np.array([0,1,1,0,1])]]
for i,inp in enumerate(inputs):
    print("Result for 1e for input" + str(i + 1) + " " + str(inp[0]) +": " + str(onee(inp[0])))

print("--------------------------------------------------------------------------------------------------------")

def oneg(alpha):
    #return np.random.uniform()
    return np.random.rand()


def onef(J, B, alpha, iterations):
    J_mean = np.array([0 for step in range(len(B))])
    lst = []
    for i in range(iterations):
        J_new = onee(J)
        num = oned(J_new, B, alpha)
        den = oned(J, B, alpha)
        lst.append(num)
        acceptance_ratio = num*1.0/den*1.0
        if np.random.rand() <= acceptance_ratio:
            J = J_new
        J_mean += np.array(J)
    return J_mean/(1.0*iterations)


inputs = [[np.array([0,0,0,0,0]),np.array([1,0,0,1,1]), 0.5, 10000], [np.array([0,0,0,0,0,0,0,0]),np.array([1,0,0,0,1,0,1,1]), 0.11, 10000], [np.array([0,0,0,0,0,0,0]),np.array([1,0,0,1,1,0,0]), 0.75, 10000]]
for i,inp in enumerate(inputs):
    print("Result for 1f for input" + str(i + 1) + " " + str(inp[0])  + " "+ str(inp[1])  + " "+ str(inp[2])+ " "+ str(inp[3]) +": " + str(onef(inp[0], inp[1], inp[2], inp[3])))

print("--------------------------------------------------------------------------------------------------------")


def oneh(J, B, iterations, plot = False):
    alpha_mean = 0
    alpha = 1e-1000
    lst = []
    for i in range(iterations):
        alpha_new = oneg(alpha)
        acceptance_ratio = oned(J,B, alpha_new)/oned(J, B, alpha)
        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
        alpha_mean += alpha
        lst.append(alpha)
    if plot:
        plt.hist(lst)
        plt.title("Histogram of frequency distribution of alpha over iterations")
        plt.xlabel("Alphas")
        plt.ylabel("Frequency")
        plt.show()
    return alpha_mean/iterations

inputs = [[np.array([0,1,0,1,0]), np.array([1,0,1,0,1]), 10000], [np.array([0,0,0,0,0]), np.array([1,1,1,1,1]), 10000], [np.array([0,1,1,0,1]), np.array([1,0,0,1,1]), 10000], [np.array([0,1,1,1,1,1,1,0]), np.array([1,0,0,1,1,0,0,1]), 10000], [np.array([0,1,1,0,1,0]), np.array([1,0,0,1,1,1]), 10000]]

for i,inp in enumerate(inputs):
    print("Result for 1h for input" + str(i + 1) + " " + str(inp[0])  + " "+ str(inp[1])  + " "+ str(inp[2]) +": " + str(oneh(inp[0], inp[1], inp[2],False)))




print("--------------------------------------------------------------------------------------------------------")


def onei(J,alpha):
    return (onee(J),oneg(alpha))


def onej(B, iterations, plot = False):
    alpha_mean = 0
    J_mean = np.zeros(len(B))
    J = np.array([0 for s in range(len(B))])
    alpha = 1e-1000
    lst = []
    for i in range(iterations):
        J_new,alpha_new = onei(J,alpha)
        acceptance_ratio = oned(J_new,B, alpha_new)/oned(J, B, alpha)
        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new
        alpha_mean += alpha
        J_mean += J
        lst.append(alpha)
    if plot:
        plt.hist(lst)
        plt.title("Histogram of frequency distribution of alpha over iterations")
        plt.xlabel("Alphas")
        plt.ylabel("Frequency")
        plt.show()
        plt.plot([i+1 for i in range(10000)],lst)
        plt.title("Alpha as function of iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Alphas")
        plt.show()
    return J_mean/iterations,alpha_mean/iterations

inputs = [[np.array([1,1,0,1,1,0,0,0]), 10000]]
for i,inp in enumerate(inputs):
    print("Result for 1j for input" + str(i + 1) + " " + str(inp[0]) + " " + str(inp[1])+ ":" + str(onej(inp[0], inp[1], True)))

print("--------------------------------------------------------------------------------------------------------")


def onek(Jn, alpha):
    if Jn == 0:
        return  (1 - alpha)*0.1 + alpha*0.8
    if Jn == 1:
        return  (1 - alpha)*0.8 + alpha*0.1

inputs = [[1,0.6], [0,0.99], [0,0.33456], [1,0.5019]]
for i,inp in enumerate(inputs):
    print("Result for 1k for input" + str(i + 1) + " " + str(inp[0]) + " " + str(inp[1])+ ":" + str(onek(inp[0], inp[1])))


print("--------------------------------------------------------------------------------------------------------")

def onel(B, iterations):
    alpha_mean = 0
    J_mean = np.zeros(len(B))
    J = np.array([0 for s in range(len(B))])
    alpha = 1e-1000
    black_preds = 0
    lst = []
    for i in range(iterations):
        J_new,alpha_new = onei(J,alpha)
        acceptance_ratio = oned(J_new,B, alpha_new)/oned(J, B, alpha)
        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new
        black_preds += onek(J[-1], alpha)
        lst.append(alpha)
    return black_preds/iterations

inputs = [[[0, 0, 1], 10000], [[0, 1, 0, 1, 0, 1], 10000], [[0, 1, 0, 0, 0, 0, 0], 10000], [[1, 1, 1, 1, 1], 10000]]
for i,inp in enumerate(inputs):
    print("Result for 1l for input" + str(i + 1) + " " + str(inp[0]) + " " + str(inp[1])+ ":" + str(onel(inp[0], inp[1])))

print("--------------------------------------------------------------------------------------------------------")


lengths = [10, 15, 20, 25]
prediction_prob = list()
for l in lengths:
    B_array = np.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
    for B in B_array:
        prediction_prob.append(onel(B, 100000))
        print('Prob of next entry in ', B , 'is black is', prediction_prob[-1])
        
print("---------------------------------------------------------------------------------------------------------")
# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(np.array(prediction_prob), file_name)
