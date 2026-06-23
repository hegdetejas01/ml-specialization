# Single variable linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotGraph(x,y,title,xlabel,ylabel):

    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def hypothesis(x,w,b):
    """
    This function returns the hypothesis of the model
    for linear regression: y = wx + b
    """
    h = w*x + b
    return h


def computeCost(x,y,w,b):
    m = x.shape[0]

    predictions = hypothesis(x, w, b)
    cost = 1/(2*m)*(np.sum((predictions-y)**2))
    return cost


def calculateGradient(x,y,w,b):
    m = x.shape[0]
    error = hypothesis(x,w,b) - y
    dj_dw = (np.sum(error*x))/m
    dj_db = (np.sum(error))/m
    return dj_dw, dj_db


def gradientDecent(x,y,w,b,alpha,iterations):
    j_list = []
    i_list = []
    w_list = []

    for i in range(iterations):
        dj_dw, dj_db = calculateGradient(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        newCost = computeCost(x,y,w,b)
            
        w_list.append(w)
        j_list.append(newCost)
        i_list.append(i)

    plotGraph(i_list, j_list, "Cost Versus Iterations", "Number of iterations", "Cost")
    plotGraph(w_list, j_list, "Cost Versus W", "W", "Cost")

    return w,b,j_list,w_list


def readData():
    file = "Datasets/Salary_Data.csv"
    data = pd.read_csv(file)
    x = data['YearsExperience'].values
    y = data['Salary'].values
    return x, y 


def get_stats(data):
    return np.mean(data), np.std(data)

def normalize(data, mean, std):
    return (data - mean) / std


def plotFinalGraphWithPoints(x,y,w,b,x_point, y_point):
    p = w*x + b
    plt.plot(x, p, c="r")
    plt.scatter(x_point, y_point, marker="*", c="b")
    plt.scatter(x, y, marker="x", c="r")
    plt.show()


init_w = 0
init_b = 0
alpha = 0.0001
iterations = 100000

x,y = readData()
x_mean, x_std = get_stats(x)
y_mean, y_std = get_stats(y)

x_train = normalize(x, x_mean, x_std)
y_train = normalize(y, y_mean, y_std)
final_w, final_b, j_list, w_list = gradientDecent(x_train, y_train, init_w, init_b, alpha, iterations)


try:
    exp = float(input("Enter the years of experience: "))
except:
    print("Try Again. Invalid Data Type.")
else:
    normX = normalize(exp, x_mean, x_std)
    salaryPredictionNorm = final_w * normX + final_b
    salaryPredictionFinal = (salaryPredictionNorm * y_std) + y_mean
    print("The salary will be : ", int(salaryPredictionFinal))
    plotFinalGraphWithPoints(x_train, y_train, final_w, final_b, normX, salaryPredictionNorm)