# Single variable linear regression
# This has 2 models. One salary model, the second student model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statements as st

def plotGraph(x,y,title,xlabel,ylabel):
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def hypothesis(x,w,b):
    st.hypothesis_description
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

    return w,b,j_list,w_list,i_list


def readData():
    userOption = input(st.user_question_one)
    
    if int(userOption) == 1:
        data = pd.read_csv(st.salary_file)
        x = data[st.salary_x].values
        y = data[st.salary_y].values

    elif int(userOption) == 2:
        data = pd.read_csv(st.student_file)
        x = data[st.student_x].values
        y = data[st.student_y].values

    else:
        print(st.thank_you_user)
        exit

    return x, y, userOption


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

x,y,userOption = readData()

x_mean, x_std = get_stats(x)
y_mean, y_std = get_stats(y)
x_train = normalize(x, x_mean, x_std)
y_train = normalize(y, y_mean, y_std)

final_w, final_b, j_list, w_list, i_list = gradientDecent(x_train, y_train, init_w, init_b, alpha, iterations)


try:
    if int(userOption) == 1:
        exp = float(input(st.user_years_exp))
    else:
        exp = float(input(st.user_hours_study))

except:
    print(st.invalid_data_type)

else:
    normX = normalize(exp, x_mean, x_std)
    normY = final_w * normX + final_b
    finalY = (normY * y_std) + y_mean

    if int(userOption) == 1:
        print(st.final_salary_msg, int(finalY))
    else:
        print(st.final_score_msg, int(finalY))

    plotFinalGraphWithPoints(x_train, y_train, final_w, final_b, normX, normY)

    userOption = input(st.user_question_two)
    if userOption.upper() == 'Y':
        plotGraph(i_list, j_list, st.c_v_i, st.iter, st.cost)
        plotGraph(w_list, j_list, st.c_v_w, "W", st.cost)
    else:
        print(st.thank_you_user)