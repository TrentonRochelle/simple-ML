import numpy as np
import matplotlib.pyplot as plt
import math
import os

def main():
    currentFile = __file__
    realPath = os.path.realpath(currentFile)
    dirPath = os.path.dirname(realPath)
    filepath = dirPath + "/salammbo_libsvm.txt"
    with open(filepath) as f:
        content = f.readlines()
        data = np.zeros([len(content),1],dtype='float')
        f.close()
        for i in range(len(content)):
            x = content[i].split()
            data[i][0]= x[0]
            for index,a in enumerate(x):
                if(index>0):
                    temp = a.split(":")
                    shape = np.shape(data)
                    if(int(shape[1])<int(temp[0])+1):
                        empty = np.zeros((int(shape[0]),int(temp[0])-int(shape[1])+1))
                        data = np.concatenate((data,empty), axis=1)
                    data[i][int(temp[0])] = temp[1]
        for index,column in enumerate(data.T):
            if index!=0:   
                largest = max(column)
                column/=largest
    
    tempp = np.vsplit(data,2)
    en_data = tempp[0]
    fr_data = tempp[1]
    m_en = len(en_data[:,0])
    en_data_x_norm = np.c_[np.ones(m_en),en_data[:,1]] 
    en_data_y_norm = np.c_[en_data[:,2]] 
    fr_data_x_norm = np.c_[np.ones(m_en),fr_data[:,1]] 
    fr_data_y_norm = np.c_[fr_data[:,2]] 
    
    
    
    
    #Batch Gradient Descent on each data normalized    
    graph_normed = plt.scatter(en_data_x_norm[:,1],en_data_y_norm,color='r')
    graph_normed = plt.scatter(fr_data_x_norm[:,1],fr_data_y_norm,color='b')  
    theta_en = np.array([[0],[0]])
    theta_en,J_hist_en_norm = BatchGradientDescent(en_data_x_norm, en_data_y_norm,theta_en,.2,600)
    line_en = theta_en[1]*en_data_x_norm[:,1] + theta_en[0]
    graph = plt.plot(en_data_x_norm[:,1],line_en,color='r')
    theta_fr = np.array([[0],[0]])
    theta_fr,J_hist_fr_norm = BatchGradientDescent(fr_data_x_norm, fr_data_y_norm,theta_fr,.2,600)
    line_fr = theta_fr[1]*fr_data_x_norm[:,1] + theta_fr[0]
    graph = plt.plot(fr_data_x_norm[:,1],line_fr,color='b')
    print("Batch Normalized")
    plt.show(graph)
    graph2=plt.plot(J_hist_en_norm,color='r')
    graph2=plt.plot(J_hist_fr_norm,color='b')
    print("Batch Cost per iteration")
    plt.show(graph2)
    
    #Stochastic Gradient Descent on each data normalized    
    for i in range(5):
        en_data_x_norm = np.vstack((en_data_x_norm,en_data_x_norm))
        en_data_y_norm = np.vstack((en_data_y_norm,en_data_y_norm))
        fr_data_x_norm = np.vstack((fr_data_x_norm,fr_data_x_norm))
        fr_data_y_norm = np.vstack((fr_data_y_norm,fr_data_y_norm))
    theta_en_norm = np.array([[0],[0]])
    theta_en_norm,J_hist_en_norm = SGD(en_data_x_norm, en_data_y_norm,theta_en_norm,.2)
    line_en_norm = theta_en_norm[1]*en_data_x_norm[:,1] + theta_en_norm[0]
    theta_fr_norm = np.array([[0],[0]])
    theta_fr_norm,J_hist_fr_norm = SGD(fr_data_x_norm, fr_data_y_norm,theta_fr_norm,.2)
    line_fr_norm = theta_fr_norm[1]*fr_data_x_norm[:,1] + theta_fr_norm[0]
    graph4 = plt.scatter(en_data_x_norm[:,1],en_data_y_norm,color='r')
    graph4 = plt.plot(en_data_x_norm[:,1],line_en_norm,color='r')
    graph4 = plt.scatter(fr_data_x_norm[:,1],fr_data_y_norm,color='b')
    graph4 = plt.plot(fr_data_x_norm[:,1],line_fr_norm,color='b')
    print("Stochastic Normalized")
    plt.show(graph4)
    graph5=plt.plot(J_hist_en_norm[:,1],J_hist_en_norm[:,0],color='r')
    graph5=plt.plot(J_hist_fr_norm[:,1],J_hist_fr_norm[:,0],color='b')
    print("Stochastic Cost per iteration")
    plt.show(graph5)
    
    for i in range(len(data)):
        if data[i][0] == 1:
            graph6 = plt.scatter(data[i][1],data[i][2],color='b')
        else:
            graph6 = plt.scatter(data[i][1],data[i][2],color='r')
    
    
    
    theta = np.array([[1.],[1.],[1.]])
    theta = SGDPerceptron(data[:,[1,2]], data[:,0],theta,.000244)
    print(theta)
    line = (-1.0/theta[2])*(data[:,2]*theta[1]+theta[0])
    graph6 = plt.plot(data[:,2],line,color='y')
    print("Threshold Stochastic Perceptron")
    plt.show(graph6)
    
    
    for i in range(len(data)):
        if data[i][0] == 1:
            graph6 = plt.scatter(data[i][1],data[i][2],color='b')
            graph7 = plt.scatter(data[i][1],data[i][2],color='b')
        else:
            graph6 = plt.scatter(data[i][1],data[i][2],color='r')
            graph7 = plt.scatter(data[i][1],data[i][2],color='r')
    
    theta = np.array([[1.],[1.],[1.]])
    theta = LogisticRegression(data[:,[1,2]], data[:,0],theta,10)
    print(theta)
    line = (-1.0/theta[2])*(data[:,2]*theta[1]+theta[0])
    graph7 = plt.plot(data[:,2],line,color='y')
    print("Logistic Stochastic Perceptron")
    plt.show(graph7)

def Cost(x,y,weights):
    m = float(len(y))
    predictions = x.dot(weights)
    sqrErrors = np.square(predictions - y)
    return(1/(2*m) * np.sum(sqrErrors))
    
def BatchGradientDescent(x,y,weights,alpha,num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        #print(Cost(x,y,weights))
        y_hat = np.dot(x, weights)
        delta = np.dot(x.T, y_hat-y)/m
        #print(Cost(x,y,weights))
        weights = weights - alpha * delta
        J_history[i] = Cost(x,y,weights)
    return(weights,J_history)
    #return(weights)
   
def SGD(x,y,weights,alpha):
    m=len(y)
    J_history = np.zeros((m,2),dtype=float)
    for i in range(m):
        y_hat = np.dot(x[i],weights)
        delta = np.dot(np.array([[x[i][0]],[x[i][1]]]), np.array([y_hat-y[i]]))
        weights = weights - alpha *delta
        J_history[i][0] = Cost(x,y,weights)
        J_history[i][1] = i+1
    return(weights,J_history)

def LogisticRegression(x,y,weights,alpha):
    while(Loss(x,y,weights)>2):
        #print(Loss(x,y,weights))
        for i in range(len(y)):
            y_hat = sigmoid(x[i],weights)
            weights[0] += alpha*(y[i]-y_hat)*y_hat*(1-y_hat)
            weights[1] += alpha*(y[i]-y_hat)*y_hat*(1-y_hat)*x[i][0]
            weights[2] += alpha*(y[i]-y_hat)*y_hat*(1-y_hat)*x[i][1]
    return(weights)


def Threshold(x,weights):
    activation = weights[0] +weights[1]*x[0] + weights[2]*x[1]
    if activation >= 0.0:
        return 1.0 
    else:
        return 0.0


def sigmoid(x,weights):
    z = weights[0]+weights[1]*x[0] + weights[2]*x[1]
    return(1/(1+ math.exp((-z))))


def Loss(x,y,weights):
    total = 0
    for i in range(len(x)):
        total+=(y[i]-sigmoid(x[i],weights))**2
    return(total)
    
    
def SGDPerceptron(x,y,weights,alpha):
    wrong = 30
    a = 0
    while wrong >0:
        a+=1
        wrong = 0
        for i in range(len(y)):
            prediction = Threshold(x[i],weights)
            if(prediction!=y[i]):
                wrong+=1
            weights[0] += alpha * (y[i]-prediction)
            weights[1] += alpha * (y[i]-prediction)*x[i][0]
            weights[2] += alpha * (y[i]-prediction)*x[i][1]
    return(weights)


main()