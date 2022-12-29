# python 3.7 plz
import crayons
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils 

# keras py -3.7
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.image import load_img,  img_to_array, array_to_img
from keras import regularizers
from keras.layers import Dropout,Dense,Activation,BatchNormalization

# iris dataset
from sklearn.datasets import load_iris

# PCA - Kmeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def myModel(input,n_class,deep):
    model = Sequential()
    model.add(Dense(HIDDEN, input_shape=input))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    for i in range(deep):
        model.add(Dense(HIDDEN))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    model.summary()
    return model

def history_anarysis(name):
    _res_log["loss_{}".format(name)] = history.history["loss"]
    _res_log["val_loss_{}".format(name)] = history.history["val_loss"]
    _res_log["accuracy_{}".format(name)] = history.history["accuracy"]
    _res_log["val_accuracy_{}".format(name)] = history.history["val_accuracy"]

def mk_history():
    loss_dataframe = _res_log.filter(like='loss', axis=1)
    epochs = range(1,len(_res_log) + 1)
    # loss(MSE) learning curb
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = 0
    for name in mode_list:
        lossQuery = loss_dataframe.filter(like='{}'.format(name), axis=1)
        plt.plot(epochs , lossQuery["loss_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,label = "T_{}".format(name))
        plt.plot(epochs , lossQuery["val_loss_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,alpha=0.5, label = "V_{}".format(name))
        m += 1
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    #plt.yscale("log")
    plt.legend()
    fig1.savefig("loss.png")
    plt.clf()
    plt.close()

    loss_dataframe = _res_log.filter(like='loss', axis=1)
    epochs = range(1,len(_res_log) + 1)
    # loss(MSE) learning curb
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = 0
    for name in mode_list:
        lossQuery = loss_dataframe.filter(like='{}'.format(name), axis=1)
        plt.plot(epochs , lossQuery["loss_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,label = "T_{}".format(name))
        plt.plot(epochs , lossQuery["val_loss_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,alpha=0.5, label = "V_{}".format(name))
        m += 1
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0,0.5)
    #plt.yscale("log")
    plt.legend()
    fig1.savefig("loss_.png")
    plt.clf()
    plt.close()

    accuracy_dataframe = _res_log.filter(like='accuracy', axis=1)
    epochs = range(1,len(_res_log) + 1)
    # accuracy(MSE) learning curb
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = 0
    for name in mode_list:
        accuracyQuery = accuracy_dataframe.filter(like='{}'.format(name), axis=1)
        plt.plot(epochs , accuracyQuery["accuracy_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5, label = "T_{}".format(name))
        plt.plot(epochs , accuracyQuery["val_accuracy_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,alpha=0.5, label = "V_{}".format(name))
        m += 1
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    #plt.yscale("log")
    plt.legend()
    fig1.savefig("accuracy.png")
    plt.clf()
    plt.close()

    accuracy_dataframe = _res_log.filter(like='accuracy', axis=1)
    epochs = range(1,len(_res_log) + 1)
    # accuracy(MSE) learning curb
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = 0
    for name in mode_list:
        accuracyQuery = accuracy_dataframe.filter(like='{}'.format(name), axis=1)
        plt.plot(epochs , accuracyQuery["accuracy_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5, label = "T_{}".format(name))
        plt.plot(epochs , accuracyQuery["val_accuracy_{}".format(name)] , color = color_list[m],marker = marker_list[m],ms = 1,lw = 0.5,alpha=0.5, label = "V_{}".format(name))
        m += 1
    plt.xlabel("Epochs")
    plt.ylabel("accuracy_")
    plt.ylim(0.5,1)
    #plt.yscale("log")
    plt.legend()
    fig1.savefig("accuracy_.png")
    plt.clf()
    plt.close()


if __name__ == '__main__':
    print(crayons.red("Main Program"))
    mode_list = ["NN1","NN2","NN3"]
    #mode_list = ["NN1"]
    nn_deep = [1,2,3]
    marker_list = ["4", "8", "s", "p", "*"]
    color_list = ["r", "g", "b", "c", "m"]
    BATCH = 3
    EPOCHS = 100
    HIDDEN = 10
    VAL = 1/5
    opt = keras.optimizers.Adam(learning_rate=0.001)
    iris = load_iris()
    x_train = iris.data
    y_train = iris.target
    x_train = x_train.astype('float32')

    # Normalization
    x_train[:,0] = x_train[:,0]/max(x_train[:,0])
    x_train[:,1] = x_train[:,1]/max(x_train[:,1])
    x_train[:,2] = x_train[:,2]/max(x_train[:,2])
    x_train[:,3] = x_train[:,3]/max(x_train[:,3])
    # one-hot encoding
    y_train = np_utils.to_categorical(y_train)



