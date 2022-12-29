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



if __name__ == '__main__':
    print(crayons.red("Main Program"))
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

    pca = PCA(n_components=2)
    pca.fit(x_train)
    feature = pca.transform(x_train)   # 主成分分析で変換 (60000,784)

    fig = plt.figure()
    plt.scatter(feature[:,0],feature[:,1],alpha=0.8,c=y_train)
    plt.legend()
    plt.xlabel("x0")
    plt.ylabel("x1")
    fig.savefig("PCA.png")
    plt.clf()
    plt.close()


