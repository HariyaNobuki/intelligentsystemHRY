# python 3.7 plz
import crayons
import pandas as pd
import numpy as np
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

    print("--- explained_variance_ratio_ ---")
    print(pca.explained_variance_ratio_)
    print("--- components ---")
    print(pca.components_) 
    print("--- mean ---")
    print(pca.mean_) 
    print("--- covariance ---")
    print(pca.get_covariance())

    # いくつの成分を用いて適用するべきなのかを議論する方法
    ev_ratio = pca.explained_variance_ratio_    # これは何を表すのかはわからないけど分析の方法は割れてきた
    ev_ratio = np.hstack([0,ev_ratio.cumsum()])

    df_ratio = pd.DataFrame({"components":range(len(ev_ratio)), "ratio":ev_ratio})

    fig = plt.figure()
    plt.plot(ev_ratio)
    plt.xlabel("components")
    plt.ylabel("explained variance ratio")
    fig.savefig("NumCluster.png")
    plt.clf()
    plt.close()

    fig = plt.figure()
    plt.scatter(range(len(ev_ratio)),ev_ratio)
    fig.savefig("scat_evrat.png")
    plt.clf()
    plt.close()

    KM = KMeans(n_clusters = 10)
    result = KM.fit(feature[:,:9])

    df_eval = pd.DataFrame(confusion_matrix(y_train,result.labels_))   # 混同行列
    df_eval.columns = df_eval.idxmax()  # 縦に予測している
    df_eval = df_eval.sort_index(axis=1)

    # ここで失敗を確認できる
    print(df_eval)

    #クラスタの中のデータの最も多いラベルを正解ラベルとしてそれが多くなるようなクラスタ数を探索
    eval_acc_list=[]

    for i in range(5,15):
        KM = KMeans(n_clusters = i)
        result = KM.fit(feature[:,:9])
        df_eval = pd.DataFrame(confusion_matrix(train_labels,result.labels_))
        eval_acc = df_eval.max().sum()/df_eval.sum().sum()
        eval_acc_list.append(eval_acc)

    fig = plt.figure()
    plt.plot(range(5,15),eval_acc_list)
    plt.xlabel("The number of cluster")
    plt.ylabel("accuracy")
    fig.savefig("Thenumberofcluster.png")
    plt.clf()
    plt.close()