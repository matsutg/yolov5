import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
import codecs
import gc
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn import manifold  # MDSのライブラリimport
import torch
# t-SNE
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import temp_x


def pca_func(data0, name0, ax1):
    # 主成分分析による次元削減
    # pca = PCA(n_components=0.8, svd_solver='full', whiten=False)
    pca = PCA(n_components=3, whiten=False)
    pca.fit(data0)  # 対角化
    X_pca = pca.fit_transform(data0)  # 対角化して、dataを変換
    print(data0.shape, "->", X_pca.shape)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(data0)
    df = pd.DataFrame(data=transformed,
                      columns=['A', 'B', 'C'])

    ax1.scatter(df['A'], df['B'], df['C'], alpha=0.3, label=name0, s=10)
    ax1.scatter(df.iloc[-1]['A'], df.iloc[-1]['B'], df.iloc[-1]['C'], alpha=1, label=name0 + "_mean", s=20, c="red")

    """
    x = df['A'].to_numpy()
    y = df['B'].to_numpy()
    z = df['C'].to_numpy()
    surf = ax1.plot_trisurf(x, y, z, label=name0, cmap="winter")
    # ax1.plot_surface(x_new, y_new, z_new, label=name0, cmap="winter")
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    """
    # 主成分の寄与率を出力します
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))


def mds_func(data0, name0, ax1):
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100, random_state=0)  # 分離結果を左右する乱数とハイパーパラメータ
    X_mds = clf.fit_transform(data0)
    df = pd.DataFrame(data=X_mds,
                      columns=['A', 'B'])
    ax1.scatter(df['A'], df['B'], alpha=0.3, label=name0, s=10)


def t_sne_func(data0, name0, ax1):
    # embedding = TSNE(n_components=2, random_state=0)
    embedding = TSNE(n_components=3, random_state=0, perplexity=30.0)
    X_transformed = embedding.fit_transform(data0)
    # print(X_transformed.shape)  # (470, 2)

    df = pd.DataFrame(data=X_transformed,
                      columns=['A', 'B', 'C'])

    # ax1.plot_scatter(df['A'], df['B'], df['C'], label=name0)  # , alpha=0.3, s=10

    x = df['A'].to_numpy()
    y = df['B'].to_numpy()
    z = df['C'].to_numpy()
    surf = ax1.plot_trisurf(x, y, z, label=name0)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d


def main():
    # 統計情報の集計
    """
    # mean
    conv_path = "conv_ave/A1536.npy"
    mu_mat = np.load(conv_path)
    print("mu_mat", mu_mat)
    print("mu_mat.shape", mu_mat.shape)
    """

    # data - mean
    load_path = "conv_data"
    files_list = os.listdir(load_path)
    data_class = "00"

    # 読み込みたいデータかもしれない
    path = "data/test/images/train_41_01734.png"

    fig2 = plt.figure(figsize=(8, 5), facecolor="w")
    ax1 = fig2.add_subplot(111, projection='3d')  # fig2.add_subplot(projection='3d')
    # ax1 = Axes3D(plt.figure())

    for i, file in enumerate(files_list):
        file_path = os.path.join(load_path, file)
        np_file = np.load(file_path)
        tensor_data = torch.from_numpy(np_file.astype(np.float32)).clone()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor_data = tensor_data.to(device)
        # クラスが異なる場合 data_class更新
        if data_class != file[6:8]:  # 16新数に該当する場所が(train_のあと)[6:8]の場合の適用
            # print(file[6:8])
            # print("切り替わり", i)
            if i != 0:  # 一番初め以降は処理
                # if i > 1537:
                #    break
                # print(data)
                data = data.view(c, -1)  # [データ数, 次元数]の形
                print("サンプルデータサイズ", data.size())
                mean = torch.mean(data, axis=0)
                mean = torch.unsqueeze(mean, dim=0)
                print(mean.size())
                # 重心を追加
                data = torch.cat((data, mean), dim=0)
                print("重心を追加", data.size())

                # 標準化
                norm_data = (data - mean) / torch.std(data, axis=0)
                # テンソルをnumpyにコピー
                norm_data = norm_data.to('cpu').detach().numpy().copy()
                print(data.size())
                pca_func(norm_data, name, ax1)
                # mds_func(norm_data, name, ax1)
                # t_sne_func(norm_data, name, ax1)

                # if i > 1537 * 3:
                break

            data_class = file[6:8]  # 16新数の格納
            # 16新数表記を文字に直す
            binary_str = codecs.decode(data_class, "hex")
            name = str(binary_str, 'utf-8')
            print(name)
            data = tensor_data
            data.view(1, -1)
            c = 0
        c += 1
        if c != 1:
            data = torch.cat((data, tensor_data))
    """
    # pca_func(data)
    data = data.view(c, -1)
    mds_func(data, name)
    """

    ax1.set_title('principal component')
    ax1.set_xlabel('sample')
    ax1.set_ylabel('value')
    # ax1.set_aspect('equal')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid()
    plt.show()
    """
    # 主成分をプロットする
    
    # 花の種類別で色を変える
    for label in np.unique(Y):
        if label == 0:
            c = "red"
        elif label == 1:
            c = "blue"
        elif label == 2:
            c = "green"
        else:
            pass
        plt.scatter(transformed[Y == label, 0],
                    transformed[Y == label, 1],
                    c=c)
    
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.show()
    """
    """
    print(data.shape)
    
    data_m_mat = data - mu_mat
    print("data_m_mat", data_m_mat)
    print("data_m_mat.shape", data_m_mat.shape)
    """
    """
    # covariance matrix
    cov_mat = np.cov(data.T)
    print("dov_mat.shape", cov_mat.shape)
    
    del data
    del mu_mat
    gc.collect()
    
    cov_i_mat = np.linalg.pinv(cov_mat)
    print(cov_i_mat)
    
    del cov_mat
    gc.collect()
    
    mahala_result = np.sqrt(np.sum(np.dot(data_m_mat, cov_i_mat) * data_m_mat, axis=1))
    
    fig2 = plt.figure(figsize=(8, 5))
    ax1 = fig2.add_subplot()
    ax1.plot(mahala_result)
    ax1.set_title('Mahalanobis Distance')
    ax1.set_xlabel('sample')
    ax1.set_ylabel('Mahalanobis Distance')
    # ax1.set_aspect('equal')
    ax1.grid()
    """


def plot_2():
    load_npy = "test.npy"
    np_file = np.load(load_npy)
    fig2 = plt.figure(figsize=(8, 5), facecolor="w")
    ax1 = fig2.add_subplot(111, projection='3d')  # fig2.add_subplot(projection='3d')
    # ax1 = Axes3D(plt.figure())
    df = pd.DataFrame(data=np_file,
                      columns=['A', 'B', 'C'])

    ax1.scatter(df['A'], df['B'], df['C'], alpha=0.3, label="A", s=10)
    ax1.scatter(df.iloc[-2]['A'], df.iloc[-2]['B'], df.iloc[-2]['C'], alpha=1, label="A_mean", s=20, c="yellow")
    ax1.scatter(df.iloc[-1]['A'], df.iloc[-1]['B'], df.iloc[-1]['C'], alpha=1, label="data", s=20, c="red")
    ax1.set_title('principal component')
    ax1.set_xlabel('sample')
    ax1.set_ylabel('value')
    # ax1.set_aspect('equal')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid()
    plt.show()


if __name__ == '__main__':
    plot_2()
