import numpy as np
import pathlib
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.spatial.distance import chebyshev
from sklearn.decomposition import PCA
from sklearn import manifold  # MDSのライブラリimport
import torch
# t-SNE
from sklearn.manifold import TSNE
import codecs
import pandas as pd
from scipy.spatial import distance
import skimage.transform
from PIL import Image
import random

list_x = []
save_path = "./test_conv_data/"
load_path = "./conv_ave"
save_dist = "./Euc_dist"  # ユークリッド距離格納用フォルダ
# 同じ画像で変形したときとしていないときの距離の比較を行いたい。そのときの変形の名前を記述したい
# ファイル名：読み込んだファイル
save_compar_by_deform = "./Euc_dist/compar_by_deform/"

# 違う画像どうしで同じ変形のときの距離比較をしたい
# 書き込み構成は1行目に変形名とその下にファイル名と距離とそれを降順にしたものを記述
# ファイル名：deform
save_compar_by_files = "./Euc_dist/compar_by_files/"
min_label_list = []
min_data_list = []
file_iter_list = []
deform_flag = False
shear_flag = False
calc_flag = False
distort_flag = False
deform_name = ""
load_file_list = []
plot_conv_flag = False
flat_flag = False
copy_flag = False


def flat_list():  # 取得した特徴マップを平坦化する
    global list_x, flat_flag, copy_flag

    if flat_flag:
        return
    for i, x in enumerate(list_x):
        # print(x.shape)  # 処理前のデータのshapeの表示
        list_x[i] = torch.flatten(x)
        # list_x[i] = x.view(-1, x.size()[-1]) 2次元したければこれ
        # print(list_x[i].shape)  # 結合後のデータのshape

    list_x = torch.hstack((list_x[0], list_x[1], list_x[2]))
    print(list_x.size())
    if not plot_conv_flag:
        list_x = list_x.cpu().detach().numpy().copy()
        copy_flag = True
    flat_flag = True


def save_list(path):
    global list_x
    if not copy_flag:
        list_x = list_x.cpu().detach().numpy().copy()
    file_path = pathlib.Path(path)
    np.save(save_path + file_path.stem + ".npy", list_x)
    print("保存")


def calc_dist(path):
    global min_label_list, min_data_list, file_iter_list, load_file_list, list_x
    # print(path)
    if not copy_flag:
        list_x = list_x.cpu().detach().numpy().copy()
    print(list_x.shape)
    file_path = pathlib.Path(path)
    # file_name = file_path.stem
    # 読み込んだfile_pathから名前(file_path.stem)の先頭の番号のみを取り出す
    # file_iter_list.append((re.search(r'\d+', file_name)).group())
    # 読み込んだファイルパスを保存
    load_file_list.append(file_path)

    convs_list = os.listdir(load_path)  # ロードしたフォルダからリストとしてファイル名取得
    min_label = "0"  # 判別ラベル格納用
    dist_dict = {}

    for j, conv in enumerate(convs_list):  # 反応分布(A~z)とのユークリッド距離を測って一番近い距離のやつを判別
        conv_path = os.path.join(load_path, conv)
        np_file = np.load(conv_path)  # 平均を求めた反応分布の取得

        # ユークリッド距離
        # dist = np.linalg.norm(np_file - list_x)
        # マンハッタン距離
        # dist = np.linalg.norm(np_file - list_x, ord=1)
        # チェビシェフ距離
        dist = chebyshev(np_file, list_x)
        # print(conv[0], "との比較", dist)
        dist_dict.setdefault(conv[0], dist)  # 辞書に追加
        if j == 0:
            min_data = dist
            min_label = conv[0]
        if dist < min_data and j != 0:
            min_data = dist
            min_label = conv[0]

    #  最もユークリッド距離が小さいものをリストとして格納
    min_data_list.append(min_data)  # 最小値
    min_label_list.append(min_label)  # 最小値のときの判別ラベル
    # 大きい順に並べて，降順で表示
    dist_dic2 = sorted(dist_dict.items(), key=lambda x: x[1], reverse=True)  # 大きい順に並べる
    [print(i[0], "との比較", i[1]) for i in dist_dic2]
    # print("判別ラベル：", min_label, ", ユークリッド距離：", min_data)
    """
    print("判別ラベル：", second_min_label, ", ユークリッド距離：", second_min_data)
    if min_label == "f":
    print("一致", "判別ラベル：", min_label, ", ユークリッド距離：", min_data)
    """


# ユークリッド距離が最も小さかったデータとそのラベルとファイル名を取得, 表示
def search_min_data_file():
    global min_label_list, load_file_list, min_data_list
    min_data = min(min_data_list)
    min_data_index = min_data_list.index(min_data)
    label = min_label_list[min_data_index]
    path = load_file_list[min_data_index]
    print("最小値：", min_data, ", ラベル：", label, "\nファイルパス：", path)
    min_data_list = [min_data]
    min_label_list = [label]
    load_file_list = [path]


# 数字の挿入(アノテーション)
def autolabel(rects, ax):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        # annotationで文字やその位置を定義。文字を改行したいときは\nを挟む。
        ax.annotate(min_label_list[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14)


def plot_graph():
    """
    縦軸に最小距離
    横軸にファイル名
    グラフに識別名
    """
    acc = min_data_list  # 縦軸
    x_labels = file_iter_list  # 横軸
    bar_num_list = list(range(1, len(file_iter_list) + 1))

    fig, ax = plt.subplots(figsize=(8, 6))

    # グラデーションで表現するためのカラーリストの定義
    color_list = [cm.winter_r(i / 3) for i in range(3)]
    rect = ax.bar(bar_num_list, acc, color=color_list, width=0.5, alpha=0.80)
    # ax.set_ylim(80, 105)
    # ax.set_yticks(range(60, 110, 10))

    # 軸ラベル(tick)の変更とサイズ変更
    ax.set_xticks(bar_num_list)
    ax.set_xticklabels(x_labels, fontname="MS Gothic")  # x軸のグラフのラベル
    ax.tick_params(labelsize=10)

    ax.set_ylabel("最小距離", fontsize=13, fontname="MS Gothic")
    ax.set_xlabel("ファイル番号", fontsize=13, fontname="MS Gothic")
    # autolabel(rect, ax)
    plt.savefig('figure01.jpg')
    print("距離：", min(min_data_list))
    print("ファイル", file_iter_list[min_data_list.index(min(min_data_list))])


# 射影変換
def wrap_perspective(img):
    global deform_name, deform_flag
    deform_name = "wrap_perspec"
    deform_flag = True
    # 出力画像のパス
    output_file_path = "output.jpg"

    # 入力画像の読み込み
    # img = cv2.imread(input_file_path)
    # 　幅取得
    o_width = img.shape[1]
    # 　高さ取得
    o_height = img.shape[0]

    # 変換後の座標の指定 x1:左上　x2:右上 x3:左下 x4:右下
    x1 = np.array([0 + 32, 0])
    x2 = np.array([o_width - 64, 0])
    x3 = np.array([0 + 16, o_height])
    x4 = np.array([o_height - 16, o_height])

    # 変換前4点の座標　p1:左上　p2:右上 p3:左下 p4:右下
    p1 = np.array([0, 0])
    p2 = np.array([128, 0])
    p3 = np.array([0, 128])
    p4 = np.array([128, 128])

    # 変換前の4点
    src = np.float32([p1, p2, p3, p4])

    # 変換後の4点
    src_after = np.float32([x1, x2, x3, x4])

    # 変換後の4点 左上　右上 左下 右下
    dst = np.float32(src_after)

    # 変換行列
    M = cv2.getPerspectiveTransform(src, dst)

    # 射影変換・透視変換する
    output = cv2.warpPerspective(img, M, (o_width, o_height), borderValue=(0, 255, 255))

    print(output.shape)

    # 射影変換・透視変換した画像の保存
    cv2.imwrite(output_file_path, output)

    cv2.imshow('img', output)
    cv2.imshow('変換前', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output

    # top = (128/2)+(output.shape[0]/2)
    # left = (128)

    # output = add(output, white_img, )


# 欠損
def shear(img):
    global shear_flag
    shear_flag = True
    # 出力画像のパス
    output_file_path = "output.jpg"

    # 入力画像の読み込み
    # img = cv2.imread(input_file_path)

    # 　幅取得
    o_width = img.shape[1]
    print("幅取得：", o_width)
    # 　高さ取得
    o_height = img.shape[0]

    x0 = int(o_width / 4)
    y0 = int(o_height / 2)
    x1 = int(x0 + (o_width / 4))
    y1 = int(y0 + (o_height / 4))

    output = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)

    print(output.shape)

    # 射影変換・透視変換した画像の保存
    cv2.imwrite(output_file_path, output)

    cv2.imshow('img', output)
    cv2.imshow('変換前', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output


# 水滴みたいな歪み
def distort(img):
    # 出力画像のパス
    output_file_path = "output.jpg"
    # print(img.shape)
    # BGR→RGBの変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)

    # 画像サイズの取得
    height = np.shape(img)[0]
    width = np.shape(img)[1]

    # 水滴を落としたあとの画像として、元画像のコピーを作成。後処理で
    new_img = img.copy()
    center_x = [width * 2 / 3, width / 3]
    center_y = [height / 3, height / 3*2]
    for i in range(2):
        # 水滴の中心と半径の指定
        center = np.array((center_x[i], center_y[i]))
        r = 34
        print("中心：", center, "半径：", r)

        # ピクセルの座標を変換
        for x in range(width):
            for y in range(height):
                # dはこれから処理を行うピクセルの、水滴の中心からの距離
                d = np.linalg.norm(center - np.array((y, x)))

                # dが水滴の半径より小さければ座標を変換する処理をする
                if d < r:
                    # vectorは変換ベクトル。説明はコード外で。
                    vector = (d / r) ** 1.4 * (np.array((y, x)) - center)

                    # 変換後の座標を整数に変換
                    p = (center + vector).astype(np.int32)

                    # 色のデータの置き換え
                    new_img[y, x, :] = img[p[0], p[1], :]

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('img', new_img)
    cv2.imshow('変換前', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 射影変換・透視変換した画像の保存
    cv2.imwrite(output_file_path, new_img)
    return new_img


def add(f1, f2, out, top, left):
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)

    height, width = img1.shape[:2]
    img2[top:height + top, left:width + left] = img1

    cv2.imwrite(out, img2)
    return img2


def pca_func(data0):
    # 主成分分析による次元削減
    # pca = PCA(n_components=0.8, svd_solver='full', whiten=False)
    pca = PCA(n_components=3, whiten=False)
    pca.fit(data0)  # 対角化
    X_pca = pca.fit_transform(data0)  # 対角化して、dataを変換
    print(data0.shape, "->", X_pca.shape)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(data0)

    np.save("test.npy", transformed)
    # print(type(transformed))
    # 主成分の寄与率を出力します
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))


def mds_func(data0):
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100, random_state=0)  # 分離結果を左右する乱数とハイパーパラメータ
    X_mds = clf.fit_transform(data0)


def t_sne_func(data0):
    # embedding = TSNE(n_components=2, random_state=0)
    embedding = TSNE(n_components=3, random_state=0, perplexity=30.0)
    X_transformed = embedding.fit_transform(data0)
    # print(X_transformed.shape)  # (470, 2)


def plot_conv():
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
                print("データ", data.size())
                mean = torch.mean(data, axis=0)
                sample_data = torch.unsqueeze(list_x, dim=0)
                print("読み込みデータ", sample_data.size())
                mean = torch.unsqueeze(mean, dim=0)
                print("平均", mean.size())
                # 重心と識別データを追加
                data = torch.cat((data, mean, sample_data), dim=0)
                print("重心を追加", data.size())

                # 標準化
                norm_data = (data - mean) / torch.std(data, axis=0)
                # テンソルをnumpyにコピー
                norm_data = norm_data.to('cpu').detach().numpy().copy()
                print(data.size())
                pca_func(norm_data)
                # mds_func(norm_data)
                # t_sne_func(norm_data)

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


def test():
    """
    # plot_conv()
    """
    # 入力画像のパス
    path = "C:/Users/kikuchilab/PycharmProjects/yolo5/yolov5/data/test/images/train_41_01826.png"
    # path = "C:/Users/kikuchilab/PycharmProjects/yolo5/yolov5/output.jpg"
    img0 = cv2.imread(path)
    print("テスト実行")
    distort(img0)


if __name__ == '__main__':
    test()
