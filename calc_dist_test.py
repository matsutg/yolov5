"""
中心(平均)を求めたnpファイル(conv_ave)とテストデータ(test_conv_data)のユークリッド距離を算出
その距離が小さいものが中心に最も近いとみなせる(つまりラベルが判別される)
"""
import os
import codecs
import numpy as np


def main():
    # test_file = "train_65_01758.npy"
    test_folder = "test_conv_data"  # ロードするフォルダ
    test_convs_list = os.listdir(test_folder)
    load_conv_path = "conv_ave"
    convs_list = os.listdir(load_conv_path)  # ロードしたフォルダからリストとしてファイル名取得
    true_num, all_acc = 0, 0
    print(len(test_convs_list))

    for i, test_file in enumerate(test_convs_list):
        file_path = os.path.join(test_folder, test_file)
        load_file = np.load(file_path)
        temp, min_data = 0, 0  # 最小値を求める
        min_label = "0"

        for j, conv in enumerate(convs_list):  # 反応分布(A~z)とのユークリッド距離を測って一番近い距離のやつを判別
            conv_path = os.path.join(load_conv_path, conv)
            np_file = np.load(conv_path)  # 平均を求めた反応分布の取得

            dist = np.linalg.norm(np_file - load_file)
            # print(conv[0], "との比較", dist)
            if j == 0:
                min_data = dist
                min_label = conv[0]
            if dist < min_data and j != 0:
                min_data = dist
                min_label = conv[0]
        # 読み込み対象の正解ラベルをasciiに変換
        true_label16 = test_file[6:8]  # 16新数の格納
        # 16新数表記を文字に直す
        binary_str = codecs.decode(true_label16, "hex")
        true_label = str(binary_str, 'utf-8')
        # print("正解ラベル", true_label)
        if true_label == min_label:
            true_num += 1
            print("正解", true_label)
        else:
            print("不正解", true_label, min_label)

    all_acc = true_num / len(test_convs_list)
    print("正解率", all_acc)
    """クラスごとの正答率を見てもいいかもしれない"""

if __name__ == '__main__':
    main()
