"""
保存したnpファイルから中心(平均)を求める
"""
import os
import codecs
import numpy as np


def calc(sum_list, i, temp, save_folder, name):
    """
    :param sum_list: 合計
    :param i: 読み込み回数
    :param temp: 一つ前のインデックス
    :param save_folder: 保存フォルダのルートパス
    :param name: クラス名
    :return:
    """
    ave_list = sum_list / (i - temp)
    print("切り替わり", i - temp)
    print("平均", ave_list)
    save_path = os.path.join(save_folder, name)
    np.save(save_path + str(i) + ".npy", ave_list)  # 保存
    print("終了")


def main():
    load_path = "conv_data"  # ロードするフォルダ
    save_folder = "conv_ave"  # セーブするフォルダ
    data_class = "00"  # 読み込みファイルのクラス判別用変数

    files_list = os.listdir(load_path)  # ロードしたフォルダからリストとしてファイル名取得

    for i, file in enumerate(files_list):
        file_path = os.path.join(load_path, file)
        np_file = np.load(file_path)
        # クラスが異なる場合 data_class更新, sum_listの保存
        if data_class != file[6:8]:  # 16新数に該当する場所が(train_のあと)[6:8]の場合の適用
            if i != 0:  # 一番初め以降は処理
                calc(sum_list, i, temp, save_folder, name)  # 中心計算
            data_class = file[6:8]  # 16新数の格納
            # 16新数表記を文字に直す
            binary_str = codecs.decode(data_class, "hex")
            name = str(binary_str, 'utf-8')
            print(name)
            sum_list = np.zeros(len(np_file))  # listの合計を格納するlistの初期化
            temp = i  # クラス切り替わり時点のインデックス保存
        sum_list += np_file  # 配列の加算
    calc(sum_list, i, temp, save_folder, name)  # 最後のクラスだけ保存ができないのでここで保存


if __name__ == '__main__':
    main()

