import matplotlib.pyplot as plt
import matplotlib.cm as cm

acc = [88, 90, 98, 99]
acc_comp = [66, 72, 94, 99]

fig, ax = plt.subplots(figsize = (8, 6))

#グラデーションで表現するためのカラーリストの定義
color_list = [cm.winter_r(i/3) for i in range(3)]
rect = ax.bar([1, 2, 3, 4], acc, color = color_list, width = 0.5, alpha = 0.80)
ax.set_ylim(80, 105)
ax.set_yticks(range(60,  110, 10))

#軸ラベル(tick)の変更とサイズ変更
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["50枚", "100枚", "1000枚", "1100枚"], fontname="MS Gothic")
ax.tick_params(labelsize = 15)

ax.set_ylabel("正解率", fontsize = 18, fontname="MS Gothic")
ax.set_xlabel("教師あり画像の枚数", fontsize = 18, fontname="MS Gothic")


#数字の挿入(アノテーション)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        #  annotationで文字やその位置を定義。文字を改行したいときは\nを挟む
        ax.annotate('{}%\n(SGAN)'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14)

autolabel(rect)
plt.show()