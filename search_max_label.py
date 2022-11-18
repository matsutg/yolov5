"""
検出画像のconfをソートする。
最大画像の表示
"""
import os
import re
from PIL import Image
import matplotlib.pyplot  as plt
"""
load_path = "C:/Users/kikuchilab/PycharmProjects/yolo5/Imagehash/220211/epoch100/2H6A9795/crops/f"

load_files = os.listdir(load_path)
# print(load_files)
conf_dict = {}

for i, load_file in enumerate(load_files):
    # conf部分の取得
    conf = float(re.findall('_(.*)_(.*)\.(.*)', load_file)[0][1])
    conf_dict.setdefault(conf, load_file)

#print(conf_dict)
#print("sort")
conf_dic2 = sorted(conf_dict.items(), reverse=True) #大きい順に並べる
#print(conf_dic2)
images = []
max_label = conf_dic2[0]
print(max_label)
file_path = os.path.join(load_path, max_label[1])
image = Image.open(file_path)
image.show(image)
"""
"""
for file_name in conf_dic2:

    file_path = os.path.join(load_path, file_name[1])
    image = Image.open(file_path)

    images.append(image)

fig = plt.figure()
for i, im in enumerate(images):
    fig.add_subplot(int(len(images)/9), 10, i + 1).set_title(str(i))
    plt.subplots_adjust(wspace=0.6, hspace=0.9)
    plt.imshow(im)
    if i == 10:
        break

plt.show()
"""

s = "2121_2H6A6942_0.25836339592933655.jpg"
conf_list = []

temp = re.findall('(.*)_(.*)_(.*)\.(.*)', s)
print(temp)

m = re.search(r'\d+', s)
r = m.group()
print(r)

# conf部分の取得
conf = float(re.findall('_(.*)_(.*)\.(.*)', s)[0][1])

print(conf)

conf_list.append(conf)
