***使用データに関して***
検出・識別に使用した自然画像は東京工科大学デザイン学部の中島健太准教授からご提供頂いた。
使用の際は中島先生と連絡を取って許可をいただけたら、GoogleDriveから共有していただだく。

■ 使用方法
1.学習
教師データの前処理としてdataフォルダの各種ファイル実行し、yolov5のフォルダのdataフォルダ下に格納されたアノテーション処理のなされた教師データを用いて、yolov5フォルダのtrain.pyによって学習を行う。

2.検出・識別
学習後のモデルを用いてdetect.pyにて予測を行う。

3.精度検証
yolov5フォルダのval.pyを実行。

■ 格納ファイル・フォルダ
● data
 このフォルダーには教師用データの前処理に関するファイルを含む。

・NIST-data
 学習に使う128×128pxの手書き文字データ。アルファベットの大文字と小文字を含む。総クラス数52種。
・sepa_data.py
 NIST-dataからクラスのそれぞれのデータの数を把握する。その各クラスにおける総データ数の中で最小値を各クラスの最大教師データ数としてファイルをコピーする。同時にtrain用とval用とtest用にも振り分ける。割合は8:1：1。
・get_annotation.py
 [train, valid, test]のそれぞれのディレクトリーに対して存在する教師画像データのアノテーション処理を行う。その結果をlabelsフォルダにテキストファイルとして格納する。(このときlabelsフォルダは存在していても、上書きされる)

● yolov5
 学習、識別検出を行う際に使用するフォルダと各種ファイルを含む。オリジナルのソースコードは(「https://github.com/ultralytics/yolov5」)を参照。本研究における変更・追加ファイルは、data.yaml、detect.py。

・requirements.txt
 各種モジュール推奨環境。pipでインストールを推奨。
 ex. pip install -r Requirements.txt
・data.yaml
 検出・識別を行いたいクラス名の設定。train, validation, testのそれぞれのデータ参照先の設定。
・train.py
 学習実行ファイル。
 ex. train.py --img 128 --epochs 100 --data data.yaml --weights yolov5s.pt
・detect.py
 検出・識別を行う実行ファイル。実行すると動画と予測した矩形領域の抽出結果が出てくる。矩形領域の抽出結果のファイル名には、動画をフレームごとに読み込んだ際のインクリメント処理の番号＋動画名＋クラス確率を付与する処理を追加。
 ex. python detect.py --source ./2H6A9795.mov --weights runs/train/exp2/weights/best.pt --img 1920 --save-crop --save-conf --save-txt
・val.py
 validation, testデータなどによるモデルの検証用ファイル。再現率、適合率、平均適合率が出ます。
 ex. python val.py --data data.yaml --weights runs/train/exp2/weights/best.pt --img 128 --task val
・calc_center.py (220630追記)
 保存したnpファイルから中心(平均)を求める。
・runs
学習結果、検出結果、精度検証結果を各種train, detect, valフォルダとして格納するフォルダ。それぞれフォルダが事前に存在しなくても実行可。
・yolov5s.pt
 事前学習済みモデル。学習に使う。

・data------hyp.scratch.yaml	ハイパーパラメータの設定。
	  --test		test用データ格納フォルダ
	  --train		train用データ格納フォルダ
	  --valid		validation用データ格納フォルダ
・model-----common.py		レイヤーごとの処理のモジュールをまとめたファイル
・utils-----autoanchor.py	データに対してアンカーが正しいかをチェックする。
	  --augmentations.py	画像処理の内容を含む。画像の複製、リサイズなど。
	  --callbacks.py	処理中に使用するcallbackを定義する。
	  --datasets.py		動画・画像を読み込む。フォルダの新規作成なども行う。
	  --general.py		座標やスケールの変換、非最大抑制(NMS)の処理、フォルダやファイルが重複した際のインクリメント処理を行う。
	  --loss.py		lossの計算処理を行う。
	  --metrics.py		モデルの精度を測る。各クラスにおける適合率、再現率の算出。それらの曲線グラフの作成。平均適合率の算出。
	  --plots.py		検出領域にラベルと矩形をプロット。その結果をディレクトリーに保存。学習結果(lossなど)をプロットし保存。
	  --torch_utils.py	使用cpuの設定。処理時間の計算。