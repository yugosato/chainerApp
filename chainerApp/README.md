
### CNNに画像データセット（3チャンネル画像）を学習させるコード（クラス分類）
#### 使用方法（準備）  
※ データセット画像はクラス毎にディレクトリに分けておく
1. データセット画像パス・ラベルのリストを学習用と検証用それぞれ作成
1. データセット平均画像を作成 (正規化用)
1. 作業ディレクトリへコピー
```
$ cd script
$ python makelist.py "(学習/検証 画像フォルダへのパス)"
$ python compute_mean.py images.txt
$ cp images.txt mean.npy chainerApp
```
（リスト例） images.txt
```
/home/yugo/LFW-Dataset/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg 0
/home/yugo/LFW-Dataset/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg 1
:
```

#### 使用方法（学習）  
作業ディレクトリで
```
$ python trainmodel.py "images_train.txt" "images_test.txt"
```
で学習開始（学習終了後、学習済みモデル・ログ・学習経過画像が出力される）
* オプション
```
--help, -h
--arch, -a            # ネットワークモデル (default='nin')
--epoch, -E           # 学習回数 (default=10)
--batchsize, -B       # ミニバッチサイズ 学習 (default=32)
--test_batchsize, -b  # ミニバッチサイズ 検証 (default=250)
--gpu, -g             # GPU (default=-1)
--mean, -m            # 正規化用平均画像 (default='mean.npy')
--root, -R            # データセットディレクトリ (default='.')  
--out, -o             # 結果出力ディレクトリ (default='result')  
```

（googlenetの学習をGPUで回す場合の例）  
```
$ python trainmodel.py "images_train.txt" "images_test.txt" -a googlenet -g 0
```
