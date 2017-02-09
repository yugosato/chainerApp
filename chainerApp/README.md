
###ConvNetに画像データセット（3チャンネル画像）を学習させるコード

* 使用方法（準備編）  
学習を始める前に、データセット画像とラベルのリスト（Caffe準拠）を学習用と検証用それぞれ作成  
コマンド `python make_list.py '(学習/検証 画像フォルダへのパス)'`  
(ex) image_list_train.txt  
/home/yugo/LFW-Dataset/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg 0  
/home/yugo/LFW-Dataset/lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg 1

* 使用方法（学習編）  
コマンド `python main.py 'image_list_train.txt' 'image_list_test.txt'`  
で学習開始（学習終了後、学習済みモデル・ログが出力される）
* 各種変数オプション ※()はデフォルト値  
コマンド `python main.py --help`  
--arch, -a           #ネットワークモデル ('nin')  
--epoch, -E          #学習回数 (10)  
--batchsize, -B      #ミニバッチサイズ-学習 (32)  
--test_batchsize, -b #ミニバッチサイズ-検証 (250)  
--gpu, -g            #CPUorGPU (-1)  
--loaderjob, -j      #並列実行数 (none)  
--mean, -m           #平均画像 ('mean.npy')  
--root, -R           #データセットフォルダ ('.')  
--out, -o            #結果出力フォルダ ('result')  
(ex) googlenetの学習をGPUで回す場合  
コマンド `python main.py 'image_list_train.txt' 'image_list_test.txt' -a googlenet -g 0`
