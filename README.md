# TexAtt

TexAtt是引入了外部的知识，实现对于训练样本的注意力优化的方法！


## How to use
run ```./TexAtt/main.py``` file to train and test models. 

Principal environmental dependencies as follows:
- [PyTorch 1.9.1](https://pytorch.org/)
- [pytorch-pretrained-bert 0.6.2](https://pypi.org/project/pytorch-pretrained-bert/)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)


## Training code



[//]: # (本机)
python main.py --seed 666  --data_path  E:/data/NLPCC_SD_2016/IphoneSE   --model_path  E:/model/white_model/chinesebert   

[//]: # (服务器)

python main.py --seed 351  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinesebert   --learning_rate  3e-5   --batch_size 8   --kg_each_seq_length 18  --kg_seq_length 128  --seq_length 128 --k 5  --epochs_num  15
lion 
              precision    recall  f1-score     support
0              0.478261  0.750000  0.584071   44.000000
1              0.480000  0.363636  0.413793   33.000000
2              0.839286  0.643836  0.728682   73.000000
accuracy       0.613333  0.613333  0.613333    0.613333
macro avg      0.599182  0.585824  0.575515  150.000000
weighted avg   0.654342  0.613333  0.616987  150.000000
micro avg      0.613333  0.613333  0.613333  150.000000


python main.py --seed 351  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinesebert   --learning_rate  1e-4   --batch_size 8   --kg_each_seq_length 18  --kg_seq_length 128  --seq_length 128 --k 5  --epochs_num  15
lion
              precision    recall  f1-score     support
0              0.293333  1.000000  0.453608   44.000000
1              0.000000  0.000000  0.000000   33.000000
2              0.000000  0.000000  0.000000   73.000000
accuracy       0.293333  0.293333  0.293333    0.293333
macro avg      0.097778  0.333333  0.151203  150.000000
weighted avg   0.086044  0.293333  0.133058  150.000000
micro avg      0.293333  0.293333  0.293333  150.000000



python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-bert-wwm   --learning_rate  3e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
 


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm-large   --learning_rate  15e-6   --batch_size 4   --kg_each_seq_length 32  --kg_seq_length 128  --seq_length 128 --k 5  --epochs_num  15
x


python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  3e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
             precision    recall  f1-score     support
0              0.612245  0.681818  0.645161   44.000000
1              0.583333  0.636364  0.608696   33.000000
2              0.815385  0.726027  0.768116   73.000000
accuracy       0.693333  0.693333  0.693333    0.693333
macro avg      0.670321  0.681403  0.673991  150.000000
weighted avg   0.704746  0.693333  0.696977  150.000000
micro avg      0.693333  0.693333  0.693333  150.000000


python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
              precision    recall  f1-score     support
0              0.682927  0.636364  0.658824   44.000000
1              0.534884  0.696970  0.605263   33.000000
2              0.803030  0.726027  0.762590   73.000000
accuracy       0.693333  0.693333  0.693333    0.693333
macro avg      0.673614  0.686454  0.675559  150.000000
weighted avg   0.708808  0.693333  0.697540  150.000000
micro avg      0.693333  0.693333  0.693333  150.000000


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 3  --epochs_num  15
              precision    recall  f1-score  support
0              0.566038  0.681818  0.618557    44.00
1              0.433962  0.696970  0.534884    33.00
2              0.909091  0.547945  0.683761    73.00
accuracy       0.620000  0.620000  0.620000     0.62
macro avg      0.636364  0.642244  0.612400   150.00
weighted avg   0.703934  0.620000  0.631881   150.00
micro avg      0.620000  0.620000  0.620000   150.00




python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter at fw_once
              precision    recall  f1-score     support
0              0.615385  0.727273  0.666667   44.000000
1              0.606061  0.606061  0.606061   33.000000
2              0.784615  0.698630  0.739130   73.000000
accuracy       0.686667  0.686667  0.686667    0.686667
macro avg      0.668687  0.677321  0.670619  150.000000
weighted avg   0.695692  0.686667  0.688599  150.000000
micro avg      0.686667  0.686667  0.686667  150.000000



python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter  
["bias", "LayerNorm.weight"]
              precision    recall  f1-score     support
0              0.615385  0.727273  0.666667   44.000000
1              0.606061  0.606061  0.606061   33.000000
2              0.784615  0.698630  0.739130   73.000000
accuracy       0.686667  0.686667  0.686667    0.686667
macro avg      0.668687  0.677321  0.670619  150.000000
weighted avg   0.695692  0.686667  0.688599  150.000000
micro avg      0.686667  0.686667  0.686667  150.000000


python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter cat前 fw_co
CB-NTR
              precision    recall  f1-score     support
0              0.524590  0.727273  0.609524   44.000000
1              0.500000  0.545455  0.521739   33.000000
2              0.849057  0.616438  0.714286   73.000000
accuracy       0.633333  0.633333  0.633333    0.633333
macro avg      0.624549  0.629722  0.615183  150.000000
weighted avg   0.677087  0.633333  0.641195  150.000000
micro avg      0.633333  0.633333  0.633333  150.000000


python main.py --seed 3562  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter cat前 fw_co
drop 0.3
CB-NTR
              precision    recall  f1-score     support
0              0.777778  0.636364  0.700000   44.000000
1              0.575000  0.696970  0.630137   33.000000
2              0.797297  0.808219  0.802721   73.000000
accuracy       0.733333  0.733333  0.733333    0.733333
macro avg      0.716692  0.713851  0.710953  150.000000
weighted avg   0.742666  0.733333  0.734621  150.000000
micro avg      0.733333  0.733333  0.733333  150.000000



python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 256  --seq_length 128 --k 6  --epochs_num  15
lion
adapter cat前 fw_co
drop 0.5
CB-NTR
              precision    recall  f1-score     support
0              0.622222  0.636364  0.629213   44.000000
1              0.512821  0.606061  0.555556   33.000000
2              0.803030  0.726027  0.762590   73.000000
accuracy       0.673333  0.673333  0.673333    0.673333
macro avg      0.646024  0.656151  0.649120  150.000000
weighted avg   0.686147  0.673333  0.677919  150.000000
micro avg      0.673333  0.673333  0.673333  150.000000


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 256  --seq_length 128 --k 6  --epochs_num  15
lion
adapter cat前 fw_co
drop 0.2
R-BCE-Focal
              precision    recall  f1-score     support
0              0.666667  0.454545  0.540541   44.000000
1              0.525000  0.636364  0.575342   33.000000
2              0.712500  0.780822  0.745098   73.000000
accuracy       0.653333  0.653333  0.653333    0.653333
macro avg      0.634722  0.623910  0.620327  150.000000
weighted avg   0.657806  0.653333  0.647748  150.000000
micro avg      0.653333  0.653333  0.653333  150.000000


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter cat前 fw_co
drop 0.2
R-BCE-Focal
multi-head

              precision    recall  f1-score  support
0              0.553571  0.704545  0.620000    44.00
1              0.463415  0.575758  0.513514    33.00
2              0.867925  0.630137  0.730159    73.00
accuracy       0.640000  0.640000  0.640000     0.64
macro avg      0.628304  0.636813  0.621224   150.00
weighted avg   0.686722  0.640000  0.650184   150.00
micro avg      0.640000  0.640000  0.640000   150.00


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
lion
adapter cat前 fw_co
drop 0.2
DB-loss
多头 4
              precision    recall  f1-score     support
0              0.620000  0.704545  0.659574   44.000000
1              0.533333  0.727273  0.615385   33.000000
2              0.872727  0.657534  0.750000   73.000000
accuracy       0.686667  0.686667  0.686667    0.686667
macro avg      0.675354  0.696451  0.674986  150.000000
weighted avg   0.723927  0.686667  0.693860  150.000000
micro avg      0.686667  0.686667  0.686667  150.000000



[//]: # (原版结构)
python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
              precision    recall  f1-score  support
0              0.482759  0.318182  0.383562    44.00
1              0.600000  0.272727  0.375000    33.00
2              0.575472  0.835616  0.681564    73.00
accuracy       0.560000  0.560000  0.560000     0.56
macro avg      0.552743  0.475509  0.480042   150.00
weighted avg   0.553672  0.560000  0.526706   150.00
micro avg      0.560000  0.560000  0.560000   150.00

python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/开放二胎   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
              precision    recall  f1-score  support
0              0.482759  0.318182  0.383562    44.00
1              0.600000  0.272727  0.375000    33.00
2              0.575472  0.835616  0.681564    73.00
accuracy       0.560000  0.560000  0.560000     0.56
macro avg      0.552743  0.475509  0.480042   150.00
weighted avg   0.553672  0.560000  0.526706   150.00
micro avg      0.560000  0.560000  0.560000   150.00



python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/开放二胎   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
DBloss
              precision    recall  f1-score  support
0              0.284672  0.886364  0.430939     44.0
1              0.461538  0.181818  0.260870     33.0
2              0.000000  0.000000  0.000000     73.0
accuracy       0.300000  0.300000  0.300000      0.3
macro avg      0.248737  0.356061  0.230603    150.0
weighted avg   0.185042  0.300000  0.183800    150.0





python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/开放二胎   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  1e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
BCELOSS
              precision    recall  f1-score     support
0              0.444444  0.181818  0.258065   44.000000
1              0.541667  0.393939  0.456140   33.000000
2              0.574074  0.849315  0.685083   73.000000
accuracy       0.553333  0.553333  0.553333    0.553333
macro avg      0.520062  0.475024  0.466429  150.000000
weighted avg   0.528920  0.553333  0.509457  150.000000
micro avg      0.553333  0.553333  0.553333  150.000000




----以上都是32
python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/开放二胎   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
base
8
adamw
              precision    recall  f1-score     support
0              0.625000  0.462963  0.531915   54.000000
1              0.833333  0.166667  0.277778   30.000000
2              0.528846  0.833333  0.647059   66.000000
accuracy       0.566667  0.566667  0.566667    0.566667
macro avg      0.662393  0.487654  0.485584  150.000000
weighted avg   0.624359  0.566667  0.531751  150.000000
micro avg      0.566667  0.566667  0.566667  150.000000

attTEX
              precision    recall  f1-score     support
0              0.750000  0.833333  0.789474   54.000000
1              0.466667  0.466667  0.466667   30.000000
2              0.750000  0.681818  0.714286   66.000000
accuracy       0.693333  0.693333  0.693333    0.693333
macro avg      0.655556  0.660606  0.656809  150.000000
weighted avg   0.693333  0.693333  0.691830  150.000000
micro avg      0.693333  0.693333  0.693333  150.000000




python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/俄罗斯在叙利亚的反恐行动   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
DBloss

base
              precision    recall  f1-score     support
0              0.349206  0.385965  0.366667   57.000000
1              0.000000  0.000000  0.000000   32.000000
2              0.413793  0.590164  0.486486   61.000000
accuracy       0.386667  0.386667  0.386667    0.386667
macro avg      0.254333  0.325376  0.284384  150.000000
weighted avg   0.300974  0.386667  0.337171  150.000000
micro avg      0.386667  0.386667  0.386667  150.000000


attTEX
              precision    recall  f1-score     support
0              0.462963  0.438596  0.450450   57.000000
1              0.545455  0.187500  0.279070   32.000000
2              0.435294  0.606557  0.506849   61.000000
accuracy       0.453333  0.453333  0.453333    0.453333
macro avg      0.481237  0.410885  0.412123  150.000000
weighted avg   0.469309  0.453333  0.436825  150.000000
micro avg      0.453333  0.453333  0.453333  150.000000



python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/深圳禁摩限电   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15

base
              precision    recall  f1-score     support
0              0.519380  0.893333  0.656863   75.000000
1              0.153846  0.071429  0.097561   28.000000
2              0.800000  0.090909  0.163265   44.000000
accuracy       0.496599  0.496599  0.496599    0.496599
macro avg      0.491075  0.351890  0.305896  147.000000
weighted avg   0.533750  0.496599  0.402586  147.000000
micro avg      0.496599  0.496599  0.496599  147.000000


attTEX
              precision    recall  f1-score     support
0              0.828947  0.840000  0.834437   75.000000
1              0.653846  0.607143  0.629630   28.000000
2              0.822222  0.840909  0.831461   44.000000
accuracy       0.795918  0.795918  0.795918    0.795918
macro avg      0.768339  0.762684  0.765176  147.000000
weighted avg   0.793582  0.795918  0.794535  147.000000
micro avg      0.795918  0.795918  0.795918  147.000000


python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/春节放鞭炮   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
bert
              precision    recall  f1-score     support
0              0.651515  0.741379  0.693548   58.000000
1              0.344828  0.434783  0.384615   23.000000
2              0.709091  0.565217  0.629032   69.000000
accuracy       0.613333  0.613333  0.613333    0.613333
macro avg      0.568478  0.580460  0.569065  150.000000
weighted avg   0.630975  0.613333  0.616501  150.000000
micro avg      0.613333  0.613333  0.613333  150.000000


texatt
              precision    recall  f1-score  support
0              0.735294  0.862069  0.793651    58.00
1              0.588235  0.434783  0.500000    23.00
2              0.830769  0.782609  0.805970    69.00
accuracy       0.760000  0.760000  0.760000     0.76
macro avg      0.718100  0.693153  0.699874   150.00
weighted avg   0.756664  0.760000  0.754291   150.00
micro avg      0.760000  0.760000  0.760000   150.00

python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15
bert
0              0.470588  0.181818  0.262295   44.000000
1              0.375000  0.090909  0.146341   33.000000
2              0.504000  0.863014  0.636364   73.000000
accuracy       0.493333  0.493333  0.493333    0.493333
macro avg      0.449863  0.378580  0.348333  150.000000
weighted avg   0.465819  0.493333  0.418832  150.000000
micro avg      0.493333  0.493333  0.493333  150.000000




texatt
              precision    recall  f1-score  support
0              0.630435  0.659091  0.644444     44.0
1              0.558140  0.727273  0.631579     33.0
2              0.852459  0.712329  0.776119     73.0
accuracy       0.700000  0.700000  0.700000      0.7
macro avg      0.680344  0.699564  0.684048    150.0
weighted avg   0.722582  0.700000  0.705696    150.0
micro avg      0.700000  0.700000  0.700000    150.0



python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chinese-roberta-wwm   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15

llm test
python main.py --seed 42  --data_path  /home/yindechun/Desktop/yanyu/data/NLPCC_SD_2016/IphoneSE   --model_path  /home/yindechun/Desktop/yanyu/model/blank/chatyuan   --learning_rate  2e-5   --batch_size 8   --kg_each_seq_length 32  --kg_seq_length 168  --seq_length 128 --k 5  --epochs_num  15