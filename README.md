## Data preprocessing

### Datasets
ShanghaiTech partB 数据集请自行下载到与该md文件同级目录下

在训练前需要通过下述matlab代码生成密度图和分割图,训练集和测试集都要生成



### Density map generation
基于《**[ Counting-with-Focus-for-Free](https://github.com/shizenglin/Counting-with-Focus-for-Free)**》中等式（1）和（7）生成密度图，代码在¨preprocess/getDmap.m¨ 中。



### Segmentation map generation

基于《**[ Counting-with-Focus-for-Free](https://github.com/shizenglin/Counting-with-Focus-for-Free)**》中等式（2）生成分割图，使用与密度图生成相同的 sigma，代码在¨preprocess/getPmap.m¨ 中。



## Training

在训练前请在https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth 把预训练模型参数下载到与该md文件同级目录下。

1. 按照“data preprocessing”步骤准备数据。
2. 使用这个命令训练和测试：python headCounting_shanghaitech_segLoss.py

