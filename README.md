# AIexperiment  TinySSD
## 人工智能综合实验作业
### 一、环境配置  

CUDA 版本 >= 11.7  
Python >= 3.8 （我使用的是3.10）  
PyTorch >= 1.10（我是用的是1.10）  

下载代码：
```
git clone https://github.com/Cathrinex/AIexperiment.git
### 二、准备数据集
1.确定待检测对象，得到jpg格式图片，放在detection/target/下  
2.下载背景图片1000张左右，放在detection/background/下  
3. 生成训练样本，放在detection/sysu_train/下，可调用detection/create_train.py实现自动合成detection/sysu_train/images里面是合成的训练样本，detection/sysu_train/label.csv里面是标注信息  
4.准备一张测试图片，放在detection/test/下  

### 三、训练
首先修改data.py的路径为自己电脑的数据集的路径，之后运行train.py进行训练模型，可以修改batchsize和epoch，训练结束之后可以得到模型。  
### 测试
首先修改test.py的路径为自己电脑的测试集的路径，并修改模型的的名字为想要使用的模型，之后运行test.py即可得到目标检测后的图片。  

### 四、测试结果
下面的三张图是用初始的epoch=30的预训练模型测试的结果：
![Figure_1_epoch30](https://user-images.githubusercontent.com/117092266/199030825-b9305f57-def7-40c9-b576-23dca9e369bb.png)
![Figure_2_epoch30](https://user-images.githubusercontent.com/117092266/199030852-7306e876-b435-49e9-b8e9-4dd61314f969.png)
![Figure_3_epoch30](https://user-images.githubusercontent.com/117092266/199030878-daac98cb-7fb5-4ec3-b942-b4e460253b58.png)

### 五、效果提升
为了提升目标检测的效果，我尝试了一些数据增强的方法，在训练阶段中对图片加入了高斯噪声，之后将训练的次数增加到了190，测试结果的对比如下：  
![image](https://user-images.githubusercontent.com/117092266/199035298-6d44b2b5-29dd-4952-9739-675de4bb8bfd.png)
左侧的图片是效果提升前的图片，右侧的图片是效果提升后的图片，我们可以看到进行了效果提升后，目标检测的效果明显变好了。
