## 一、项目背景介绍
   <font size=4> 
   随着国内汽车保有量和换购人群迅速增加，二手车市场规模不断扩大。与欧美成熟市场相比，我国的二手车市场尚处在起步阶段。据协会不完全统计，我国二手车市场在遭受疫情前一直呈现快速增长趋势，从细分市场结构来看，基本型乘用车类型、3-6年车龄、3万元及以下价格的二手车受到市场的欢迎，为了拓宽车辆二手市场，几大app（懂车帝/瓜子二手车）上为了方便二手汽车回收给用户设置了对二手汽车估价的平台，而汽车分类对汽车回收价格有较大关系，通过图像分类的技术，我们可以快速的确定车辆的型号和年份，对车辆估值起到一个重大作用。

## 二、数据集介绍

<font size=4>数据集仅包含图片文件，并已根据类别以文件夹形式存放。
来源：https://ai.stanford.edu/~jkrause/cars/car_dataset.html

<font size=4>Stanford Cars汽车数据集包含196类汽车的16,185张图像。数据被分成8,144张训练图像和8,041张测试图像，其中每个类别大致上被分成了50-50。类别通常是在品牌、型号、年份的层面上，例如，2012年特斯拉Model S或2012年宝马M3双门跑车。

<font size=4>===== BibTeX =====

<font size=4>@inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
  title = {3D Object Representations for Fine-Grained Categorization},
  booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
  year = {2013},
  address = {Sydney, Australia},
  author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
}
 
<font size=4>===== 数据存放结构如下：=====

<font size=4>
    
``` 
Stanford Cars
   |———————label1
   |        └——————xx05.jpg
   |        └——————xx06.jpg
   └——————label2
   |        └——————xxxx.jpg
   |        └——————xxxx.jpg
   |———————label1
   |        └——————xx05.jpg
   |        └——————xx06.jpg
   | ———————......        
            
```
数据集图片展示

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/d846c366b4de497a87c82843c30f68a1d9d6d596b8bd4e5f9b7dde2fe235745a" width = "50%" height = "50%" /><img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/11e610a2effd4245a0dd67fe955ef971e402c1e708b34a9e9dd493d4951c7885" width = "50%" height = "50%" />


## 三、模型介绍
想写好一个好的精品项目，项目的内容必须包含理论内容和实践相互结合，该部分主要是理论部分，向大家介绍一下你的模型原理等内容
### ResNet

<font size=4 >ResNet(Residual Network)是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。针对随着网络训练加深导致准确度下降的问题，ResNet提出了残差学习方法来减轻训练深层网络的困难。在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含两条路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。
    
<font size=4 >残差模块如图1所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。</font>

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/resnet_block.jpg?raw=true" width="400"><br/>
<font size=4 >图1. 残差模块
</p>

<font size=4 >图2展示了50、101、152层网络连接示意图，使用的是瓶颈模块。这三个模型的区别在于每组中残差模块的重复次数不同(见图右上角)。ResNet训练收敛较快，成功的训练了上百乃至近千层的卷积神经网络。

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/resnet.png?raw=true"><br/>
<font size=4 >图2. 基于ImageNet的ResNet模型
</p>


# 四、数据准备和模型训练
## 4.1、 数据准备
    
   <font size=4>本项目采用Stanford Cars汽车数据集[Stanford Cars](https://aistudio.baidu.com/aistudio/datasetdetail/85765) 
    
   <font size=4>通过解压命令将数据集解压到work文件夹中
    
   



```python
%cd /home/aistudio/work
!tar -xf ../data/data85765/Stanford_Cars.tar
## 解压数据集到work目录下
```


```python
! pip install "paddlex<=2.0.0" -i https://mirror.baidu.com/pypi/simple
```

<font size=4>解压后我们可以发现

<font size=4>运行前

![](https://ai-studio-static-online.cdn.bcebos.com/91084dd347e94502b070f7769d907dff0359a097a8e9482d8a39da52659cdaa1)

<font size=4>在文件夹中，文件夹的目录有空格，会影响imageNet格式数据集的读取，所以我们运行下面程序，将空格转化为下划线

<font size=4>运行后

![](https://ai-studio-static-online.cdn.bcebos.com/a61731d8906c4125ba68b6844160b9353ed95fea07274fa09e49be5f9e0fa5f0)





```python
import os
image_path = "work/train"
for data_dir in os.listdir("work/train"):
    newName = data_dir.replace(' ', '_')
    os.rename(os.path.join(image_path,data_dir),os.path.join(image_path,newName))
```

<font size=4>我们执行PaddleX指令，生成train_list.txt、labels.txt、val_list.txt

<font size=4>操作详解：[https://paddlex.readthedocs.io/zh_CN/release-1.3/data/format/classification.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/data/format/classification.html)



```python
import os
os.system("paddlex --split_dataset --format ImageNet --dataset_dir work/test --val_value 0.2 --test_value 0.1")
```

# 五、模型训练
##  <font size=4>5.1、我们使用PaddleX来实现模型的训练，我们提前装好PaddleX

<font size=4>PaddleX文档：[https://paddlex.readthedocs.io/zh_CN/release-1.3/quick_start.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/quick_start.html)



## 5.2、数据集读取

<font size=4>我们安装好了PaddleX，我们首先确定我们是一个图像分类项目
    
![image.png](attachment:8083610f-2e2f-487f-82db-1699d24fed6e.png)

<font size=4>PaddleX共提供了20+的图像分类模型，可满足开发者不同场景的需求下的使用。

<font size=4>从数据处理开始，我们确定数据集格式为imageNet格式

<font size=4>[文档](https://paddlex.readthedocs.io/zh_CN/release-1.3/data/format/classification.html)介绍了如何去加载imageNet格式数据集



```python
import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-imagenet
train_dataset = pdx.datasets.ImageNet(
    data_dir='work/train',
    file_list='work/train/train_list.txt',
    label_list='work/train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='work/train',
    file_list='work/train/val_list.txt',
    label_list='work/train/labels.txt',
    transforms=eval_transforms)

```

## 5.3、配置好参数开始训练

<font size=4>参数详解 ：[https://paddlex.readthedocs.io/zh_CN/release-1.3/appendix/parameters.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/appendix/parameters.html)



```python
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet50_vd_ssld(num_classes=num_classes)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/parameters.md
model.train(
    num_epochs=20,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    warmup_steps = 5,
    warmup_start_lr = 0.005,
    lr_decay_epochs=[8,12, 14],
    lr_decay_gamma = 0.5,
    learning_rate=0.01,

    save_dir='output/ResNet50_vd_ssld',
    use_vdl=True)
```

<font size=4>完整代码如下：


```python
import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-imagenet
train_dataset = pdx.datasets.ImageNet(
    data_dir='work/train',
    file_list='work/train/train_list.txt',
    label_list='work/train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='work/train',
    file_list='work/train/val_list.txt',
    label_list='work/train/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet50_vd_ssld(num_classes=num_classes)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/parameters.md
model.train(
    num_epochs=20,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    warmup_steps = 5,
    warmup_start_lr = 0.005,
    lr_decay_epochs=[8,12, 14],
    lr_decay_gamma = 0.5,
    learning_rate=0.01,

    save_dir='output/ResNet50_vd_ssld',
    use_vdl=True)


```

# 六、模型评估
该部分主要是对训练好的模型进行评估，可以是用验证集进行评估，或者是直接预测结果。评估结果和预测结果尽量展示出来，增加吸引力。


```python
import paddlex as pdx
test_jpg = 'work/train/Dodge_Caliber_Wagon_2007/006824.jpg'
model = pdx.load_model('output/ResNet50_vd_ssld/best_model')
result = model.predict(test_jpg)
print("Predict Result: ", result)
```

    2022-02-26 21:51:01 [INFO]	Model[ResNet50_vd_ssld] loaded.
    Predict Result:  [{'category_id': 61, 'category': 'Chevrolet_HHR_SS_2010', 'score': 1.0}]


# 七、项目小结
（1）本项目基于PaddleX工具，加载imagenet格式数据集在Resnet模型上训练，可以看出PaddleX使得代码更加简洁清晰，可以快速的搭建模型训练全流程，从模型效果上看，模型还有很多改进和学习的地方。

（2）从各个方面来讲，模型调优和数据增强也是提高模型精度的一个重要的点，后续可以在数据和模型参数上下功夫，欢迎大家来改进调优模型，。 


# 八、作者简介
江西理工大学大三在读 智能科学与技术专业

对目标检测、OCR识别、slam方面比较感兴趣

欢迎有相同兴趣的与我交流学习，感谢飞桨提供的平台。
 
 欢迎大家fork，一起交流学习

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
