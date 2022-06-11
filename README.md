<style>
</style>

首先将数据集分为train和test两个文件，test文件下有Annotations、ImageSets和JPEGImages三个子文件，train文件下也有Annotations、ImageSets和JPEGImages三个子文件。

![](C:\Users\jasin\AppData\Roaming\marktext\images\2022-06-12-00-28-26-image.png)

然后运行voc_label.py，将数据集转化为yolov5能训练的数据集

到yolov5官网上选择下载一个自己需要用到的权重文件，放入weights文件夹内

![](C:\Users\jasin\AppData\Roaming\marktext\images\2022-06-12-00-28-45-image.png)

根据自己需求来修改data目录下的myvoc文件

到models目录下打开自己下载的权重文件修改类别数目（nc）

打开train.py文件

![](C:\Users\jasin\AppData\Roaming\marktext\images\2022-06-12-00-28-54-image.png)

在该句中修改为自己下载的权重文件

parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5m.pt', help='initial weights path')

在该句中修改为在models文件下的权重文件所对应的yaml文件

parser.add_argument('--cfg', type=str, default='models/yolov5m.yaml', help='model.yaml path')

在该句中选择所要训练的次数

parser.add_argument('--epochs', type=int, default=1)

在该句中根据自己显卡的性能选择相应的batch-size

parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')

将训练完的模型放入yolov5-deepsort的weights文件里，然后将所需运行的视频放在video文件夹下然后运行main即可进行识别。

所需要的python库

# YOLOv5 requirements

# Usage: pip install -r

requirements.txt

# Base

----------------------------------------

matplotlib>=3.2.2

numpy>=1.18.5

opencv-python>=4.1.1

Pillow>=7.1.2

PyYAML>=5.3.1

requests>=2.23.0

scipy>=1.4.1 # Google Colab version

torch>=1.7.0

torchvision>=0.8.1

tqdm>=4.41.0

protobuf<=3.20.1 #
https://github.com/ultralytics/yolov5/issues/8012

# Logging

-------------------------------------

tensorboard>=2.4.1

# wandb

# Plotting

------------------------------------

pandas>=1.1.4

seaborn>=0.11.0

# Export

-------------------------------------

coremltools>=4.1 # CoreML export

onnx>=1.9.0 # ONNX export

onnx-simplifier>=0.3.6 # ONNX simplifier

scikit-learn==0.19.2 # CoreML quantization

tensorflow>=2.4.1 # TFLite export

 tensorflowjs>=3.9.0 # TF.js export

openvino-dev # OpenVINO export

# Extras

--------------------------------------

ipython # interactive notebook

psutil # system utilization

thop # FLOPs computation

#albumentations>=1.0.3

#pycocotools>=2.0 # COCO mAP

 roboflow

Pytorch版本1.8.0
