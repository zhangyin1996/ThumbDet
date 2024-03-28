# ThumbDet
[Pattern Recognition 2023] Official Pytorch Code for ThumbDet: One thumbnail image is enough for object detection
## Abstract
Computer vision fields have witnessed great success thanks to deep convolutional neural networks (CNNs). However, state-of-the-art methods often benefit from large models and datasets, which introduce heavy parameters and computational requirements. Deploying such large models in real-world applications is very difficult because of the limited computing resources. Although many researchers focus on designing efficient block structures to compress model parameters, they ignore that the role of large-scale input images is also an important factor for algorithm efficiency. Reducing input resolution is a useful method to boost runtime efficiency, however, traditional interpolation methods assume a fixed degradation criterion that greatly hurts performance. To solve the above problems, in this paper, we propose a novel framework named ThumbDet for reducing model computation while maintaining detection accuracy. In our framework, we first design an image down-sampling module to learn a small-scale image that looks realistic and contains discriminative properties. Furthermore, we propose a distillation-boost supervision strategy to maintain the detection performance of small-scaled images as the original-size inputs. Extensive experiments conducted on a standard object detection dataset MS COCO demonstrate the effectiveness of the proposed method when using very low-resolution images (i.e. 4x down-sampling) as inputs. In particular, ThumbDet achieves satisfactory detection performance (i.e. 32.3% in mAP) while drastically reducing computation and memory requirements (i.e. speed up of 1.26x), outperforming the traditional interpolation methods ( e.g. bicubic) by +3.2% absolutely in terms of mAP.

![image text](https://github.com/zhangyin1996/ThumbDet/blob/main/pipline.png "Pipeline")

## 1. Preparation before you start
### a. Environment and detaset
+ This code is based on [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), please refer to it for environment installation and dataset preparation. Many thanks for this great work!
+ In our setting: **cuda=11.3, python=3.7, pytorch=1.7.0**
### b. Download weight of Teacher model
+ The weight of Teacher model could down from [baiduyun](https://pan.baidu.com/s/18CJN4cbpUpxOUNzrxoii1w?pwd=1234)(pswd:1234).
+ Put `r50_deformable_detr-checkpoint.pth` in `ThumbDet/` file.
### c. Replace the dataset path with your own
+ Modify the code `args.coco_path` (about line 115 in the `main.py`).

## 2. Train
+ cd ThumbDet/ 
+ The command for training ThumbDet on 8 RTX6000 GPUs is as following:
```
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```
+ For 4x downsample, the training time is about 4 days.
+ For 2x downsampling, you could modifiy the code `models/backbone.py` on line 94~98, and use the same training command.

## 3. Evaluation
+ You could use your trained model or use ours model from [baiduyun](https://pan.baidu.com/s/1Dl7VI7wHsW0TlP3dZNF2aA?pwd=1234)(pswd1234), **4x or 2x downsampling**.
+ The command for evaluating ThumbDet on one GPU is as following:
```
./configs/r50_deformable_detr.sh --resume <path to pre-trained model> --eval
```
## 4. Citation
If you find ThumbDet useful in your research, please consider citing:
```
@article{zhang2023thumbdet,
  title={ThumbDet: One thumbnail image is enough for object detection},
  author={Zhang, Yongqiang and Zhang, Yin and Tian, Rui and Zhang, Zian and Bai, Yancheng and Zuo, Wangmeng and Ding, Mingli},
  journal={Pattern Recognition},
  volume={138},
  pages={109424},
  year={2023},
  publisher={Elsevier}
}
```
