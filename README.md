# end-to-end-for-plate-recognition
多标签分类,端到端的中文车牌识别基于mxnet .
从[xlvector的ocr代码](https://github.com/szad670401/learning-dl/tree/master/mxnet/ocr)修改，减少了参数，由于我没有显卡。单线程 9 samples/s 速度 ，用CPU在MBP上跑了50w张样本。识别率到了81%。不过还没有完全收敛。

## 训练好的模型
https://github.com/ibyte2011/end-to-end-for-chinese-plate-recognition

## 关于车牌识别
生成的车牌对于实际车牌并不是效果很好，在结合真实样本和GAN，训练了一个更好的模型，对真实车牌表现很好。
并实现了一整套车牌识别的系统命名为HyperLPR https://github.com/zeusees/HyperLPR

## 依赖:
 + Numpy
 + Mxnet
 + Opencv
 
## 生成的车牌样张
通过渲染车牌加上畸变、噪声、与自然环境结合生成车牌的样本。

 ![image](./recognize_samples/00.jpg)
  ![image](./recognize_samples/01.jpg)
   ![image](./recognize_samples/02.jpg)
    ![image](./recognize_samples/03.jpg)
     ![image](./recognize_samples/04.jpg)
        ![image](./recognize_samples/06.jpg)
    ![image](./recognize_samples/07.jpg)
     ![image](./recognize_samples/08.jpg)   ![image](./recognize_samples/02.jpg)
    ![image](./recognize_samples/09.jpg)
     ![image](./recognize_samples/10.jpg)
         ![image](./recognize_samples/11.jpg)
## 识别样张

<img src='./recognize_samples/Screen Shot 2016-08-07 at 12.51.56 AM.png' />

<img src='./recognize_samples/Screen Shot 2016-08-07 at 12.53.41 AM.png' />

<img src='./recognize_samples/Screen Shot 2016-08-07 at 12.55.45 AM.png' />
 

