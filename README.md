## PyramidBox: A Context-assisted Single Shot Face Detector.##
[A PyTorch Implementation of Dual Shot Face Detector](https://arxiv.org/abs/1803.07737)


### Description
I train pyramidbox with pytorch and the result approaches the original paper result,the pretrained model can be downloaded in [vgg](https://pan.baidu.com/s/1Q-YqoxJyqvln6KTcIck1tQ),the final model can be downloaded in [Pyramidbox](https://pan.baidu.com/s/1VtzgB9srkJY4SUtVM3n8tw).the AP in WIDER FACE as following:

|             | Easy MAP | Medium MAP |  hard MAP |
| --------    | ---------|------------| --------- |
| origin paper|	0.960    |    0.948   |  0.888    |
| this repo   | 0.948    |    0.938   |  0.880    |

the AP in AFW,PASCAL,FDDB as following:

| 	AFW     |   PASCAL	|   FDDB   |
| --------- |-----------| ---------|
|	99.65   |    99.02  |  0.983   |

the gap is small with origin paper,I train 120k batch_size 4 which is different from paper,which maybe cause the gap,if you have more gpu ,the final result maybe better.

### Requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### Prepare data 
1. download WIDER face dataset
2. modify data/config.py 
3. ``` python prepare_wider_data.py```


### Train 
``` 
python train.py --batch_size 4  
		--lr 5e-4
```

### Evalution
according to yourself dataset path,modify data/config.py 
1. Evaluate on AFW.
```
python tools/afw_test.py
```
2. Evaluate on FDDB 
```
python tools/fddb_test.py
```
3. Evaluate on PASCAL  face 
``` 
python tools/pascal_test.py
```
4. test on WIDER FACE 
```
python tools/wider_test.py
```
### Demo 
you can test yourself image
```
python demo.py
```

### Result
<div align="center">
<img src="https://github.com/yxlijun/Pyramidbox.pytorch/blob/master/tmp/gsmarena_001.jpg" height="300px" alt="demo" >
<img src="https://github.com/yxlijun/Pyramidbox.pytorch/blob/master/tmp/0_Parade_marchingband_1_488.jpg" height="300px" alt="demo" >
</div>

### References
* [PyramidBox: A Context-assisted Single Shot Face Detector](https://arxiv.org/abs/1803.07737)
* [PyramidBox model](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/face_detection)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
