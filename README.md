# SDnet 
Code for the paper: 

**A Hierarchical Static-Dynamic Encoder-Decoder Structure for 3D Human Motion Prediction with Residual CNNs**, by Jin Tang, Jin Liu, and Jianqin Yin.


## Dependencies

- python >= 3.5
- tensorflow >= 1.0
 
## Get the data
[Download](https://pan.baidu.com/s/1NcqhaQiOUl9vmeVQDXcs8Q) processed data

key:  ```st5r```
## Training

To start training the network on **G3D** dataset, use the following command:

```python train.py --train_data_paths [Path To Your G3D traning data] --valid_data_paths [Path To Your G3D test data]```

To start training the network on **FNTU** dataset, use the following command:

```python train.py --train_data_paths [Path To Your FNTU traning data] --valid_data_paths [Path To Your FNTU test data] --encoder_length 4 --decoder_length 6```
## Reference
- Code is based on PredCNN: [GitHub](https://github.com/xzr12/PredCNN)
