# 3DCTN

**3DCTN: 3D Convolution-Transformer Network for Point Cloud Classification**

This is a Pytorch implementation of 3DCTN.

Paper link: https://ieeexplore.ieee.org/document/9861747


**Abtract**

Point cloud classification is a fundamental task in 3D applications. However, it is challenging to achieve effective feature learning due to the irregularity and unordered nature of point
clouds. Lately, 3D Transformers have been adopted to improve point cloud processing. Nevertheless, massive Transformer layers tend to incur huge computational and memory costs. This paper
presented a novel hierarchical framework that incorporated convolutions with Transformers for point cloud classification, named 3D Convolution-Transformer Network (3DCTN). It combined the
strong local feature learning ability of convolutions with the remarkable global context modeling capability of Transformers. Our method had two main modules operating on the downsampling
point sets. Each module consisted of a multi-scale local feature aggregating (LFA) block and a global feature learning (GFL) block, which were implemented by using the Graph
Convolution and Transformer respectively. We also conducted a detailed investigation on a series of self-attention variants to explore better performance for our network. Various experiments
on ModelNet40 and ScanObjectNN datasets demonstrated that our method achieves state-of-the-art classification performance with a lightweight design.


**Architecture**

<img width="766" alt="1" src="https://user-images.githubusercontent.com/92398834/194783420-73776a42-ebb1-488f-b5f8-218578f3aedc.png">



**Heat Map Visualization**

<img width="725" alt="2" src="https://user-images.githubusercontent.com/92398834/194783430-eaed25a9-f1a1-464f-834c-eb810c0a9eaf.png">



**Install**

The latest codes are tested on CUDA10.1, PyTorch 1.6 and Python 3.8:

**Data Preparation**

Download alignment ModelNet (https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in data/modelnet40_normal_resampled/

**Run**

python train_classification.py --use_normals --model pointnet2_cls_msg --log_dir pointnet2_cls_msg_github --learning_rate 0.01 --batch_size 16 --optimizer SGD --epoch 300 --process_data


**Citation**

If it is helpful for your work, please cite this paper:

@ARTICLE{9861747,  
      author={Lu, Dening and Xie, Qian and Gao, Kyle and Xu, Linlin and Li, Jonathan},  
      journal={IEEE Transactions on Intelligent Transportation Systems},   
      title={3DCTN: 3D Convolution-Transformer Network for Point Cloud Classification},   
      year={2022},  
      volume={},  
      number={},  
      pages={1-12},  
      doi={10.1109/TITS.2022.3198836}}
