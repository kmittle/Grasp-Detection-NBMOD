This is the repository for the **NBMOD** (Noisy Background Multi-Object Dataset for grasp detection) and the code of the paper ***NBMOD: Find It and Grasp It in Noisy Background***.


# Introduction
We propose a dataset called **NBMOD** (Noisy Background Multi-Object Dataset for grasp detection) consisting of **31,500 RGB-D images**, which is composed of three subsets: Simple background Single-object Subset (**SSS**), Noisy background Single-object Subset (**NSS**), and Multi-Object grasp detection Subset (**MOS**). Among them, the SSS subset consists of 13,500 images, the NSS subset consists of 13,000 images, and the MOS subset consists of 5,000 images.

Unlike the renowned Cornell dataset, the NBMOD dataset differs in that its backgrounds are no longer simple whiteboards. The NSS and MOS subsets comprise a substantial number of images with noise, where this noise corresponds to interfering objects unrelated to the target objects for grasping detection. Moreover, in the MOS subset, each image encompasses multiple target objects for grasp detection, which closely resembles real-world working environments.

To enhance the task of grasp detection, we propose a novel mechanism called **RAM** (Rotation Anchor Mechanism) and design three detection network architectures: **RARA** (network with Rotation Anchor and Region Attention), **RAST** (network with Rotation Anchor and Semi Transformer), and **RAGT** (network with Rotation Anchor and Global Transformer). These architectures aim to improve the accuracy and robustness of grasp detection by incorporating rotation anchor-based methods and attention mechanisms.

Some samples of NBMOD are shown in the following figure:

![image](picture/dataset.png) 

Annotations of some samples in NBMOD are illustrated in the following figure as examples:

![image](picture/annotation.png) 


# Model Architectures
The architectures of the RAST, RARA, and RAGT models are depicted in the following figures:

**RAST:**

![image](picture/RAST.png) 

**RARA:**

![image](picture/RARA.png) 

**RAGT:**

![image](picture/RAGT.png) 


# Detection Results
Detection results on SSS and NSS:

![image](picture/detected-single-obj.png) 

Detection results on MOS:

![image](picture/detected-multi-obj.png) 


# Requirements
Our experimental setup is as follows:

    python                    3.9.7
    torch.version.cuda        11.3
    torch                     1.12.1+cu113
    torchaudio                0.12.1+cpu
    torchdata                 0.6.0             
    torchinfo                 1.7.2             
    torchstat                 0.0.7                 
    torchsummary              1.5.1                  
    torchvision               0.13.1+cu113         
    torchviz                  0.0.2                  
    tornado                   6.2             
    tqdm                      4.65.0         
    thop                      0.1.1-2209072238       
    tensorboard               2.9.1                 
    tensorboard-data-server   0.6.1                 
    tensorboard-plugin-wit    1.8.1              
    tensorboardx              2.5.1                
    opencv-contrib-python     4.7.0.72          
    opencv-python             4.7.0.72   

    CUDA Version              11.2


# Download NBMOD and Model Weights
Currently, the open-source code available is for the RAGT-3/3 model. You can modify the variable 'anchor' in the 'Anchor.py' file to change the number of anchors in each grid cell. The code for the RAST and RAGT models will be uploaded soon.

The NBMOD is available at [NBMOD](https://pan.baidu.com/s/1kHtTKYkqFciJpfiMkEENaQ), with the password for extraction being 6666.

The weights of models are available at [Weights](https://pan.baidu.com/s/18tAB5Yuu0yAJiyQvjE2vJw). The password for extraction is 6666.


# Citation
You can find a paper for explaining the NBMOD and our models on [arXiv](https://arxiv.org/abs/2306.10265).

If you use this library or find the documentation useful for your research, please consider citing:

    @article{cao2023nbmod,
      title={NBMOD: Find It and Grasp It in Noisy Background},
      author={Cao, Boyuan and Zhou, Xinyu and Guo, Congmin and Zhang, Baohua and Liu, Yuchen and Tan, Qianqiu},
      journal={arXiv preprint arXiv:2306.10265},
      year={2023}
    }
