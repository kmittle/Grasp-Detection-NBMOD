This is the repository for the **NBMOD** (Noisy Background Multi-Object Dataset for grasp detection) and the code of the paper [***NBMOD: Find It and Grasp It in Noisy Background***](https://arxiv.org/abs/2306.10265).


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
Currently, the open-source code available is for the RAGT-3/3 model. You can modify the variable 'num_anchors' in the 'Anchor.py' file to change the number of anchors in each grid cell. The code for the RAST and RAGT models will be uploaded soon.

The NBMOD is available at [NBMOD](https://pan.baidu.com/s/1kHtTKYkqFciJpfiMkEENaQ), with the password for extraction being 6666.

The weights of models are available at [Weights](https://pan.baidu.com/s/18tAB5Yuu0yAJiyQvjE2vJw). The password for extraction is 6666.


# Training & Testing
The images in NBMOD have a resolution of 640x480, and the label files are in XML format.

If you want to utilize our code, you can refer to our training and testing process as follows:

1) The 640x480 images are padded with zeros to a size of 640x640 and then resized to 416x416. You can use the `resize_to_416.py` file in the `\data_preprocess\` directory to complete this step.

2) Use the `original_img.py` file in the `\data_preprocess\data_augmentation\label\` directory to parse the coordinates of oriented bounding boxes from XML files into TXT files. In the TXT file, the coordinates are in the format of nx5, where "n" represents the number of annotation bounding boxes contained in the image, and "5" denotes the five coordinate parameters of the five-dimensional grasp representation.

    For example:

        x1 y1 w1 h1 theta1
        x2 y2 w2 h2 theta2
        x3 y3 w3 h3 theta3
        ......
        xn yn wn hn thetan

   During the experiment, to accelerate the training process, we employed a strategy of performing data augmentation before training rather than augmenting the data during the training process. Under the `\data_preprocess\data_augmentation` directory, there are additional code files for data augmentation, which perform transformations on both images and coordinates. If you need to perform data augmentation, you can refer to these Python programs.

   The purpose of the `rearrangement.py` file in the `\data_preprocess\` directory is to renumber the augmented data after the augmentation process.

3) To start the training process, you can modify the paths of the dataset's images and labels in the `train_grasp.py` file. Additionally, you can set your desired batch size, loss function weights, and the number of training epochs. Once these modifications are done, you can run the `train_grasp.py` file to begin the training.

    In the experiment, we employed the AdamW optimizer with its default training parameters. You can modify the `train_grasp.py` file to implement a more fine-grained training strategy according to your requirements.

4) The `evaluate_acc.py` file is used to test the accuracy of the model. After modifying the paths of the test dataset's images and labels, as well as the model's weight path, you can run the file to evaluate the accuracy of the model.

    Please ensure that the test dataset's image and label data formats remain consistent with those of the training dataset.

   The `draw_single_box.py` script is used to visualize the detection results by drawing only the bounding box with the highest confidence. On the other hand, `grasp_detect_multibox.py` can be used to visualize all prediction boxes with confidence scores greater than a specified threshold.


# Citation
You can find a paper for explaining the NBMOD and our models on [arXiv](https://arxiv.org/abs/2306.10265).

If you use this library or find the documentation useful for your research, please consider citing:

    @article{cao2023nbmod,
      title={NBMOD: Find It and Grasp It in Noisy Background},
      author={Cao, Boyuan and Zhou, Xinyu and Guo, Congmin and Zhang, Baohua and Liu, Yuchen and Tan, Qianqiu},
      journal={arXiv preprint arXiv:2306.10265},
      year={2023}
    }
