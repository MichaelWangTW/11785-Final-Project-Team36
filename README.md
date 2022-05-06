# 11785 Intro to Deep Learning
### Transformer On Small Data Set
#### Team36: JiaChen Xu, Chen Wu, Shihong Liu, LiangChun Wang

This is the repository of final project for 11-785 Intro to Deep Learning at CMU Spring 2022

The structure of this folder looks like this:

```python

-configs
    -cifar10_simple/general.yaml
    -cifar100_simple/general.yaml
    -flowers102_simple.yaml
    -svhn_simple/genral.yaml
-models
    -_init_.py
    -alternet.py
    -alternet_deformable.py
    -attention.py
    -embeddings.py
    -ensemble.py
    -resnet.py
    -vit.py
-ops
    -_init_.py
    -ACClip.py
    -cifarc.py
    -cifarp.py
    -datasets.py
    -norm.py
    -schedulers.py
    -test.py
    -train.py
-(model folder defined by user)
- classification.ipynb
- robustness.ipynb
-README.md
```

Our code implementation is heavily dependent on the source code released by original authors of the paper. The links could be found as follows:

+ how-do-vits-work: (https://github.com/xxxnell/how-do-vits-work)

Below is the small dataset we use for this project

- CIFAR 10
- CIFAR 100
- SVHN
--- 

### Model Description
For this project, we run four different models on three small dataset.
1. Vit
2. AlterNet
3. AlterNet with Deformable Attention 
4. Resnet

### Data Preprocessing

We used different ways to preprocess the small datasets we have. We use yaml file for writing configuration files. To customized the dataset such as custom data augmentation, you can just modify the dataset.py.
The main difference between simple and general yaml file is that simple.yaml use SGD as optimizer while general.yaml use ACClip as optimizer. 

### Caution
We set up our develop environment on google colab. To run our repository, make sure to set up environment that similar to colab. 

### How to run

Train process
1. Create a folder where you can save the model and specify the folder name in classification.ipynb

2. Select the yaml file under configs folder. Modify the yaml file to fit your setting. Specify the yaml file path in classification.ipynb

3. Select the model described above and specify the model name in classification.ipynb

4. Execute all cells in classification.ipynb 

Test process

1. Execute the last block in classification.ipynb


Note that there are multiple hyper parameter you can change under the yaml file. Our code will automatically parse the aurguments in yaml file and use it as mdoel parameter. 

Below is the example of yaml file. 

A typical workflow would be like this:

+  You decide on some hyperparamters such as epoch, warm up epoch, batch size, optimizer, scheduler and modified it in the yaml file. 
+  Decide the image size, patch size and window size for the dataset in order to do vision transformer. You can set this in a varaiable called `vit_kwargs` under `classification.ipynb`
+  Do training from beginning to end
+  Currently, our code automatically save the model each 10 epochs
+  Load the model and do the testing

The way how this works is that, you will upload this whole folder to your google drive, you in colab mount your google drive, mount the folder and run classification.ipynb. 
A few things to notice in pipeline workflow:

1. The two files that you might frequently need to change is the `{dataset_name}_simple/general.yaml` and `classification.ipynb`

2. `classification.ipynb` automatically download the dataset and do data augmentations. 

3. go to `classification.ipynb`, find the varaible called `config_path` and set the yaml file path you just modified in step 1.

4. go to `classification.ipynb`, find the varaible called `model` and set the model name you want to execute.

5. To do transformers, you can specify the image size, path size and window size for transformer parameters.  

6. You should probably now can do training. Run the command given above

7. After finish training, the model should be stored under the folder you specify.

8. You should probably now can do testing. Run the last block in `classification.ipynb`.


