# conda


Conda ENV :

Environnment : Ubuntu 16.0
               GCC > 5.0

3 pythons :   Python 3.7  AND python 3.6  AND python 3.5


A)  conda_list.txt  with HARD LINKS
  Python 3.7  AND python 3.6  AND python 3.5
  MKL 2019 or 2018
  TensorFlow==1.13.1  CPU with enabled AVX Wheel
  pytorch >=1.0   CPU

  torchvision     CPU
  Keras

  Others packages :
    xgboost  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel pandas
    matplotlib seaborn
    https://github.com/facebook/Ax
    https://github.com/pytorch/botorch
    pip install arrow==0.10.0 attrdict==2.0.0  kmodes==0.9 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0


B) conda_list.txt  with HARD LINKS
  Python 3.7  AND python 3.6  AND python 3.5
  MKL 2019 or 2018
  TensorFlow >=2.0  GPU with AVX enabled Wheel
  pytorch >=1.0   CPU

  torchvision     CPU
  Keras

  Others packages :
    xgboost  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel pandas
    matplotlib seaborn
    https://github.com/facebook/Ax
    https://github.com/pytorch/botorch
    pip install arrow==0.10.0 attrdict==2.0.0  kmodes==0.9 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0


C) conda_list.txt  with HARD LINKS
  Python 3.7  AND python 3.6  AND python 3.5
  MKL 2019 or 2018
  TensorFlow >=1.9  GPU with AVX enabled Wheel
  pytorch >=1.0     GPU

  torchvision       GPU
  Keras

  Others packages :
    xgboost  lightgbm catboost pytorch scikit-learn  chainer  dask  ipykernel pandas
    cudatoolkit
    botorch
    https://github.com/facebook/Ax
    matplotlib seaborn
    pip install arrow==0.10.0 attrdict==2.0.0  kmodes==0.9 tables==3.3.0 tabulate==0.8.2 uritemplate==3.0.0






Wheel for AVX
https://github.com/inoryy/tensorflow-optimized-wheels/releases/download
https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/



















