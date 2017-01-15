The Nature Conservancy Fisheries Monitoring challenge
=====================================================

Simple classification solution for the The Nature Conservancy Fisheries Monitoring challenge (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring).

Dataset structure
-----------------

Create a parent folder for the dataset. The "-dataset" argument of main.py need to point to this folder. Download test_stg1.zip and train.zip from https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data, and extract it to the folder you created. Create a subfolder named bbox. Go to this forum topic https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25902/complete-bounding-box-annotation?forumMessageId=147220 and download annotation json files. Note: the other_labels.json is in 8th post, the others are in the 1st. Now you are ready to train.

Directory structure should look like this:

```
Dataset
│
├── bbox
│    ├─  alb_labels.json
│    ├─  bet_labels.json
│    ├─  ...
│    └─  yft_labels.json
│
├── test_stg1
│    ├─  img_00005.jpg
│    └─  ...
│
└─── train
     ├─  ALB
     ├─  BET
     └─  ...
```

Training
--------

./main.py -dataset 'your dataset folder' -name 'name of the current train version'

A directory with the given name will be created. All training related informations will be saved here.

Debugging in TensorBoard
------------------------

tensorboard --logdir='name of training'/log

Results
-------

With this simple classification approach, you can reach log loss of about 1. For significantly better scores, different methods are needed (like detection).