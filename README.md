# SpanEmo
***

Source code for the paper ["SpanEmo: Casting Multi-label Emotion Classification as Span-prediction"]() in EACL2021.


# Dependencies
We use Python=3.6, torch=1.2.0, cudatoolkit=10.0. Other packages can be installed via:
```angular2html
python install -r requirements.txt
```
The model was trained on an Nvidia GeForce GTX1080 with 11GB memory, Ubuntu 18.10.

***

# Usage

You first need to download the dataset [Link](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) for the language of your choice  (i.e., English, Arabic or Spanish) and then place them in the data directory.


Then run the main script to do the followings:
 * data loading and preprocessing
 * model creation and training

### Training
```
python scripts/train.py --train-path {} --dev-path {}

options:
    -h, --help                show help message and exit
    --config-file             json file name to save chosen arguemnts
    --loss-type=<str>         Which loss to use cross-ent|corr|joint 
    --max-length=<int>        text length
    --output-dropout=<float>  prob of dropout applied to the output layer
    --checkpoint-path=<str>   path in which the trained model should be saved
    --seed=<int>              fixed random seed number 
    --train-batch-size=<int>  batch size 
    --eval-batch-size=<int>   batch size 
    --max-epoch=<int>         max number of epochs
    --ff-lr=<float>              ffn learning rate 
    --bert-lr=<float>         bert learning rate
    --lang=<str>              language, English|Arabic|Spanish
    --dev-path=<str>          path in which the valid set is saved
    --train-path=<str>        path in which the train set is saved
    --alpha-loss=<float>      alpha value
```



Once the above step is done, you can then evaluate on the test set using the trained model:

## Evaluation
```
python scripts/test.py 

Options:
    -h --help                  show help message and exit
    --model-path=<str>         path in which the trained model is saved
    --loss-type=<str>          Which loss to use cross-ent|corr|joint.
    --max-length=<int>         text length
    --output-dropout=<float>   prob of dropout applied to output layer [default: 0.1]
    --seed=<int>               fixed random seed number
    --test-batch-size=<int>    test set batch size [default: 32]
    --lang=<str>               language, English|Arabic|Spanish
    --test-path=<str>          path in which the test set is saved
```
***

# Citation
```
@inproceedings{AlhuzaliSpanEmo,  
title = "SpanEmo: Casting Multi-label Emotion Classification as Span-prediction",  
author = "Alhuzali, Hassan and Ananiadou, Sophia",  
booktitle = "Proceedings of the 16th conference of the European Chapter of the Association for Computational Linguistics (EACL)",  
year = "2021",  
publisher = "Association for Computational Linguistics",  
pages = ""  
} 
```
