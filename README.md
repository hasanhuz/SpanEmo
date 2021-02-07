# SpanEmo
***

Source code for the paper ["SpanEmo: Casting Multi-label Emotion Classification as Span-prediction"](https://arxiv.org/pdf/2101.10038.pdf) in EACL2021.

<p align="center">
  <img src="https://github.com/hasanhuz/SpanEmo/blob/master/SpanEmo_arch.PNG?raw=true" alt="Photo" border="5"/> 
</p>


# Dependencies
We use Python=3.6, torch=1.2.0. Other packages can be installed via:
```angular2html
python install -r requirements.txt
```
The model was trained on an Nvidia GeForce GTX1080 with 11GB memory, Ubuntu 18.10.

***

# Usage

You first need to download the dataset [Link](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) for the language of your choice  (i.e., English, Arabic or Spanish) and then place them in the data directory `data/`.


Next, run the main script to do the followings:
 * data loading and preprocessing
 * model creation and training

### Training
```
python scripts/train.py --train-path {} --dev-path {}

Options:
    -h --help                         show this screen
    --loss-type=<str>                 which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 0]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --lang=<str>                      language choice [default: English]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
```



Once the above step is done, you can then evaluate on the test set using the trained model:

## Evaluation
```
python scripts/test.py --test-path {} --model-path {}

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
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
