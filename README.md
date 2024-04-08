# NER-CLUE

## Introduction

A Pytorch implementation of using BERT("bert-base-chinese" on Huggingface) on NER(Named Entity Recognition) task, [CLUE](https://www.cluebenchmarks.com/introduce.html) dataset.

CLUE is a Chinese NER dataset, with span-based label. For example, 

```
{"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
```

Our method conducts character-level tokenization and re-labels every character based on the original span-based labels, turning the problem into a character-level token classification task.

We use [`bert-base-chinese`](https://huggingface.co/google-bert/bert-base-chinese) model on Hugging Face.

We have a post-process method to transform the prediction of the model, into the original span-based label.

## How to Use

First you need to create a `checkpoints` directory on your own, for saving trained model checkpoints.

And you need to pip install the necessary libraries like transformers, scikit-learn, etc.

Then you can run `train.py`, `evaluate.py` and `infer.py` to get the results.
 