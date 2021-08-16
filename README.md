# SASRec.paddle
A PaddlePaddle implementation of Self-Attentive Sequential Recommendation.

## Introduction

![model](.\images\model.png)

论文：[Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)

## Results

| Datasets     | Metrics | Paper's | Ours   | abs. improv. |
| ------------ | ------- | ------- | ------ | ------------ |
| MovieLens-1m | HIT@10  | 0.8245  | 0.8255 | 0.0010       |
| MovieLens-1m | NDCG@10 | 0.5905  | 0.5947 | 0.0042       |

## Requirement

- Python >= 3
- PaddlePaddle >= 2.0.0
- PaddleNLP >= 2.0.0
- see `requirements.txt`

## Dataset

![result](.\images\dataset.png)

本次实验中，我们采用了原作者处理后的数据集，剔除了活动次数少于 5 的用户记录，清洗后格式后`<user id, item id>`，以`user id`为第一关键字、`time`为第二关键字排序。

## Usage

### Train

1. 下载[数据集](https://raw.githubusercontent.com/kang205/SASRec/master/data/ml-1m.txt)到 `data/preprocessed` 文件夹

```shell
python preprocess.py
```

2. 开始训练

```shell
bash train.sh
```

### Download Trained model

[SASRec model](https://cowtransfer.com/s/013a779f0c7242)

将模型分别放置于 `output` 目录下，如下运行 `eval` bash 脚本即可测试模型。

### Test

```shell
bash eval.sh
```

可以得到如下结果：

![result](.\images\result.png)

模型在 200 epochs 左右收敛，继续训练性能会有小幅提升。

## Details

1. 原文中的`LayerNorm`层为`MultiHeadAttention`和`Point-wise FFN`的前置，实验证明后置模型性能更优，这也与`Transformer`原始架构相符。
2. 原文优化器为`Adam`，使用`AdamW`获得了更好的收敛效果。

## References

```
@misc{kang2018selfattentive,
      title={Self-Attentive Sequential Recommendation}, 
      author={Wang-Cheng Kang and Julian McAuley},
      year={2018},
      eprint={1808.09781},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

* https://github.com/kang205/SASRec
* https://github.com/pmixer/SASRec.pytorch
