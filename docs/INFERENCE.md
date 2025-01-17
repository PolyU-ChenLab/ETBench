# Inference E.T. Chat on Charades-STA

The performance of E.T. Chat on full Charades-STA test set is as follows.

| Model | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|:-:|:-:|:-:|:-:|:-:|
| E.T. Chat | 65.7 | 45.9 | 20.0 | 42.3 |

## 📦 Data Preparation

Download [Charades-STA](https://github.com/jiyanggao/TALL) dataset and organize it as follows.

```
ETBench
└─ data
   └─ charades
      ├─ videos
      │  ├─ 0A8CF.mp4
      │  ├─ 0A8ZT.mp4
      │  └─ ...
      ├─ charades_sta_train.txt
      └─ charades_sta_test.txt
```

## 🚀 Run Inference

Run the following command to inference E.T. Chat on Charades-STA dataset.

```shell
python etchat/eval/infer_charades.py \
    --anno_path data/charades/charades_sta_test.txt \
    --data_path data/charades/videos \
    --pred_path charades_sta \
    --model_path <path-to-model>
```

The model outputs would be saved in the `charades_sta` folder.

## 🔮 Compute Metrics

Run the following command to compute metrics.

```shell
python etchat/eval/eval_grounding.py charades_sta
```
