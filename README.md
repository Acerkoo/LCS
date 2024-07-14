# LCS: A Language Converter Strategy for Zero-Shot Neural Machine Translation

In this repo, we supply the codes and scripts for the implementation of LCS.

## Codes

In this section, we introduce two branches of LCS implementation based on the open-source toolkit fairseq (version 1.0.0). *(Both are placed in `fairseq`, and we only place the modified python files to save space.)*

**The first branch**

We report all scores in our paper with the first branch (**fairseq-converter**) 

In this implementation, we place the source language tag on the encoder side and the target language tag on the decoder side. 

Like this: 

`source:` `<en> Hello, how are you?` 

`target: ` ` <de> Hallo, wie geht’s?`

In order to acquire the target language on the encoder side, we adjust the following python files to get the desired target language.

```shell
fairseq-converter
└── fairseq
    ├── criterions
    │   └── label_smoothed_cross_entropy_le.py
    ├── models
    │   ├── fairseq_encoder.py
    │   └── transformer
    │       ├── transformer_config.py
    │       └── transformer_encoder.py
    ├── sequence_generator.py
    └── tasks
        └── translation_label.py			
```

And the corresponding scripts of three datasets for preparing, training, and testing are placed in ``scripts``.

**The second branch**

To simplify the implementation, we provide the second branch (**fairseq-LCS**)

In this implementation, we place the extra target language tag on the encoder side, which is treated as the padding token during the calculation.

Like this: 

`source:` `<de> <en> Hello, how are you?` 

`target: ` ` <de> Hallo, wie geht’s?`

Compared to the first branch, this one only has two modified python files as the following:

```shell
fairseq-LCS
└── fairseq
    └── models
        └── transformer
            ├── transformer_config.py
            └── transformer_encoder.py
```

And we also provide the corresponding training and test scripts in `scripts` with the `_2` suffix.

**Difference**

We examine the difference between both implementations in OPUS-100 dataset.

In the fair comparison setting, we list the scores of both implementations.

|      Implementation       | Supervised | Zero-Shot | Accuracy |
| :-----------------------: | :--------: | :-------: | :------: |
| fairseq-converter (first) |   24.80    |   15.22   |  85.35   |
|   fairseq-LCS (second)    |   24.63    |   15.19   |  85.39   |

_**Supervised** and **Zero-shot** represent the average scareBLEU (%), and **Accuracy** represents the language accuracy on the zero-shot translation. The above two models are trained with setting k to 2._

The above table shows that the first branch yields a little advantage and the difference is negligible.