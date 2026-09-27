---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: proto=tcp; service=-; state=CON; dur=0.27674; spkts=6; dpkts=2; sbytes=998;
    dbytes=86; rate=25.294499; sttl=62; dttl=252; sload=24051.45508; dload=1243.043945;
    sloss=2; dloss=1; sinpkt=55.348; dinpkt=0.009; sjit=2795.303507; djit=0.0; tcprtt=0.158295;
    synack=0.101948; ackdat=0.056347; smean=166; dmean=43
- text: proto=iatp; service=-; state=INT; dur=1e-05; spkts=2; dpkts=0; sbytes=200;
    dbytes=0; rate=142857.1409; sttl=254; dttl=0; sload=114285712.0; dload=0.0; sloss=0;
    dloss=0; sinpkt=0.007; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0;
    ackdat=0.0; smean=100; dmean=0
- text: proto=ddx; service=-; state=INT; dur=5e-06; spkts=2; dpkts=0; sbytes=200;
    dbytes=0; rate=200000.0051; sttl=254; dttl=0; sload=160000000.0; dload=0.0; sloss=0;
    dloss=0; sinpkt=0.005; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0;
    ackdat=0.0; smean=100; dmean=0
- text: proto=tcp; service=http; state=FIN; dur=0.187711; spkts=10; dpkts=6; sbytes=954;
    dbytes=268; rate=79.910074; sttl=254; dttl=252; sload=36609.46875; dload=9546.589844;
    sloss=2; dloss=1; sinpkt=19.279111; dinpkt=36.254199; sjit=1001.610756; djit=59.602035;
    tcprtt=0.051246; synack=0.006436; ackdat=0.04481; smean=95; dmean=45
- text: proto=smp; service=-; state=INT; dur=5e-06; spkts=2; dpkts=0; sbytes=200;
    dbytes=0; rate=200000.0051; sttl=254; dttl=0; sload=160000000.0; dload=0.0; sloss=0;
    dloss=0; sinpkt=0.005; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0;
    ackdat=0.0; smean=100; dmean=0
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 10 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label          | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:---------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Analysis       | <ul><li>'proto=tcp; service=http; state=FIN; dur=0.814095; spkts=10; dpkts=8; sbytes=794; dbytes=1178; rate=20.882083; sttl=62; dttl=252; sload=7026.207031; dload=10131.49512; sloss=2; dloss=2; sinpkt=90.362667; dinpkt=95.126289; sjit=4922.148396; djit=149.538375; tcprtt=0.310128; synack=0.148208; ackdat=0.16192; smean=79; dmean=147'</li><li>'proto=unas; service=-; state=INT; dur=8e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=125000.0003; sttl=254; dttl=0; sload=100000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.008; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li><li>'proto=pipe; service=-; state=INT; dur=9e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=111111.1072; sttl=254; dttl=0; sload=88888888.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.009; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li></ul>                                                                                                                     |
| Backdoor       | <ul><li>'proto=merit-inp; service=-; state=INT; dur=1e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=1000000.003; sttl=254; dttl=0; sload=800000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.001; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li><li>'proto=ospf; service=-; state=INT; dur=59.30122; spkts=196; dpkts=0; sbytes=53312; dbytes=0; rate=3.288297; sttl=254; dttl=0; sload=7155.333496; dload=0.0; sloss=0; dloss=0; sinpkt=304.108656; dinpkt=0.0; sjit=357.462031; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=272; dmean=0'</li><li>'proto=ospf; service=-; state=REQ; dur=59.21376; spkts=96; dpkts=0; sbytes=26112; dbytes=0; rate=1.604357; sttl=254; dttl=0; sload=3491.080322; dload=0.0; sloss=0; dloss=0; sinpkt=646.310625; dinpkt=0.0; sjit=834.247875; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=272; dmean=0'</li></ul>                                                                                                                                        |
| DoS            | <ul><li>'proto=ospf; service=-; state=REQ; dur=57.889004; spkts=60; dpkts=0; sbytes=16320; dbytes=0; rate=1.019192; sttl=254; dttl=0; sload=2217.761475; dload=0.0; sloss=0; dloss=0; sinpkt=1010.394563; dinpkt=0.0; sjit=1226.7735; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=272; dmean=0'</li><li>'proto=skip; service=-; state=INT; dur=3e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=333333.3215; sttl=254; dttl=0; sload=266666656.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.003; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li><li>'proto=unas; service=-; state=INT; dur=3e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=333333.3215; sttl=254; dttl=0; sload=266666656.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.003; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li></ul>                                                                                                                                                            |
| Exploits       | <ul><li>'proto=tcp; service=http; state=FIN; dur=1.152587; spkts=10; dpkts=8; sbytes=836; dbytes=354; rate=14.749428; sttl=254; dttl=252; sload=5226.503418; dload=2151.681396; sloss=2; dloss=1; sinpkt=121.340889; dinpkt=149.442422; sjit=7034.639274; djit=246.655875; tcprtt=0.200764; synack=0.106489; ackdat=0.094275; smean=84; dmean=44'</li><li>'proto=mtp; service=-; state=INT; dur=5e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=200000.0051; sttl=254; dttl=0; sload=160000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.005; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li><li>'proto=tcp; service=http; state=FIN; dur=0.328956; spkts=10; dpkts=8; sbytes=810; dbytes=1242; rate=51.678643; sttl=62; dttl=252; sload=17728.81445; dload=26435.14648; sloss=2; dloss=2; sinpkt=36.550667; dinpkt=40.184; sjit=1870.385191; djit=62.322527; tcprtt=0.107786; synack=0.04701; ackdat=0.060776; smean=81; dmean=155'</li></ul>                                                               |
| Fuzzers        | <ul><li>'proto=tcp; service=-; state=FIN; dur=1.052039; spkts=10; dpkts=6; sbytes=756; dbytes=268; rate=14.258026; sttl=254; dttl=252; sload=5178.515137; dload=1703.358887; sloss=2; dloss=1; sinpkt=111.216667; dinpkt=198.173594; sjit=7785.599522; djit=352.276437; tcprtt=0.119062; synack=0.061164; ackdat=0.057898; smean=76; dmean=45'</li><li>'proto=tcp; service=-; state=FIN; dur=0.39491; spkts=10; dpkts=6; sbytes=596; dbytes=268; rate=37.983337; sttl=254; dttl=252; sload=10878.42773; dload=4537.742676; sloss=2; dloss=1; sinpkt=43.878889; dinpkt=59.177199; sjit=2217.205868; djit=107.339852; tcprtt=0.14403; synack=0.075559; ackdat=0.068471; smean=60; dmean=45'</li><li>'proto=tcp; service=-; state=FIN; dur=0.516711; spkts=10; dpkts=6; sbytes=586; dbytes=268; rate=29.029767; sttl=254; dttl=252; sload=8174.782715; dload=3468.0896; sloss=2; dloss=1; sinpkt=54.670667; dinpkt=83.837797; sjit=2704.303944; djit=118.291781; tcprtt=0.150502; synack=0.097513; ackdat=0.052989; smean=59; dmean=45'</li></ul>               |
| Generic        | <ul><li>'proto=udp; service=dns; state=INT; dur=9e-06; spkts=2; dpkts=0; sbytes=114; dbytes=0; rate=111111.1072; sttl=254; dttl=0; sload=50666664.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.009; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=57; dmean=0'</li><li>'proto=udp; service=dns; state=INT; dur=9e-06; spkts=2; dpkts=0; sbytes=114; dbytes=0; rate=111111.1072; sttl=254; dttl=0; sload=50666664.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.009; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=57; dmean=0'</li><li>'proto=udp; service=dns; state=INT; dur=3e-06; spkts=2; dpkts=0; sbytes=114; dbytes=0; rate=333333.3215; sttl=254; dttl=0; sload=152000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.003; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=57; dmean=0'</li></ul>                                                                                                                                                                              |
| Normal         | <ul><li>'proto=tcp; service=http; state=FIN; dur=0.661656; spkts=10; dpkts=10; sbytes=804; dbytes=1330; rate=28.715827; sttl=62; dttl=252; sload=8753.792969; dload=14472.77734; sloss=2; dloss=2; sinpkt=73.517333; dinpkt=64.571555; sjit=4564.721941; djit=110.757008; tcprtt=0.109361; synack=0.05091; ackdat=0.058451; smean=80; dmean=133'</li><li>'proto=tcp; service=http; state=FIN; dur=1.501572; spkts=60; dpkts=16; sbytes=68195; dbytes=698; rate=49.947654; sttl=254; dttl=252; sload=357273.5625; dload=3489.676025; sloss=27; dloss=1; sinpkt=25.052339; dinpkt=102.143141; sjit=2413.959242; djit=128.161133; tcprtt=0.201299; synack=0.071565; ackdat=0.129734; smean=1137; dmean=44'</li><li>'proto=tcp; service=-; state=FIN; dur=0.022939; spkts=40; dpkts=42; sbytes=2542; dbytes=23508; rate=3531.10419; sttl=31; dttl=29; sload=864553.8125; dload=8003487.5; sloss=7; dloss=14; sinpkt=0.579667; dinpkt=0.547829; sjit=37.850544; djit=36.773212; tcprtt=0.000601; synack=0.000475; ackdat=0.000126; smean=64; dmean=560'</li></ul> |
| Reconnaissance | <ul><li>'proto=udp; service=-; state=INT; dur=4e-06; spkts=2; dpkts=0; sbytes=168; dbytes=0; rate=250000.0006; sttl=254; dttl=0; sload=168000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.004; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=84; dmean=0'</li><li>'proto=ddx; service=-; state=INT; dur=9e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=111111.1072; sttl=254; dttl=0; sload=88888888.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.009; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0'</li><li>'proto=udp; service=-; state=INT; dur=9e-06; spkts=2; dpkts=0; sbytes=168; dbytes=0; rate=111111.1072; sttl=254; dttl=0; sload=74666664.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.009; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=84; dmean=0'</li></ul>                                                                                                                                                                                   |
| Shellcode      | <ul><li>'proto=tcp; service=-; state=FIN; dur=1.768067; spkts=10; dpkts=8; sbytes=1132; dbytes=354; rate=9.61502; sttl=254; dttl=252; sload=4610.685059; dload=1402.661743; sloss=2; dloss=1; sinpkt=193.740333; dinpkt=229.562578; sjit=12256.41399; djit=505.692031; tcprtt=0.25337; synack=0.161126; ackdat=0.092244; smean=113; dmean=44'</li><li>'proto=tcp; service=-; state=FIN; dur=0.637217; spkts=10; dpkts=8; sbytes=562; dbytes=354; rate=26.67851; sttl=254; dttl=252; sload=6352.624023; dload=3891.923828; sloss=2; dloss=1; sinpkt=70.685111; dinpkt=83.131141; sjit=4522.097925; djit=136.937391; tcprtt=0.126879; synack=0.055291; ackdat=0.071588; smean=56; dmean=44'</li><li>'proto=tcp; service=-; state=FIN; dur=0.434038; spkts=10; dpkts=8; sbytes=700; dbytes=354; rate=39.167076; sttl=254; dttl=252; sload=11611.88574; dload=5713.785156; sloss=2; dloss=1; sinpkt=46.441222; dinpkt=54.54543; sjit=3244.676654; djit=111.323633; tcprtt=0.105041; synack=0.052212; ackdat=0.052829; smean=70; dmean=44'</li></ul>              |
| Worms          | <ul><li>'proto=tcp; service=http; state=FIN; dur=5.273311; spkts=66; dpkts=362; sbytes=3216; dbytes=481994; rate=80.973792; sttl=254; dttl=252; sload=4806.088379; dload=729201.0625; sloss=2; dloss=179; sinpkt=81.112923; dinpkt=14.410776; sjit=3969.577022; djit=1746.923307; tcprtt=0.159584; synack=0.071012; ackdat=0.088572; smean=49; dmean=1331'</li><li>'proto=tcp; service=http; state=FIN; dur=1.68251; spkts=10; dpkts=6; sbytes=1296; dbytes=268; rate=8.915252; sttl=254; dttl=252; sload=5548.852539; dload=1065.075439; sloss=2; dloss=1; sinpkt=181.198556; dinpkt=320.416812; sjit=12532.37074; djit=594.37325; tcprtt=0.164179; synack=0.080419; ackdat=0.08376; smean=130; dmean=45'</li><li>'proto=tcp; service=http; state=FIN; dur=0.223749; spkts=10; dpkts=6; sbytes=1308; dbytes=268; rate=67.039407; sttl=254; dttl=252; sload=42118.625; dload=8008.974609; sloss=2; dloss=1; sinpkt=23.364; dinpkt=42.921; sjit=1447.381706; djit=70.750062; tcprtt=0.015213; synack=0.00914; ackdat=0.006073; smean=131; dmean=45'</li></ul> |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("proto=ddx; service=-; state=INT; dur=5e-06; spkts=2; dpkts=0; sbytes=200; dbytes=0; rate=200000.0051; sttl=254; dttl=0; sload=160000000.0; dload=0.0; sloss=0; dloss=0; sinpkt=0.005; dinpkt=0.0; sjit=0.0; djit=0.0; tcprtt=0.0; synack=0.0; ackdat=0.0; smean=100; dmean=0")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 24  | 24.0   | 24  |

| Label          | Training Sample Count |
|:---------------|:----------------------|
| Analysis       | 5                     |
| Backdoor       | 5                     |
| DoS            | 5                     |
| Exploits       | 5                     |
| Fuzzers        | 5                     |
| Generic        | 5                     |
| Normal         | 5                     |
| Reconnaissance | 5                     |
| Shellcode      | 5                     |
| Worms          | 5                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0071 | 1    | 0.1059        | -               |
| 0.3546 | 50   | 0.2299        | -               |
| 0.7092 | 100  | 0.1695        | -               |

### Framework Versions
- Python: 3.14.7
- SetFit: 1.2.0
- Sentence Transformers: 6.0.1
- Transformers: 5.16.1
- PyTorch: 2.14.0+cu130
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->