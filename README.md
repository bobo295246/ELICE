# ELICE
 ELICE is evaluated on four tasks of the CLUE benchmark (Mandarin NLU tasks).
With the basic version, ELICE-base has 12M (million) parameters but achieves
99% of the performance of BERT-base (108M) on average. Furthermore, on IMDb
sentiment classification (English NLU task), ELICE-base (13M) achieves 95% of
the accuracy of XLNet (110M). Additionally, we deploy ELICE in the practical
system for real-time ad-hoc document retrieval regarding the legal compliance in
financial industry, and the fine-tuned ELICE outperforms the pre-trained baseline
and the whitening representations. Those indicate the effectiveness, the robustness
and the flexibility of ELICE in the field of language understanding.

Chinese Language Understanding Evaluation Benchmar
----------------------------------------------------------------------------
##### ELICE performance on CLUE

| MODEL   | #Params   | AFQMC  | TNEWS'  | IFLYTEK'   | OCNLI    |
| :----:| :----: | :----: | :----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>        | 108M |  73.70 | 56.58  | 60.29 | 72.20 |
| <a href="https://github.com/google-research/albert">ALBERT-xxlarge</a>     | 235M   | 75.6 | 59.46 | 62.89 | 77.70 |  
| <a href="https://github.com/google-research/albert">ALBERT-xlarge</a>      | 60M   | 69.96 | 57.36 | 59.50 | - |  
| <a href="https://github.com/google-research/albert">ALBERT-large</a>      | 18M   | 74  | 55.16 | 57.00 | - | 
| <a href="https://github.com/google-research/albert">ALBERT-base</a>       | 12M   | 72.55  | 55.06 | 56.58 | - | 
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a>        | 4M | 69.92 | 53.35 | 48.71 | 65.12 | 
| **ELICE-tiny**   | 6M  | 69.59 | 63.03 | 56.85 | 63.23 | 63.16 | 63.18 |
| **ELICE-base**   | 12M | **71.15** | **64.68** | **58.92** | 67.12 |  64.92 | 65.50 |



## Inference 

| MODEL   |  AFQMC  | TNEWS'  | IFLYTEK' |  OCNLI  |  IMDB   |
| :----:|:-------:|:-------:|:-------:|:--------:|:-------:|
| **ELICE-tiny**   | 7.69ms  | 9.23ms  | 8.56ms  |   7.59   |    -    | 
| **ELICE-base**   | 12.71ms | 11.59ms | 13.61ms |  11.25   | 15.62ms | 
