# XLMRoBERTa

The XLM-RoBERTa model was proposed in Unsupervised Cross-lingual Representation Learning at Scale by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook’s RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

The abstract from the paper is the following:

This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code, data, and models publicly available.

* https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta
* https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr

## IO
Explanation of inputs and outputs as specified in the trainer file:


### Input
- `transcript` (`FreeAnnotation`): The input text to analyze the sentiment from 
  
### Output
The output of the model are three continuous annotations:
- `embeddings` (`SSIStream`): The 768 feature embeddings of the trained model.

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "XLM_RoBERTa",
  "data": json.dumps([
    {"src":"file:annotation:free", "type":"input", "id":"transcript", "uri":"path/to/my/transcript.annotation"},
    {"src":"file:stream:SSIStream", "type":"output", "id":"embeddings",  "uri":"path/to/my/embeddings.stream"}
  ]),
  "trainerFilePath": "modules\\xlm_roberta\\xlm_roberta.trainer",
 "frame_size": "40ms",
 "left_context": "960ms"
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

### License

### CITATION
@article{conneau2019unsupervised,
title={Unsupervised Cross-lingual Representation Learning at Scale},
author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
journal={arXiv preprint arXiv:1911.02116},
year={2019}
}

@article{goyal2021larger,
title={Larger-Scale Transformers for Multilingual Masked Language Modeling},
author={Goyal, Naman and Du, Jingfei and Ott, Myle and Anantharaman, Giri and Conneau, Alexis},
journal={arXiv preprint arXiv:2105.00572},
year={2021}
}