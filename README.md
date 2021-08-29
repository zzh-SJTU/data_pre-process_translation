# data_pre-process_translation
datapre.py为数据集加载并分batch，构造词典
transla.py为数据集应用于tranformer翻译任务的demo
环境：torchtext 0.10.0
     spacy 3.1.1
     torch 1.9.0
注意：由于IWSLT2016数据集默认使用谷歌网盘下载，国内无法直接调用，需要手动下载后更改IWSLT2016源代码(替换url下载)。
新增ppl和bleu计算
