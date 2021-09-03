# data_pre-process_translation
## datapre.py为数据集加载并分batch，构造词典
## transla.py为IWSLT16数据集应用于tranformer翻译任务的demo
## 环境：
     torchtext 0.10.0
     spacy 3.1.1
     torch 1.9.0
## 注意：由于IWSLT2016数据集默认使用谷歌网盘下载，国内无法直接调用，需要手动下载后更改IWSLT2016源代码(替换url下载)。
## 新增ppl和bleu计算
## 新增WMT14数据集的载入（任意数据集均可载入）（dataset_WMT14.py 为数据加载，构造词典）
## tanla_WMT14.py为WMT14数据集应用于tranformer翻译任务的demo，其中各种超参数可以调整
## DATA
   WMT14 可以通过该链接下载https://nlp.stanford.edu/projects/nmt/
