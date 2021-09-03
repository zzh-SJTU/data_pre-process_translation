
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
#读取翻译任务的数据集txt文件，source和target配对，参数为文件的地址。创建该实例之后可以直接放入dataloader中。
class Custom_dataset(Dataset):
    def __init__(self, Sourcefile_dir, targetfile_dir):
        self.sour=open(Sourcefile_dir, encoding='utf-8').readlines()
        self.tar=open(targetfile_dir, encoding='utf-8').readlines()
        self.counter=0
    def __len__(self):
        return len(self.sour)
    def __getitem__(self, idx):
        return self.sour[idx], self.tar[idx]
    def __iter__(self):
        return iter([self.sour[0], self.tar[0]])
    def __next__(self):
        self.counter=self.counter+1
        return [self.sour[self.counter], self.tar[self.counter]]

import torch
import torchtext.datasets
from torchtext import data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
#将该地址更换为实际数据存储的地址
source_train_dir=r"D:\科研\WMT14\train.en"    
source_test_dir=r"D:\科研\WMT14\newstest2015.en"
target_train_dir=r"D:\科研\WMT14\train.de"
target_test_dir=r"D:\科研\WMT14\newstest2015.de"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    WMT14_train=Custom_dataset(source_train_dir,target_train_dir)
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(WMT14_train, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)
# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

BATCH_SIZE = 10
WMT14_train=Custom_dataset(source_train_dir,target_train_dir)
train_dataloader = DataLoader(WMT14_train, batch_size=BATCH_SIZE, collate_fn=collate_fn)


