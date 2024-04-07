import json

import torch
from torch.utils.data import Dataset


def tokenize_and_convert_labels(data, with_label=True):
    """对原始数据进行分词，将实体label转化为BIO表示

    Args:
    -----
    data: 一条原始数据，包含文本和实体标签（test.json中有id字段，但没有label字段）

    Examples:
    ---------
    data = {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
        "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}},
    }

    Returns:
    --------
    data = {
        "text": 原始text字段,
        "label": 原始label字段,
        "tokens": 按字符分词后的tokens列表,
        "token_labels": 每个token对应的BIOES标签列表
    }
    """
    text = data["text"]

    # 按字符分词
    def tokenize_by_char(text):
        tokens = []
        for char in text:
            tokens.append(char)
        return tokens

    tokens = tokenize_by_char(text)
    data["tokens"] = tokens

    if not with_label:
        return data

    # 将原始实体标签转化成BIO标签
    label = data["label"]
    labels = ["O"] * len(tokens)
    for entity_type in label:
        for entity in label[entity_type]:
            for span in label[entity_type][entity]:
                start, end = span
                labels[start] = f"B-{entity_type}"
                if start < end:
                    for i in range(start + 1, end + 1):
                        labels[i] = f"I-{entity_type}"

    data["token_labels"] = labels
    return data


class CLUENER(Dataset):
    def __init__(self, root, tokenizer, max_length=50):
        self.root = root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._raw_data = self._load_data()
        self._process_data()

    def _load_data(self):
        data = []
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                data.append(sample)
        return data

    def _process_data(self):
        """
        Args:
        -----
        self._raw_data: 从json文件中读取的原始数据，一条数据是一个字典

        Example:
        data = {
            "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
            "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}},
        }

        Returns:
        --------
        dataset: 处理后的数据集，包含以下字段
            - self.data: 处理后的样本数据列表，每个样本是一个字典，包含以下字段
                - text: 原始文本
                - label: 原始实体标签
                - tokens: 按字符分词后的tokens列表
                - token_labels: 每个token对应的BIOES标签列表
            - self.label2id: label到id的映射
            - self.id2label: id到label的映射
            - self.label_names: 含有所有label name的列表
        """
        self.label_names = set()
        self.data = []

        for sample in self._raw_data:
            sample = tokenize_and_convert_labels(sample)
            self.data.append(sample)
            for token_label in sample["token_labels"]:
                self.label_names.add(token_label)

        self.label_names = list(self.label_names) + ["PAD"]
        self.label_names.sort()
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        tokens = data["tokens"]
        token_labels = data["token_labels"]

        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        # 编码label
        labels = torch.tensor(
            [self.label2id[label] for label in token_labels]
            + [self.label2id["PAD"]] * (self.max_length - len(token_labels))
        )

        return inputs, labels, data["text"]


class CLUENERForInference(Dataset):
    def __init__(self, root, tokenizer, max_length=50):
        self.root = root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._raw_data = self._load_data()
        self._process_data()

    def _load_data(self):
        data = []
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                data.append(sample)
        return data

    def _process_data(self):
        """
        Args:
        -----
        self._raw_data: 从json文件中读取的原始数据，一条数据是一个字典
            - id: 样本id
            - text: 原始文本

        Returns:
        --------
        dataset: 处理后的数据集，包含以下字段
            - self.data: 处理后的样本数据列表，每个样本是一个字典，包含以下字段
                - id: 样本id
                - text: 原始文本
                - tokens: 按字符分词后的tokens列表
        """
        self.data = []

        for sample in self._raw_data:
            sample = tokenize_and_convert_labels(sample, with_label=False)
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        tokens = data["tokens"]

        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        return data["id"], data["text"], inputs
