def generate_label(text, token_label_names):
    """将BIO形式的token标签转化成span形式的实体标签
    Args:
    -----
    text: 输入文本
    token_label_names: 每个token对应的BIO标签列表

    Returns:
    --------
    label: 抽取的实体
        - entity_type: 实体类型
            - entity: 实体文本
                - [[start, end]]

    Examples:
    ---------
    text = "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，"
    token_label_names = [
        "B-company",
        "I-company",
        "I-company",
        "I-company",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-name",
        "I-name",
        "I-name",
        "O",
        "O",
        "O",
    ]
    label = generate_label(text, token_label_names)

    Outputs:
    {'company': {'浙商银行': [[0, 3]]}, 'name': {'叶老桂': [[9, 11]]}}
    """
    label = {}

    entity_type = None
    entity = None
    start = None

    for i, token_label in enumerate(token_label_names):
        if i >= len(text):
            break

        if token_label.startswith("B-"):
            if entity_type is None:
                entity_type = token_label[2:]
                entity = text[i]
                start = i
            else:
                if entity_type != token_label[2:]:
                    label.setdefault(entity_type, {}).setdefault(entity, []).append(
                        [start, i - 1]
                    )
                    entity_type = token_label[2:]
                    entity = text[i]
                    start = i
        elif token_label.startswith("I-"):
            if entity_type is None or entity is None:
                continue
            entity += text[i]
        elif token_label == "O" or token_label == "PAD":
            if entity_type is not None:
                label.setdefault(entity_type, {}).setdefault(entity, []).append(
                    [start, i - 1]
                )
                entity_type = None
                entity = None
                start = None
        else:
            raise ValueError(f"Invalid token label: {token_label}")

    return label
