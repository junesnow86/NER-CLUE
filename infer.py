import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer

from modules.data_processing import CLUENER, CLUENERForInference
from modules.inference import generate_label


@torch.no_grad()
def infer(model, test_data, id2label, device="cuda", checkpoint_save_path=None):
    """
    Args:
    -----
    test_data: 一个CLUEForInference对象
    model: 训练好的模型

    Returns:
    --------
    output_data: 输出数据，包含以下字段
        - id: 样本id
        - text: 原始文本
        - label: 抽取的实体
            - entity_type: 实体类型
                - entity: 实体文本
                    - [[start, end]]

    Label Example:
    --------------
    {
        "company": {"浙商银行": [[0, 3]]},
        "name": {"叶老桂": [[9, 11]]},
    }
    """
    if checkpoint_save_path:
        model.load_state_dict(torch.load(checkpoint_save_path))

    model.to(device)
    model.eval()

    dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    output_data = []

    for batch in tqdm(dataloader, desc="inferencing", position=0, leave=True):
        id, text, inputs = batch
        id = id.item()
        text = text[0]
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        token_label_names = [id2label[pred] for pred in preds]

        label = generate_label(text, token_label_names)

        output_data.append(
            {
                "id": id,
                "text": text,
                "label": label,
            }
        )

    return output_data


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_data = CLUENER("data/cluener/train.json", tokenizer)
    id2label = train_data.id2label

    test_data = CLUENERForInference("data/cluener/test.json", tokenizer)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese", num_labels=len(train_data.label_names)
    )

    output_data = infer(
        model, test_data, id2label, checkpoint_save_path="checkpoints/cluener.pth"
    )

    with open("results_of_test_data.json", "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        "Inference finished! Check the results in 'results_of_test_data.json' under the same dir."
    )
