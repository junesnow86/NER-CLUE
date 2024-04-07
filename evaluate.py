import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer

from modules.data_processing import CLUENER


@torch.no_grad()
def evaluate(
    model,
    dev_data,
    label2id,
    target_label_names,
    batch_size,
    consider_special_tokens=False,
    device="cuda",
    checkpoint_save_path=None,
):
    if checkpoint_save_path:
        model.load_state_dict(torch.load(checkpoint_save_path))

    model.to(device)
    model.eval()

    dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    all_label_ids = []
    all_preds = []

    for batch in tqdm(dev_dataloader, desc="evaluating", position=0, leave=True):
        inputs, labels, _ = batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        all_label_ids.extend(labels.view(-1).cpu().numpy().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).view(-1).cpu().numpy().tolist())

    if not consider_special_tokens:
        target_names = [
            label for label in target_label_names if (label != "PAD" and label != "O")
        ]
    else:
        target_names = target_label_names
    target_labels = [label2id[label] for label in target_names]

    report = classification_report(
        all_label_ids,
        all_preds,
        labels=target_labels,
        output_dict=True,
        target_names=target_names,
        zero_division=0,
    )
    report_str = classification_report(
        all_label_ids,
        all_preds,
        labels=target_labels,
        target_names=target_names,
        zero_division=0,
    )
    print(report_str)
    return report


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_data = CLUENER("data/cluener/train.json", tokenizer)
    target_label_names = train_data.label_names
    label2id = train_data.label2id

    dev_data = CLUENER("data/cluener/dev.json", tokenizer)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese", num_labels=len(target_label_names)
    )

    report = evaluate(
        model,
        dev_data,
        label2id=label2id,
        target_label_names=target_label_names,
        batch_size=128,
        checkpoint_save_path="checkpoints/cluener.pth",
    )
