import torch
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizer

from modules.data_processing import CLUENER


def accuracy(logits, labels):
    """
    logits: [batch_size, seq_len, num_labels]
    labels: [batch_size, seq_len]
    """
    flattened_targets = labels.view(-1)
    flattened_preds = torch.argmax(logits, dim=-1).view(-1)
    return accuracy_score(
        flattened_targets.cpu().numpy(), flattened_preds.cpu().numpy()
    )


def train(
    model,
    train_data,
    val_data,
    num_epochs,
    batch_size,
    lr,
    device="cuda",
    patience=5,
    checkpoint_save_path=None,
):
    model.to(device)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr)

    best_validation_accuracy = 0.0
    wait = 0

    for epoch in range(num_epochs):
        # training
        model.train()
        training_loss = 0.0
        training_accuracy = 0.0
        for batch in tqdm(
            train_dataloader,
            desc=f"training epoch {epoch+1}/{num_epochs}",
            position=0,
            leave=True,
        ):
            inputs, labels, _ = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            training_loss += loss.item()

            # training accuracy
            training_accuracy += accuracy(logits, labels)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss /= len(train_dataloader)
        training_accuracy /= len(train_dataloader)
        print(f"training loss: {training_loss}, training accuracy: {training_accuracy}")

        # validation
        model.eval()
        validation_loss = 0.0
        validation_accuracy = 0.0
        for batch in tqdm(
            val_dataloader,
            desc=f"validation epoch {epoch+1}/{num_epochs}",
            position=0,
            leave=True,
        ):
            inputs, labels, _ = batch
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                validation_loss += loss.item()

                # validation accuracy
                validation_accuracy += accuracy(logits, labels)

        validation_loss /= len(val_dataloader)
        validation_accuracy /= len(val_dataloader)
        print(
            f"validation loss: {validation_loss}, validation accuracy: {validation_accuracy}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            if checkpoint_save_path:
                torch.save(model.state_dict(), checkpoint_save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    dataset = CLUENER("data/cluener/train.json", tokenizer)

    # split train dataset into train and validation
    train_ratio = 0.9
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese", num_labels=len(dataset.label_names)
    )

    train(
        model,
        train_data,
        val_data,
        num_epochs=30,
        batch_size=128,
        lr=1e-5,
        device="cuda",
        patience=5,
        checkpoint_save_path="checkpoints/cluener.pth",
    )
