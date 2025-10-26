import glob
import random

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from attention_test_model import CrossAttentionClassifier
from conditional_test_model import ContextConditionedClassifier
from second_test_model import BinaryClassifier as SecondBinaryClassifier
from test_model import BinaryClassifier
from torch.utils.data import DataLoader, IterableDataset


def evaluate_model(model, loader, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    batch_count = 0
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    correct = 0
    total = 0

    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {total} samples...")
            batch_count += 1

    avg_loss = total_loss / batch_count
    accuracy = 100 * correct / total

    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")

    return avg_loss, accuracy


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    batch_count = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.unsqueeze(1).to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    accuracy = 100 * correct / total
    return avg_loss, accuracy


class ParquetDataset(IterableDataset):
    def __init__(
        self, folder_path, features, target, batch_size=1024, shuffle_files=True
    ):
        self.folder_path = folder_path
        self.files = glob.glob(f"{folder_path}/*.parquet") if folder_path else []
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle_files:
            random.shuffle(files)

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                batch_size=self.batch_size, columns=self.features + [self.target]
            ):
                batch = batch.to_pydict()
                x = torch.tensor(
                    [
                        [batch[f][i] for f in self.features]
                        for i in range(len(batch[self.target]))
                    ],
                    dtype=torch.float32,
                )
                y = torch.tensor(batch[self.target], dtype=torch.float32)
                yield x, y


if __name__ == "__main__":
    features = ["POS_vector", "Patho_Vector"]
    target = "ClinSigSimple"
    test_dataset = ParquetDataset(
        folder_path="df_test_unique_with_context_new.parquet",
        features=features,
        target=target,
        batch_size=256,
        shuffle_files=False,
    )
    print(test_dataset.files)

    test_loader_1 = DataLoader(test_dataset, batch_size=None, num_workers=0)
    test_loader_2 = DataLoader(test_dataset, batch_size=None, num_workers=0)
    test_loader_3 = DataLoader(test_dataset, batch_size=None, num_workers=0)
    test_loader_4 = DataLoader(test_dataset, batch_size=None, num_workers=0)
    print("First Binarty Classifier Evaluation:")
    model = BinaryClassifier(
        input_dim=4112, hidden_dims=[512, 256, 128], dropout=0.2427013306774357
    )
    model.load_state_dict(torch.load("first_model_best.pth", map_location="cuda"))
    model.to("cuda")
    val_loss, val_acc = validate(model, test_loader_1, "cuda")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-----------------------------------")
    print("Second Binarty Classifier Evaluation:")
    model = SecondBinaryClassifier(
        input_dim=2056, hidden_dims=[1024, 512, 128], dropout=0.4768807022739411
    )
    model.load_state_dict(torch.load("second_model_best.pth", map_location="cuda"))
    model.to("cuda")
    val_loss, val_acc = evaluate_model(model, test_loader_2)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-----------------------------------")
    print("Context Conditioned Classifier Evaluation:")
    model = ContextConditionedClassifier(
        input_dim=2056,
        hidden_dim=512,
        dropout=0.23312273899203848,
        classifier_dropout=0.26182611276296736,
    )
    model.load_state_dict(torch.load("conditional_best_model.pth", map_location="cuda"))
    model.to("cuda")
    val_loss, val_acc = evaluate_model(model, test_loader_3)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-----------------------------------")
    print("Cross Attention Classifier Evaluation:")
    model = CrossAttentionClassifier(
        input_dim=2056, hidden_dim=512, num_heads=16, dropout=0.11889171411317409
    )
    model.load_state_dict(torch.load("cross_attention_best.pth", map_location="cuda"))
    model.to("cuda")
    val_loss, val_acc = evaluate_model(model, test_loader_4)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-----------------------------------")
