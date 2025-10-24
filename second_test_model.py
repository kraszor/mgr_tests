import glob
import math
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, IterableDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=2056, hidden_dims=[512, 256, 128], dropout=0.3):
        super(BinaryClassifier, self).__init__()

        branch1_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            branch1_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.branch1 = nn.Sequential(*branch1_layers)

        branch2_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            branch2_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.branch2 = nn.Sequential(*branch2_layers)

        concat_dim = hidden_dims[-1] * 2
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]

        h1 = self.branch1(x1)
        h2 = self.branch2(x2)

        h_concat = torch.cat([h1, h2], dim=1)

        output = self.fusion(h_concat)
        return output


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / (batch_idx + 1)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    batch_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
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


def train_model_tune(config, data=None, checkpoint_dir=None):
    train_loader, val_loader = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(
        input_dim=2056, hidden_dims=config["hidden_dims"], dropout=config["dropout"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = getattr(optim, config["optimizer"])
    optimizer = optimizer_cls(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        tune.report(metrics={"loss": val_loss, "accuracy": val_acc})

        if val_loss < best_val_loss:
            print("Zapisuję model")
            import os

            print("Saving model to:", os.path.abspath("best_model_ray_tune_2.pth"))
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_ray_tune_2.pth")


def tune_hyperparameters(train_loader, val_loader, num_samples=15, max_num_epochs=10):
    config = {
        "epochs": tune.choice([8, 10, 12]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.1, 0.5),
        "optimizer": tune.choice(["Adam", "AdamW", "SGD"]),
        "hidden_dims": tune.choice(
            [
                [512, 256, 128],
                [1024, 512, 128],
                [256, 128, 64],
            ]
        ),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        tune.with_parameters(train_model_tune, data=(train_loader, val_loader)),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="raytune_binary_classifier_2",
        resume="AUTO",
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("\n✅ Najlepsza konfiguracja:")
    print(best_trial.config)
    print(f"Val Accuracy: {best_trial.last_result['accuracy']:.2f}%")

    return best_trial


class ParquetDataset(IterableDataset):
    def __init__(
        self, folder_path, features, target, batch_size=1024, shuffle_files=True
    ):
        self.folder_path = folder_path
        self.files = glob.glob(f"{folder_path}/*.parquet")
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


# @title
if __name__ == "__main__":
    all_files = glob.glob(os.path.abspath("df_train_unique_2_new.parquet/*.parquet"))
    print(all_files)
    random.shuffle(all_files)

    split_ratio = 0.8
    split_idx = math.floor(len(all_files) * split_ratio)

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    print(f"Train files: {train_files}")
    print(f"Val files: {val_files}")
    features = ["POS_vector", "Patho_Vector"]
    target = "ClinSigSimple"

    train_dataset = ParquetDataset(
        folder_path=None,
        features=features,
        target=target,
        batch_size=256,
        shuffle_files=True,
    )
    train_dataset.files = train_files

    val_dataset = ParquetDataset(
        folder_path=None,
        features=features,
        target=target,
        batch_size=256,
        shuffle_files=False,
    )
    val_dataset.files = val_files

    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    best_trial = tune_hyperparameters(train_loader, val_loader)

    best_config = best_trial.config
    print("\nNajlepsze hiperparametry:", best_config)

    model = BinaryClassifier(
        input_dim=4112,
        hidden_dims=best_config["hidden_dims"],
        dropout=best_config["dropout"],
    )

    model.load_state_dict(torch.load("best_model_ray_tune_2.pth", map_location="gpu"))
    model.eval()

    print("\n✅ Najlepszy model został wczytany i gotowy do użycia.")
