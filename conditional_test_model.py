import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

class ContextConditionedClassifier(nn.Module):
    def __init__(self, input_dim=2056, hidden_dim=512, dropout=0.2, classifier_dropout=0.3):
        super().__init__()
        
        self.context_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.context_to_scale = nn.Linear(hidden_dim, hidden_dim)
        self.context_to_shift = nn.Linear(hidden_dim, hidden_dim)
        
        self.mutation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        mutation = x[:, 0, :]
        context = x[:, 1, :]
        
        h_ctx = self.context_net(context)
        
        scale = self.context_to_scale(h_ctx)
        shift = self.context_to_shift(h_ctx)
        
        h_mut = self.mutation_net(mutation)
        
        h_conditioned = scale * h_mut + shift
        
        return self.classifier(h_conditioned)

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
    
    model = ContextConditionedClassifier(
        input_dim=2056,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        classifier_dropout=config["classifier_dropout"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_cls = getattr(optim, config["optimizer"])
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        tune.report(metrics={"loss": val_loss, "accuracy": val_acc})

        if val_loss < best_val_loss:
            print("Zapisuję model")
            import os
            print("Saving model to:", os.path.abspath("best_model_context_conditioned.pth"))
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_context_conditioned.pth")


def tune_hyperparameters(train_loader, val_loader, num_samples=15, max_num_epochs=10):
    config = {
        "epochs": tune.choice([8, 10, 12]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.1, 0.4),
        "classifier_dropout": tune.uniform(0.2, 0.5),
        "optimizer": tune.choice(["Adam", "AdamW"]),
        "hidden_dim": tune.choice([256, 512, 1024])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2
    )

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        tune.with_parameters(train_model_tune, data=(train_loader, val_loader)),
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="raytune_context_conditioned",
        resume="AUTO"  
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("\n✅ Najlepsza konfiguracja:")
    print(best_trial.config)
    print(f"Val Accuracy: {best_trial.last_result['accuracy']:.2f}%")

    return best_trial

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import glob
    import random
    import math
    import os
    import pyarrow.parquet as pq
    from torch.utils.data import IterableDataset

    class ParquetDataset(IterableDataset):
        def __init__(self, folder_path, features, target, batch_size=1024, shuffle_files=True):
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
                for batch in parquet_file.iter_batches(batch_size=self.batch_size, 
                                                      columns=self.features + [self.target]):
                    batch = batch.to_pydict()
                    x = torch.tensor(
                        [[batch[f][i] for f in self.features] for i in range(len(batch[self.target]))],
                        dtype=torch.float32
                    )
                    y = torch.tensor(batch[self.target], dtype=torch.float32)
                    yield x, y

    all_files = glob.glob(os.path.abspath("df_train_unique_2_new.parquet/*.parquet"))
    random.shuffle(all_files)

    split_ratio = 0.8
    split_idx = math.floor(len(all_files) * split_ratio)

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    features = ['POS_vector', 'Patho_Vector']
    target = 'ClinSigSimple'

    train_dataset = ParquetDataset(
        folder_path=None,
        features=features,
        target=target,
        batch_size=256,
        shuffle_files=True
    )
    train_dataset.files = train_files

    val_dataset = ParquetDataset(
        folder_path=None,
        features=features,
        target=target,
        batch_size=256,
        shuffle_files=False
    )
    val_dataset.files = val_files

    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    best_trial = tune_hyperparameters(train_loader, val_loader, num_samples=15)

    best_config = best_trial.config
    print("\nNajlepsze hiperparametry:", best_config)

    model = ContextConditionedClassifier(
        input_dim=2056,
        hidden_dim=best_config["hidden_dim"],
        dropout=best_config["dropout"],
        classifier_dropout=best_config["classifier_dropout"]
    )