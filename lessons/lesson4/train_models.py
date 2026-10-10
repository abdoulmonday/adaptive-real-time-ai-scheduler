"""Train/evaluate two MLP models on Fashion-MNIST (genuine dataset, not synthetic)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from models import make_model


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return total_loss / total, correct / total


def train_one(model: nn.Module, loader: DataLoader, device: str, epochs: int, lr: float):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        correct, seen, total_loss = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            seen += labels.numel()
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
        result = {'epoch': epoch, 'train_loss': total_loss / seen, 'train_accuracy': correct / seen}
        logs.append(result)
        print(f"    epoch {epoch:>2}/{epochs}: loss={result['train_loss']:.4f}, train_acc={result['train_accuracy']:.2%}")
    return logs


def main():
    parser = argparse.ArgumentParser(description='Train two Fashion-MNIST MLPs')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--limit-train', type=int, default=0, help='Optional training subset for pipeline smoke checks')
    parser.add_argument('--limit-test', type=int, default=0, help='Optional test subset for pipeline smoke checks')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.lr <= 0 or args.limit_train < 0 or args.limit_test < 0:
        parser.error('Numeric parameters must be positive (limits can be zero)')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but no CUDA-capable device is available')
    # torchvision is imported on demand; synthetic smoke_test.py requires only torch.
    from torchvision import datasets, transforms

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    transform = transforms.ToTensor()
    print('Loading Fashion-MNIST (download may be needed on first run)...')
    training = datasets.FashionMNIST(str(args.data_dir), train=True, download=True, transform=transform)
    testing = datasets.FashionMNIST(str(args.data_dir), train=False, download=True, transform=transform)
    if args.limit_train:
        training = Subset(training, range(min(args.limit_train, len(training))))
    if args.limit_test:
        testing = Subset(testing, range(min(args.limit_test, len(testing))))
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = {
        'dataset': 'Fashion-MNIST', 'train_count': len(training), 'test_count': len(testing),
        'epochs': args.epochs, 'seed': args.seed, 'learning_rate': args.lr,
        'device': args.device, 'torch_version': torch.__version__, 'threads': args.threads,
        'models': {},
    }
    model_dir = args.output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('small', 'large'):
        print(f'\nTraining {name.upper()} model...')
        model = make_model(name)
        logs = train_one(model, train_loader, args.device, args.epochs, args.lr)
        loss, accuracy = evaluate(model, test_loader, args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f'    test_acc={accuracy:.2%}, params={params}')
        result['models'][name] = {
            'parameters': params, 'test_loss': loss, 'test_accuracy': accuracy,
            'training_history': logs,
        }
        torch.save(model.cpu().state_dict(), model_dir / f'{name}.pt')
    output = args.output_dir / 'training_summary.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nSaved:', output)
    print('Model weights:', model_dir)
    if args.limit_train or args.limit_test:
        print('PIPELINE CHECK ONLY: subset accuracy is not a final experiment result.')


if __name__ == '__main__':
    main()
