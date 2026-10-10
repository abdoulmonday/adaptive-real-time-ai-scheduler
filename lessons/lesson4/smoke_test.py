"""Offline sanity check. Synthetic random data, no accuracy claims."""
import torch
from torch import nn
from models import make_model


def main():
    torch.manual_seed(42)
    for name in ('small', 'large'):
        model = make_model(name)
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        outputs = model(x)
        assert tuple(outputs.shape) == (16, 10)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f'{name}: forward + backward OK; params={sum(p.numel() for p in model.parameters())}; synthetic loss={loss.item():.3f}')
    print('Smoke test passed: not a real training or validation run.')


if __name__ == '__main__':
    main()
