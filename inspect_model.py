"""Inspect final layer weights and biases of the trained Net.

Usage:
  python inspect_model.py --model sample.pt

This prints statistics for fc3 weights and biases, top classes by bias and by weight norm,
and runs a few synthetic forward passes (zero input and random noise) to observe predicted classes.
"""
import argparse
import torch
from pytorchexample.task import Net


def extract_state_dict(obj):
    if isinstance(obj, dict):
        if all(hasattr(v, 'dim') for v in obj.values()):
            return obj
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        for v in obj.values():
            if isinstance(v, dict) and all(hasattr(x, 'dim') for x in v.values()):
                return v
    return None


def analyze(model_path, device):
    print('Loading model:', model_path)
    obj = torch.load(model_path, map_location=device)
    state = extract_state_dict(obj)
    if state is None:
        raise RuntimeError('Cannot find state_dict in file')
    # Inspect fc3 parameters
    # Determine fc3 key name by checking keys
    keys = list(state.keys())
    print('State dict keys (sample):', keys[:10])
    # common key suffix: 'fc3.weight' or 'classifier.2.weight' etc. Try to find weight/bias keys
    weight_key = None
    bias_key = None
    for k in keys:
        if k.endswith('fc3.weight') or k.endswith('fc3.weight'):
            weight_key = k
        if k.endswith('fc3.bias') or k.endswith('fc3.bias'):
            bias_key = k
    # fallback: find the last linear weight in state dict
    if weight_key is None:
        for k in reversed(keys):
            if k.endswith('.weight') and state[k].ndim == 2:
                weight_key = k
                break
    if bias_key is None:
        for k in reversed(keys):
            if k.endswith('.bias') and state[k].ndim == 1 and state[k].shape[0] == state[weight_key].shape[0]:
                bias_key = k
                break
    print('Using weight key:', weight_key)
    print('Using bias key:', bias_key)
    W = state[weight_key]
    b = state[bias_key]
    # Convert to float tensors
    W = W.to(device).float()
    b = b.to(device).float()
    # Stats
    print('fc weight shape:', W.shape)
    print('fc bias shape:', b.shape)
    print('weight stats: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}'.format(W.mean().item(), W.std().item(), W.min().item(), W.max().item()))
    print('bias stats: mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}'.format(b.mean().item(), b.std().item(), b.min().item(), b.max().item()))
    # top bias classes
    bias_vals = b.cpu().numpy()
    import numpy as np
    top_bias_idx = np.argsort(-bias_vals)[:10]
    print('Top 10 classes by bias (class, bias):')
    for idx in top_bias_idx:
        print(idx, bias_vals[idx])
    # weight norm per class (row norm)
    row_norms = torch.norm(W, dim=1).cpu().numpy()
    top_norm_idx = np.argsort(-row_norms)[:10]
    print('Top 10 classes by weight row-norm (class, norm):')
    for idx in top_norm_idx:
        print(idx, row_norms[idx])
    # check outputs for zero and random input
    net = Net()
    try:
        net.load_state_dict(state)
    except Exception:
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    with torch.no_grad():
        zero = torch.zeros(1,1,28,28, device=device)
        out_zero = torch.softmax(net(zero), dim=1).cpu().numpy()[0]
        print('Top5 for zero input:')
        top5 = np.argsort(-out_zero)[:5]
        for i in top5:
            print(i, out_zero[i])
        rand = torch.randn(1,1,28,28, device=device)
        out_rand = torch.softmax(net(rand), dim=1).cpu().numpy()[0]
        print('Top5 for random input:')
        top5r = np.argsort(-out_rand)[:5]
        for i in top5r:
            print(i, out_rand[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='final_model.pt')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    analyze(args.model, device)

