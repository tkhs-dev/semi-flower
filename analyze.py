"""Debug analysis for FEMNIST model.

Usage examples:
  python analyze.py --model sample.pt --batch-size 128
  python analyze.py --model sample.pt --save-wrong samples/ --topk 5

This script computes:
- class frequencies in the centralized test split
- confusion matrix and per-class accuracy
- lists top confusions
- optionally saves some misclassified examples for inspection

It also provides a helper to run the same analysis with image inversion to check preprocessing mismatch.
"""

import os
import argparse
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from pytorchexample.task import Net, load_centralized_dataset, collate_fn, load_data

id_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

def load_model(path, device):
    model = Net()
    state = torch.load(path, map_location=device)
    # try to extract state_dict similarly to infer.py
    def _extract_state_dict(obj):
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
    extracted = _extract_state_dict(state)
    if extracted is None:
        raise RuntimeError('Could not extract state_dict from file')
    try:
        model.load_state_dict(extracted)
    except Exception:
        model.load_state_dict(extracted, strict=False)
    model.to(device)
    model.eval()
    return model


def compute_metrics(model, dataloader, device, topk=1):
    num_classes = None
    all_preds = []
    all_labels = []
    pred_counts = Counter()
    topk_hits = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            if num_classes is None:
                num_classes = probs.shape[1]
            topvals, topidx = torch.topk(probs, k=topk, dim=1)
            all_preds.append(topidx.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            # top1 pred counts
            top1 = topidx[:, 0].cpu().numpy()
            for p in top1:
                pred_counts[int(p)] += 1
            # topk hit
            labels_np = labels.cpu().numpy()
            for i in range(labels_np.shape[0]):
                total += 1
                if labels_np[i] in topidx[i].cpu().numpy():
                    topk_hits += 1
    preds = np.vstack(all_preds)  # shape (N, topk)
    labels = np.concatenate(all_labels)  # shape (N,)
    # compute confusion matrix for top1 only
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, gt in enumerate(labels):
        p = preds[i, 0]
        cm[gt, p] += 1
    # per-class accuracy
    per_class_acc = []
    class_counts = cm.sum(axis=1)
    for c in range(num_classes):
        total_c = int(class_counts[c])
        correct = int(cm[c, c])
        acc = correct / total_c if total_c > 0 else None
        per_class_acc.append((c, total_c, correct, acc))
    topk_hit_rate = topk_hits / total if total > 0 else 0.0
    return cm, per_class_acc, topk_hit_rate, pred_counts


def print_topk_confusions(cm, topn=10):
    # cm[true, pred]
    num_classes = cm.shape[0]
    confusions = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t == p:
                continue
            confusions.append((cm[t, p], t, p))
    confusions.sort(reverse=True)
    print(f"Top {topn} confusions (true, pred, count):")
    for cnt, true, pred in confusions[:topn]:
        if cnt > 0:
            print("  ", id_mapping[true], id_mapping[pred], cnt)


def save_misclassified_examples(dataloader, model, device, out_dir, max_per_class=20):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            imgs = batch['img'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            for j in range(imgs.shape[0]):
                gt = int(labels[j].item())
                pr = int(preds[j].item())
                if gt != pr:
                    img_tensor = imgs[j].cpu()
                    # unnormalize
                    img = img_tensor * 0.5 + 0.5
                    arr = (img.squeeze(0).numpy() * 255).astype('uint8')
                    pil = Image.fromarray(arr, mode='L')
                    fname = os.path.join(out_dir, f"idx_{i}_{j}_gt_{gt}_pr_{pr}.png")
                    pil.save(fname)
                    saved += 1
                    if saved >= max_per_class:
                        return


def analyze(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    # Build centralized test DataLoader with requested batch size so class counts are accurate
    raw = load_dataset('flwrlabs/femnist', split='train')
    split = raw.train_test_split(test_size=0.2, seed=42)
    test_dataset = split['test']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    cm, per_class_acc, topk_hit_rate, pred_counts = compute_metrics(model, test_loader, device, topk=args.topk)

    total_examples = sum([t for (_, t, _, _) in per_class_acc if t is not None])
    print(f"Total examples in test set: {total_examples}")

    # print overall accuracy
    correct_total = sum([corr for (_, _, corr, _) in per_class_acc if corr is not None])
    overall_acc = correct_total / total_examples if total_examples > 0 else 0.0
    print(f"Overall accuracy: {overall_acc:.4f}")

    # Print class frequency and accuracy for classes with most examples
    per_class_acc_sorted = sorted(per_class_acc, key=lambda x: x[1], reverse=True)
    print(f"Top {args.topk} classes by frequency (class, count, correct, acc):")
    for c, cnt, corr, acc in per_class_acc_sorted[:args.topk]:
        print("  ", id_mapping[c], cnt, corr, f"{acc:.4f}" if acc is not None else "N/A")

    # Print best performing classes (with >= 10 examples)
    per_class_acc_filtered = [(c, cnt, corr, acc) for (c, cnt, corr, acc) in per_class_acc if cnt >= 10]
    per_class_acc_filtered.sort(key=lambda x: (0.0 if x[3] is None else -x[3]))
    print("Best performing classes (class, count, correct, acc) with >=10 examples:")
    for c, cnt, corr, acc in per_class_acc_filtered[:args.topk]:
        print("  ", id_mapping[c], cnt, corr, f"{acc:.4f}" if acc is not None else "N/A")

    # Print worst performing classes (with >= 10 examples)
    per_class_acc_filtered = [(c, cnt, corr, acc) for (c, cnt, corr, acc) in per_class_acc if cnt >= 10]
    per_class_acc_filtered.sort(key=lambda x: (1.0 if x[3] is None else x[3]))
    print("Worst performing classes (class, count, correct, acc) with >=10 examples:")
    for c, cnt, corr, acc in per_class_acc_filtered[:args.topk]:
        print("  ", id_mapping[c], cnt, corr, f"{acc:.4f}" if acc is not None else "N/A")

    # Top confusions
    print_topk_confusions(cm, topn=args.topk)

    # Predicted distribution (top1)
    most_common_preds = pred_counts.most_common(args.topk)
    print("Most common predicted classes (class, count):")
    for cls, cnt in most_common_preds:
        print("  ", id_mapping[cls], cnt)

    if args.save_wrong:
        print(f"Saving misclassified examples to {args.save_wrong} ...")
        save_misclassified_examples(test_loader, model, device, args.save_wrong)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='final_model.pt')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--save-wrong', type=str, default=None,
                        help='Directory to save misclassified examples')
    args = parser.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()

