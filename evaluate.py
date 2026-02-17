# evaluate.py
import torch
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def evaluate_model_comprehensive(model, testloader, criterion, class_names=['normal_mild', 'moderate', 'severe']):
    device = next(model.parameters()).device
    model.eval()

    test_loss = 0.0
    correct = 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for images, lbls in tqdm(testloader, desc="Evaluating"):
            images, lbls = images.to(device), lbls.to(device)
            with autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, lbls)
            test_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == lbls).sum().item()
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
            probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    preds, labels, probs = np.array(preds), np.array(labels), np.array(probs)
    test_loss /= len(testloader)
    acc = 100 * correct / len(testloader.dataset)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)

    print(f"\nTest Loss: {test_loss:.4f} | Accuracy: {acc:.2f}% | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print("\nClassification Report:\n", classification_report(labels, preds, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.subplot(1, 2, 2)
    sns.heatmap(cm / cm.sum(axis=1)[:, None], annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # ROC
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

    return {
        'test_loss': test_loss, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec,
        'confusion_matrix': cm, 'predictions': preds, 'labels': labels, 'probs': probs
    }
