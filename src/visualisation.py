# =========================================
# VISUALIZATION
# =========================================
def plot_history(history, save_path="training_history.jpg"):

    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Val F1")
    plt.title("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, save_path="confusion_matrix.jpg"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CFG.CLASS_NAMES,
        yticklabels=CFG.CLASS_NAMES
        )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"[INFO] Confusion matrix saved: {save_path}")

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_probs, save_path="roc_curve.jpg"):

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(CFG.NUM_CLASSES)
    )

    plt.figure(figsize=(7,6))

    for i in range(CFG.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_probs[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{CFG.CLASS_NAMES[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[INFO] ROC curve saved: {save_path}")
    
