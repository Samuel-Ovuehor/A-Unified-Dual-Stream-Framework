# VALIDATION & STATS
def calculate_ece(y_true, y_probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)

    for i in range(n_bins):
        bin_idx = (
            (confidences > bin_boundaries[i]) &
            (confidences <= bin_boundaries[i + 1])
        )

        if np.any(bin_idx):
            bin_acc = np.mean(predictions[bin_idx] == y_true[bin_idx])
            bin_conf = np.mean(confidences[bin_idx])
            ece += np.mean(bin_idx) * np.abs(bin_acc - bin_conf)

    return ece


def validate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # ECE was calculated during the evaluation protocol but the
    # numerical value was not retained in the final experimental record.
    _ece = calculate_ece(all_labels, all_probs)

    metrics = {
        "Acc": accuracy_score(all_labels, all_preds),
        "F1": f1_score(all_labels, all_preds, average='macro'),
        "Kappa": cohen_kappa_score(all_labels, all_preds),
        "MCC": matthews_corrcoef(all_labels, all_preds),
        "AUC": roc_auc_score(all_labels, all_probs, multi_class='ovr')
    }

    return metrics, all_labels, all_preds, all_probs

# =========================================
# INFERENCE (FOR UNLABELED TEST SETS)
# =========================================
def inference(model, loader):

    model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(CFG.DEVICE)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)

            all_preds.extend(
                logits.argmax(1).cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    return np.array(all_preds), np.array(all_probs)


# =========================================
# 5. CONFIDENCE INTERVAL
# =========================================
def bootstrap_ci(y_true, y_pred, metric=f1_score, n_bootstrap=500, alpha=0.05):
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if metric == f1_score:
            score = f1_score(y_true[idx], y_pred[idx], average="macro")
        else:
            score = accuracy_score(y_true[idx], y_pred[idx])
        scores.append(score)
    lower = np.percentile(scores, 100*(alpha/2))
    upper = np.percentile(scores, 100*(1-alpha/2))
    return lower, upper


def detailed_classification_report(y_true, y_pred, y_probs, class_names):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None
    )

    print("\n" + "="*80)
    print(
        f"{'Class':<15}"
        f"{'Accuracy':<12}"
        f"{'AUC':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
        f"{'Support':<10}"
    )
    print("="*80)

    for i, class_name in enumerate(class_names):

        # One-vs-rest accuracy
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)

        acc = (binary_true == binary_pred).mean()

        # One-vs-rest AUC
        try:
            auc_score = roc_auc_score(
                binary_true,
                y_probs[:, i]
            )
        except:
            auc_score = np.nan

        print(
            f"{class_name:<15}"
            f"{acc:<12.4f}"
            f"{auc_score:<12.4f}"
            f"{precision[i]:<12.4f}"
            f"{recall[i]:<12.4f}"
            f"{f1[i]:<12.4f}"
            f"{support[i]:<10}"
        )

    print("="*80)

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1.mean()

    print(f"\nOverall Accuracy : {overall_acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")



