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
    

def save_gradcam_result(model, df, tfms, fold, target_type="fusion"):

    model.eval()

    sample_rows = df.sample(5, random_state=CFG.SEED)

    for idx, sample_row in enumerate(sample_rows.itertuples()):

        img_path = os.path.join(
            CFG.TRAIN_DIR,
            f"{sample_row.id_code}.jpg"
        )
        
        orig_img = cv2.imread(img_path)        

        if orig_img is None:
            print(f"Could not read image: {img_path}")
            continue

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (CFG.IMG_SIZE, CFG.IMG_SIZE))

        input_tensor = tfms(image=orig_img)['image'].unsqueeze(0).to(CFG.DEVICE)

        if target_type == "fusion":
            target_layers = [model.backbone.stages[-1].blocks[-1]]

        elif target_type == "edge":
            target_layers = [model.edge_stream[0]]

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=None
        )

        targets = [ClassifierOutputTarget(sample_row.diagnosis)]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        visualization = show_cam_on_image(
            orig_img.astype(np.float32) / 255.0,
            grayscale_cam,
            use_rgb=True
        )

        save_name = f"gradcam_{target_type}_fold_{fold}_img_{idx+1}.jpg"

        # SIDE-BY-SIDE DISPLAY
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # GradCAM overlay
        axes[1].imshow(visualization)
        axes[1].set_title(f"GradCAM ({target_type})")
        axes[1].axis("off")

        plt.suptitle(f"Fold {fold} - Img {idx+1}")

        plt.tight_layout()

        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        del cam
        torch.cuda.empty_cache()

        print(f"[INFO] Saved GradCAM: {save_name}")
