def main():
    # 1. Load Data
    full_df = pd.read_csv(CFG.TRAIN_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)

    # --- FIX 1: Define transforms BEFORE using them ---
        
    train_tfms = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    val_tfms = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    # 2. Run Harmony Search for Hyperparameters
    # Split a small validation set JUST for the Harmony Search
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG.SEED)

    train_idx, val_idx = next(gss.split(full_df, groups=full_df["id_code"]))

    h_train = full_df.iloc[train_idx]
    h_val = full_df.iloc[val_idx]

    
    h_train_ds = MedicalImageDataset(h_train, CFG.TRAIN_DIR, train_tfms)
    h_val_ds   = MedicalImageDataset(h_val, CFG.TRAIN_DIR, val_tfms)

    print("Train dataset size:", len(h_train_ds))
    print("Val dataset size:", len(h_val_ds))
    print("Test dataset size:", len(test_df))

    best_params = harmony_search(h_train_ds, h_val_ds)
    print(f"\n[OPTIMIZATION COMPLETE] Best Params: {best_params}")

    # 3. Setup K-Fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    fold_metrics = []

    # K-FOLD LOOP
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(full_df, full_df["diagnosis"], 
                                                           groups=full_df["id_code"])):
        print(f"\n{'='*20} STARTING FOLD {fold+1} {'='*20}")

        t_df = full_df.iloc[train_idx]
        v_df = full_df.iloc[val_idx]

        train_ds = MedicalImageDataset(
            t_df,
            CFG.TRAIN_DIR,
            train_tfms
        )
        
        val_ds = MedicalImageDataset(
            v_df,
            CFG.TRAIN_DIR,
            val_tfms
        )
        # --- FIX 2: Use best_params from the search, don't overwrite them! ---
        train_loader = DataLoader(train_ds, batch_size=best_params['bs'], sampler=get_sampler(t_df),
                                 num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, batch_size=best_params['bs'], 
                                num_workers=NUM_WORKERS ,pin_memory=torch.cuda.is_available())


        model = DualStreamModel().to(CFG.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'])

        counts = t_df["diagnosis"].value_counts().sort_index().values
        criterion = CBFocalLoss(samples_per_cls=counts, beta=0.9999, gamma=2.5).to(CFG.DEVICE)

        best_f1 = 0.0
        history = {
            "train_loss": [],
            "train_f1": [],
            "train_acc": [],
            "val_f1": [],
            "val_acc": []
        } 

        
        # --- TRAINING LOOP ---
        # Wrap the epoch range in tqdm to see an overall progress bar
        # --- TRAINING LOOP (Inside Main K-Fold) ---
        for epoch in range(CFG.FINAL_EPOCHS):
            model.train()
            train_preds = []
            train_labels = []
            # Use leave=True so the bar stays on screen after 100%
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False)

            running_loss = 0.0
            for x, y in pbar:
                x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
                optimizer.zero_grad()

                outputs = model(x)
                preds = outputs.argmax(1)

                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y.cpu().numpy())

                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # Update the bar with the current loss LIVE
                pbar.set_postfix_str(f"loss={running_loss/(pbar.n+1):.4f}")

            train_acc = accuracy_score(train_labels, train_preds)
            
            train_f1 = f1_score(
                train_labels,
                train_preds,
                average='macro'
            )  
            
            # VALIDATION
            m, _, _, _ = validate(model, val_loader)

            print(
                f"[Fold {fold+1} | Epoch {epoch+1}] "
                f"Acc: {m['Acc']:.4f} | "
                f"F1: {m['F1']:.4f} | "
                f"AUC: {m['AUC']:.4f} | "
                f"Kappa: {m['Kappa']:.4f} | "
                f"MCC: {m['MCC']:.4f} | "
                f"ECE: {m['ECE']:.4f}"
            )            
        
            # SAVE HISTORY HERE
            history["train_loss"].append(running_loss / len(train_loader))
            history["train_f1"].append(train_f1)
            history["train_acc"].append(train_acc)
        
            history["val_f1"].append(m["F1"])
            history["val_acc"].append(m["Acc"])
    
            
            if m['F1'] > best_f1:
                best_f1 = m['F1']
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pth")
                # Use pbar.write instead of print to avoid breaking the bar
                pbar.write(f" -> [Fold {fold+1}] New Best F1: {best_f1:.4f}")
                print(f" -> [Fold {fold+1}, Epoch {epoch+1}] New Best F1: {best_f1:.4f}")

        # --- POST-FOLD EVALUATION & XAI ---
        # 1. Load the best weights discovered during this fold
        model.load_state_dict(torch.load(f"best_model_fold{fold+1}.pth", weights_only=True))

        # 2. Final validation for statistical reporting
        m, y_true_fold, y_pred_fold, y_probs_fold = validate(model, val_loader)

        detailed_classification_report(
            y_true_fold,
            y_pred_fold,
            y_probs_fold,
            CFG.CLASS_NAMES
        )
        from sklearn.metrics import classification_report
        
        print(
            classification_report(
                y_true_fold,
                y_pred_fold,
                target_names=CFG.CLASS_NAMES,
                digits=4
            )
        )        
        
        fold_metrics.append(m)
        
        plot_confusion(
            y_true_fold,
            y_pred_fold,
            save_path=f"confusion_matrix_fold_{fold+1}.jpg"
        )
        
        plot_history(
            history,
            save_path=f"training_history_fold_{fold+1}.jpg"
        )

        # 3. Generate Grad-CAM for both Global (Fusion) and Local (Edge) streams
        # We shuffle v_df to ensure Grad-CAM isn't always looking at the same class
        sample_v_df = v_df.sample(frac=1, random_state=CFG.SEED)
        save_gradcam_result(model, sample_v_df, val_tfms, fold+1, target_type="fusion")
        save_gradcam_result(model, sample_v_df, val_tfms, fold+1, target_type="edge")

        # 4. MEMORY CLEANUP
        # Crucial for preventing 'Out of Memory' crashes during K-Fold
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        print(f"[INFO] Fold {fold+1} metrics logged. GPU memory cleared.")

    # 4. Final Statistical Reporting (IEEE TMI Requirement)
    results_df = pd.DataFrame(fold_metrics)
    
    print("\n" + "="*40)
    print("OVERALL K-FOLD STATS")
    print("="*40)
    
    for col in results_df.columns:
    
        if np.issubdtype(results_df[col].dtype, np.number):
    
            values = results_df[col].values
    
            mean = np.mean(values)
    
            std = np.std(values, ddof=1)
    
            # 95% Confidence Interval
            ci_low, ci_high = stats.t.interval(
                confidence=0.95,
                df=len(values)-1,
                loc=mean,
                scale=stats.sem(values)
            )
    
            print(
                f"{col:10}: "
                f"{mean:.4f} ± {std:.4f} "
                f"(95% CI: {ci_low:.4f} - {ci_high:.4f})"
            )

            
    # 5. Final Production Training & Test
    print("\n" + "="*30)
    print("STARTING FINAL PRODUCTION TRAINING")
    print("="*30)

    full_train_ds = MedicalImageDataset(
        full_df,
        CFG.TRAIN_DIR,
        train_tfms
    )
    full_train_loader = DataLoader(full_train_ds, batch_size=best_params['bs'], 
                                   sampler=get_sampler(full_df), num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(
        MedicalImageDataset(
            test_df,
            CFG.TEST_DIR,
            val_tfms
        ),
        batch_size=best_params['bs'],
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    final_model = DualStreamModel().to(CFG.DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'])
    counts = full_df['diagnosis'].value_counts().sort_index().values
    criterion = CBFocalLoss(samples_per_cls=counts, beta=0.9999, gamma=2.5).to(CFG.DEVICE)


    for epoch in range(CFG.FINAL_EPOCHS):
        final_model.train()
        for x, y in tqdm(full_train_loader, desc=f"Final Production Epoch {epoch+1}", leave=False):
            x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
            optimizer.zero_grad()
            criterion(final_model(x), y).backward()
            optimizer.step()
            


    # Final Evaluation
    # Final Inference
    y_preds, y_probs = inference(final_model, test_loader)
    
    print("\nInference completed on test set.")
    print("Predictions shape:", y_preds.shape)
    print("Probabilities shape:", y_probs.shape)

    # Final Inference
    y_preds, y_probs = inference(final_model, test_loader)
    
    print("\nInference completed on test set.")
    print("Predictions shape:", y_preds.shape)
    print("Probabilities shape:", y_probs.shape)
    
    # Create submission
    submission = test_df.copy()
    submission["prediction"] = y_preds
    
    submission.to_csv("submission.csv", index=False)
    
    print("[INFO] Submission saved: submission.csv")
    
    # Save model
    torch.save(final_model.state_dict(), "final_production_model.pth")
