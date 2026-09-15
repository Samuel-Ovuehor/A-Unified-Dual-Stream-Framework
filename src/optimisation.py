# =========================================
# 4. HARMONY SEARCH
# =========================================

def harmony_search(train_ds, val_ds):

    print("\n--- Harmony Search Hyperparameter Optimization ---")

    harmony_memory = []

    # Initialize Memory
    for _ in range(CFG.HMS_TRIALS):
        harmony_memory.append({
            "lr": 10**random.uniform(-5, -3.5),
            "bs": random.choice([16, 32]),
            "f1": 0.0
        })

    for i in range(CFG.HMS_TRIALS):

        cfg = harmony_memory[i]

        loader = DataLoader(
            train_ds,
            batch_size=cfg['bs'],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        v_loader = DataLoader(
            val_ds,
            batch_size=cfg['bs'],
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

        model = DualStreamModel().to(CFG.DEVICE)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['lr']
        )

        counts = (
            train_ds.df["diagnosis"]
            .value_counts()
            .sort_index()
            .values
        )

        criterion = CBFocalLoss(
            samples_per_cls=counts,
            beta=0.9999,
            gamma=2.5
        ).to(CFG.DEVICE)

        h_pbar = tqdm(
            range(CFG.HMS_EPOCHS),
            desc=f"Harmony Trial {i+1}",
            leave=True
        )

        for epoch in h_pbar:

            model.train()
            running_loss = 0.0

            batch_pbar = tqdm(
                loader,
                desc=f" Epoch {epoch+1}",
                leave=False
            )

            for x, y in batch_pbar:

                x, y = x.to(
                    CFG.DEVICE,
                    non_blocking=True
                ), y.to(
                    CFG.DEVICE,
                    non_blocking=True
                )

                optimizer.zero_grad()

                loss = criterion(model(x), y)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

        metrics, _, _, _ = validate(
            model,
            v_loader
        )

        cfg['f1'] = metrics['F1']

        print(
            f"[TRIAL {i+1} COMPLETE] "
            f"LR: {cfg['lr']:.2e} | "
            f"BS: {cfg['bs']} | "
            f"F1: {cfg['f1']:.4f}"
        )

        del model, optimizer, loader, v_loader
        torch.cuda.empty_cache()

    best_cfg = max(
        harmony_memory,
        key=lambda x: x['f1']
    )

    return best_cfg
