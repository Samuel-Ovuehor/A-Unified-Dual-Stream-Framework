# =========================================
# DATA SAMPLER
# =========================================

def get_sampler(df):
    counts = Counter(df["diagnosis"])
    weights = {c: 1.0/v for c, v in counts.items()}
    sample_weights = [weights[l] for l in df["diagnosis"]]

    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )
