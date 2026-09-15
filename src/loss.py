# ========== LOSS FUNCTION ==============   
class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(samples_per_cls)
        
        self.register_buffer(
            'class_weights', 
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        elif targets.ndim == 2 and targets.shape[1] > 1:
            targets = targets.argmax(dim=1)
            
        class_weights = self.class_weights.to(logits.device)
    
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction='none'
        )
    
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * ce_loss
    
        return loss.mean()
