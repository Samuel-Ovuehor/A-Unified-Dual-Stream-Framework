class DualStreamModel(nn.Module):
    """
    Global context + Local Edge features.
    Uses ConvNext-Tiny as a powerful, modern backbone.
    """
    def __init__(self, model_name="convnext_tiny"):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        ch = self.backbone.num_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.edge_stream = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(ch * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, CFG.NUM_CLASSES)
        )

    def forward(self, x):

        features = self.backbone.forward_features(x)

        g_feat = self.global_pool(features).flatten(1)

        e_feat = self.edge_stream(features).flatten(1)

        combined = torch.cat([g_feat, e_feat], dim=1)

        return self.classifier(combined)
