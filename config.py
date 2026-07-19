CFG = {
    # Reproducibility
    'seed':         42,
    # Data
    'batch_size':   256,
    'num_classes':  100,
    'data_dir':     '/content/data',
    # Training protocol — identical for ALL models
    'epochs':       200,
    'lr':           0.1,
    'momentum':     0.9,
    'weight_decay': 5e-4,
    # Parameter budget
    'param_budget': 1_500_000,   # 1.5 M params target
    'param_tol':    0.05,        # ±5% tolerance
    # Persistence (Google Drive)
    'drive_base':   '/content/drive/MyDrive/COMP5329_Deep_Learning_As2',
}
