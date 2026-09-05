"""Training package for Experiment A (Pneumonia Reliability Study).

Modules:
    reproducibility - seeding, determinism, environment capture, run records
    data            - dataset + protocol-compliant preprocessing/augmentation
    models          - timm model builder + two-stage freeze/unfreeze
    engine          - train/eval loops + metrics

Entry point: src/train.py (CLI orchestrator).
"""
