"""Machine learning for Singapore Mahjong.

Pipeline:
    datagen.py   -- simulate games, dump labeled decision records
    train.py     -- fit deal-in / win-probability models (needs sklearn)
    model.py     -- pure-Python inference from exported JSON weights
    features.py  -- feature extraction shared by all three

Only train.py needs numpy/scikit-learn (`pip install -e ".[ml]"`).
The engine, agents, and server load trained models with zero extra
dependencies: weights are exported as JSON and evaluated in pure Python.
"""
