import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, accuracy_score, recall_score

from train import Predict, read_from_gc, target_col, WrongDataset

model = Predict.from_path("artifacts")
gc_df = read_from_gc(
    "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
)


def test_raise_wrong_dataset():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    with pytest.raises(WrongDataset):
        model(df)


def test_metrics():
    pred = model(gc_df)
    assert f1_score(gc_df[target_col], pred, pos_label="Yes") > 0.82
    assert accuracy_score(gc_df[target_col], pred) > 0.73
    assert recall_score(gc_df[target_col], pred, pos_label="Yes") > 0.90
