import dataclasses
import pickle
from pathlib import Path

import gcsfs
import joblib
import pandas as pd
from category_encoders import CountEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

target_col = "Adopted"
target2int = {"Yes": 1, "No": 0}


def read_from_gc(path):
    fs = gcsfs.GCSFileSystem(project="my-project")
    with fs.open(path) as f:
        df = pd.read_csv(f)
    return df


def get_xy(df, test=False):
    if test:
        return df.drop(target_col, axis=1), df[target_col]
    return df.drop(target_col, axis=1), df[target_col].map(target2int)


@dataclasses.dataclass
class Predict:
    pipe: Pipeline
    model: XGBClassifier
    cols = [
        "Type",
        "Age",
        "Breed1",
        "Gender",
        "Color1",
        "Color2",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
        "Fee",
        "PhotoAmt",
    ]
    int2target = dict(zip(target2int.values(), target2int.keys()))
    model_file = "model"
    pipe_file = "pipe.pkl"

    def __call__(self, x: pd.DataFrame):
        x = x[self.cols]
        x = self.pipe.transform(x)
        x = self.model.predict(x)
        return [self.int2target[_] for _ in x]

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        joblib.dump(self.pipe, model_dir / self.pipe_file)
        pickle.dump(self.model, open(model_dir / self.model_file, "wb"))

    @classmethod
    def from_path(cls, model_dir):
        model_dir = Path(model_dir)
        pipe = joblib.load(model_dir / cls.pipe_file)
        model = pickle.load(open(model_dir / cls.model_file, "rb"))
        return cls(pipe, model)


def log_metrics(y_true, y_pred):
    print("F1 Score", f1_score(y_true, y_pred, pos_label="Yes"))
    print("Accuracy", accuracy_score(y_true, y_pred))
    print("Recall", recall_score(y_true, y_pred, pos_label="Yes"))


def main(
    gc_path="gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
    out_path="artifacts",
):
    df = read_from_gc(gc_path)

    train, val_test = train_test_split(df, train_size=0.6, stratify=df[target_col])
    val, test = train_test_split(
        val_test, train_size=0.5, stratify=val_test[target_col]
    )

    train, train_y = get_xy(train)
    val, val_y = get_xy(val)
    test, test_y = get_xy(test, test=True)

    pipe_preprocess = Pipeline(
        [
            ("count", CountEncoder(cols=["Breed1"])),
            (
                "ordinal",
                OrdinalEncoder(
                    mapping=[
                        {
                            "col": "MaturitySize",
                            "mapping": {None: 0, "Small": 1, "Medium": 2, "Large": 3},
                        },
                        {
                            "col": "FurLength",
                            "mapping": {None: 0, "Short": 1, "Medium": 2, "Long": 3},
                        },
                        {
                            "col": "Health",
                            "mapping": {
                                None: 0,
                                "Serious Injury": 1,
                                "Minor Injury": 2,
                                "Healthy": 3,
                            },
                        },
                    ]
                ),
            ),
            (
                "ohe",
                OneHotEncoder(
                    cols=[
                        "Type",
                        "Gender",
                        "Color1",
                        "Color2",
                        "Vaccinated",
                        "Sterilized",
                    ]
                ),
            ),
        ]
    )
    model = XGBClassifier()

    train = pipe_preprocess.fit_transform(train)
    val = pipe_preprocess.transform(val)

    model.fit(train, train_y, eval_set=[(val, val_y)], early_stopping_rounds=5)

    predictor = Predict(pipe_preprocess, model)

    pred = predictor(test)

    log_metrics(test_y, pred)

    predictor.save(out_path)


if __name__ == "__main__":
    main()
