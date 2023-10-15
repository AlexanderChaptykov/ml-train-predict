from pathlib import Path

from train import Predict, read_from_gc


def main(gc_path="gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
         model_dir="artifacts",
         out_path=Path("output/results.csv")):
    df = read_from_gc(gc_path)

    predictor = Predict.from_path(model_dir)

    df["Adopted_prediction"] = predictor(df)

    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path)


if __name__ == "__main__":
    main()
