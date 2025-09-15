
import argparse
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def build_spark(app_name: str = "lab6-train-from-lab5") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )


def read_dataset(spark: SparkSession,
                 path: str,
                 fmt: str,
                 text_col: str,
                 label_col: str,
                 id_col: str):
    fmt = fmt.lower()
    if fmt == "parquet":
        df = spark.read.parquet(path)
    elif fmt == "csv":
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use csv or parquet.")

    cols = df.columns
    if id_col not in cols:
        df = df.withColumn("__row_id__", F.monotonically_increasing_id().cast("string"))
        id_col_use = "__row_id__"
    else:
        id_col_use = id_col

    if text_col not in cols:
        raise ValueError(f"Text column '{text_col}' not found in dataset. Columns: {cols}")

    has_label = (label_col in cols)

    df = df.withColumn("id", F.col(id_col_use).cast(StringType())) \
           .withColumn("text", F.col(text_col).cast(StringType()))

    if has_label:
        df = df.withColumn("label", F.col(label_col).cast(IntegerType()))
        df = df.where(F.col("label").isin(0, 1))
    else:
        df = df.withColumn("label", F.lit(None).cast(IntegerType()))

    return df.select("id", "text", "label"), has_label


def build_pipeline() -> Pipeline:
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    hashing_tf = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=1 << 18)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    return Pipeline(stages=[tokenizer, hashing_tf, idf, lr])


def evaluate(binary_df):
    out = {}
    try:
        auc = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label",
            metricName="areaUnderROC"
        ).evaluate(binary_df)
        out["auc"] = float(auc)
    except Exception:
        pass

    try:
        f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        ).evaluate(binary_df)
        out["f1"] = float(f1)
    except Exception:
        pass

    try:
        acc = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        ).evaluate(binary_df)
        out["accuracy"] = float(acc)
    except Exception:
        pass

    return out


def main():
    parser = argparse.ArgumentParser(description="Train spam classifier on Lab5 data (directly from files).")
    parser.add_argument("--input", required=True, help="Path to dataset produced/used in Lab5 (file or folder).")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Input data format.")
    parser.add_argument("--text-col", default="text", help="Name of the text column.")
    parser.add_argument("--label-col", default="label", help="Name of the label column (0/1).")
    parser.add_argument("--id-col", default="id", help="Name of the id column.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test share for randomSplit.")
    parser.add_argument("--model-dir", default="models/spam_clf", help="Where to save the fitted pipelineModel.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    spark = build_spark()

    t0 = time.time()
    df, has_label = read_dataset(
        spark, args.input, args.format, args.text_col, args.label_col, args.id_col
    )

    if not has_label:
        raise RuntimeError(
            "В датасете нет колонки метки. Для обучения нужна метка (0/1). "
            "Укажите правильное имя через --label-col."
        )

    test_size = args.test_size
    train_size = max(0.0, min(1.0, 1.0 - test_size))
    train_df, test_df = df.randomSplit([train_size, test_size], seed=args.seed)

    pipeline = build_pipeline()
    model = pipeline.fit(train_df)

    pred_df = model.transform(test_df).cache()
    metrics = evaluate(pred_df)

    model_dir = args.model_dir
    from pyspark.ml.util import MLWriter
    try:
        if os.path.isdir(model_dir):
            import shutil
            shutil.rmtree(model_dir)
    except Exception:
        pass
    model.write().overwrite().save(model_dir)

    t1 = time.time()
    result = {
        "rows_total": df.count(),
        "rows_train": train_df.count(),
        "rows_test": test_df.count(),
        "metrics": metrics,
        "model_dir": model_dir,
        "train_time_sec": round(t1 - t0, 3)
    }
    print("TRAIN_FROM_LAB5 OK:", json.dumps(result, ensure_ascii=False, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()

