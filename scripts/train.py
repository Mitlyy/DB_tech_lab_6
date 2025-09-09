import argparse
import time
from typing import Any, Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.feature import IDF, HashingTF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from utils_redis import get_redis_client, read_list_json, xadd_status


def to_spark_df(spark: SparkSession, rows: List[Dict[str, Any]]):
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("text", StringType(), False),
            StructField("label", IntegerType(), True),
        ]
    )
    normalized = []
    for r in rows:
        if "id" in r and "text" in r and "label" in r:
            normalized.append(
                {"id": str(r["id"]), "text": str(r["text"]), "label": int(r["label"])}
            )
    return spark.createDataFrame(normalized, schema=schema)


def main():
    parser = argparse.ArgumentParser(
        description="Train minimal Spark ML model from Redis"
    )
    parser.add_argument(
        "--in-key", default="lab6:input", help="Redis List with labeled rows"
    )
    parser.add_argument(
        "--status-key", default="lab6:status", help="Redis Stream for logs"
    )
    parser.add_argument(
        "--model-dir", default="models/spam_clf", help="Path to save PipelineModel"
    )
    args = parser.parse_args()

    r = get_redis_client()
    rows = read_list_json(r, args.in_key)

    spark = SparkSession.builder.appName("Lab6Train").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = to_spark_df(spark, rows)
    n = df.count()
    if n < 10:
        xadd_status(
            r, args.status_key, "train", 0, {"error": "not_enough_rows", "rows": n}
        )
        raise SystemExit(f"Need at least 10 labeled rows, got {n}")

    df = df.withColumn(
        "text", F.regexp_replace(F.col("text"), r"\s+", " ").cast(StringType())
    )

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    if test_df.count() == 0:
        test_df = train_df.limit(1)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing = HashingTF(inputCol="words", outputCol="tf", numFeatures=1 << 14)
    idf = IDF(inputCol="tf", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)

    pipeline = Pipeline(stages=[tokenizer, hashing, idf, lr])

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0

    pred = model.transform(test_df)
    bce = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    mce_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    mce_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    auc = float(bce.evaluate(pred))
    f1 = float(mce_f1.evaluate(pred))
    acc = float(mce_acc.evaluate(pred))

    model_path = args.model_dir
    model.write().overwrite().save(model_path)

    msg = {
        "rows_total": n,
        "rows_train": train_df.count(),
        "rows_test": test_df.count(),
        "metrics": {
            "auc": round(auc, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
        },
        "model_dir": model_path,
        "train_time_sec": round(train_time, 3),
    }
    xadd_status(r, args.status_key, stage="train", ok=1, msg=msg)
    print("TRAIN OK:", msg)

    spark.stop()


if __name__ == "__main__":
    main()
