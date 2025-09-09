import argparse
import time
from typing import Any, Dict, List

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from utils_redis import (get_redis_client, read_list_json, rpush_json,
                         xadd_status)


def to_spark_df_unlabeled(spark: SparkSession, rows: List[Dict[str, Any]]):
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("text", StringType(), False),
        ]
    )
    normalized = []
    for r in rows:
        if "id" in r and "text" in r and "label" not in r:
            normalized.append({"id": str(r["id"]), "text": str(r["text"])})
        elif "id" in r and "text" in r:
            normalized.append({"id": str(r["id"]), "text": str(r["text"])})
    return spark.createDataFrame(normalized, schema=schema)


def main():
    parser = argparse.ArgumentParser(
        description="Predict with saved PipelineModel and write results to Redis"
    )
    parser.add_argument(
        "--in-key", default="lab6:input", help="Redis List with rows to predict"
    )
    parser.add_argument(
        "--out-key", default="lab6:predictions", help="Redis List for JSON predictions"
    )
    parser.add_argument(
        "--status-key", default="lab6:status", help="Redis Stream for logs"
    )
    parser.add_argument(
        "--model-dir", default="models/spam_clf", help="Path to saved PipelineModel"
    )
    args = parser.parse_args()

    r = get_redis_client()
    rows = read_list_json(r, args.in_key)
    if not rows:
        xadd_status(
            r,
            args.status_key,
            "predict",
            0,
            {"error": "empty_input", "in_key": args.in_key},
        )
        raise SystemExit("No rows to predict")

    spark = SparkSession.builder.appName("Lab6Predict").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = to_spark_df_unlabeled(spark, rows)
    df = df.withColumn("text", F.regexp_replace(F.col("text"), r"\s+", " "))

    t0 = time.time()
    model = PipelineModel.load(args.model_dir)
    pred = model.transform(df)

    def prob1(vec):
        return float(vec[1])

    prob1_udf = F.udf(prob1)
    out = pred.withColumn("prob", prob1_udf(F.col("probability"))).select(
        "id", "prediction", "prob"
    )

    cnt = 0
    for row in out.collect():
        item = {
            "id": row["id"],
            "pred": int(row["prediction"]),
            "prob": float(row["prob"]),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rpush_json(r, args.out_key, item)
        cnt += 1

    took = time.time() - t0
    msg = {
        "rows": cnt,
        "out_key": args.out_key,
        "model_dir": args.model_dir,
        "predict_time_sec": round(took, 3),
    }
    xadd_status(r, args.status_key, "predict", 1, msg)
    print("PREDICT OK:", msg)

    spark.stop()


if __name__ == "__main__":
    main()
