import argparse

import pandas as pd
from utils_redis import get_redis_client, rpush_json, xadd_status


def main():
    parser = argparse.ArgumentParser(description="Load CSV -> Redis List as JSON rows")
    parser.add_argument("--csv", required=True, help="Path to CSV (utf-8)")
    parser.add_argument(
        "--key", required=True, help="Redis List key (e.g., lab6:input)"
    )
    parser.add_argument(
        "--flush", action="store_true", help="Flush list key before push"
    )
    parser.add_argument(
        "--has-label", action="store_true", help="CSV has 'label' column"
    )
    args = parser.parse_args()

    r = get_redis_client()
    if args.flush:
        r.delete(args.key)

    df = pd.read_csv(args.csv)
    expected_cols = {"id", "text"} | ({"label"} if args.has_label else set())
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    cnt = 0
    for _, row in df.iterrows():
        item = {
            "id": str(row["id"]),
            "text": str(row["text"]),
        }
        if args.has_label:
            item["label"] = int(row["label"])
        rpush_json(r, args.key, item)
        cnt += 1

    xadd_status(
        r, "lab6:status", stage="push", ok=1, msg={"key": args.key, "rows": cnt}
    )
    print(f"OK: pushed {cnt} rows to {args.key}")


if __name__ == "__main__":
    main()
