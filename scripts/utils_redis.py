import json
import os
from typing import Any, Dict, List, Optional

import redis


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
) -> redis.Redis:
    host = host or os.getenv("REDIS_HOST", "localhost")
    port = int(port or os.getenv("REDIS_PORT", "6379"))
    db = int(db or os.getenv("REDIS_DB", "0"))
    password = password or os.getenv("REDIS_PASSWORD", None)
    return redis.Redis(
        host=host, port=port, db=db, password=password, decode_responses=True
    )


def json_dumps_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def json_loads_safe(s: str) -> Dict[str, Any]:
    return json.loads(s)


def read_list_json(r: redis.Redis, key: str) -> List[Dict[str, Any]]:
    rows = r.lrange(key, 0, -1) or []
    out: List[Dict[str, Any]] = []
    for s in rows:
        try:
            out.append(json_loads_safe(s))
        except Exception as e:
            xadd_status(
                r, "lab6:status", stage="read", ok=0, msg={"key": key, "error": str(e)}
            )
    return out


def rpush_json(r: redis.Redis, key: str, item: Dict[str, Any]) -> None:
    r.rpush(key, json_dumps_safe(item))


def xadd_status(
    r: redis.Redis,
    stream_key: str,
    stage: str,
    ok: int,
    msg: Dict[str, Any],
) -> None:
    payload = {
        "stage": stage,
        "ok": str(int(ok)),
        "msg": json_dumps_safe(msg),
    }
    r.xadd(stream_key, payload)
