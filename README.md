# Лабораторная работа 6  
**Тема:** Хранение и применение модели с использованием внешнего источника данных (Redis)

## Цель работы
Освоить полный цикл работы с моделью машинного обучения в Spark:  
- загрузка данных из внешнего источника (Redis);  
- обучение модели и сохранение её в файловую систему;  
- упаковка модели в дистрибутив (zip);  
- применение модели для новых данных с записью результатов обратно в Redis;  
- фиксация протокола выполнения через служебные ключи Redis.

## Ход работы

### Сборка Docker-образа
Собран собственный контейнер с Python 3.12, Java 21, Spark ML и утилитами для работы с Redis.

```bash
sudo docker compose build model
```

### Redis
```
sudo docker compose up -d redis
```
### Загрузка обучающих данных
```
sudo docker compose run --rm model \
  python scripts/push_to_redis.py \
    --csv data/train.csv \
    --key lab6:input \
    --flush \
    --has-label

sudo docker exec -it lab6-redis redis-cli LLEN lab6:input
```

### Обучение модели
Модель: пайплайн `Tokenizer → HashingTF → IDF → LogisticRegression.`
```
sudo docker compose run --rm model \
  python scripts/train.py \
    --in-key lab6:input \
    --model-dir models/spam_clf
```

### Результат 
```
TRAIN OK: {
  'rows_total': 20, 'rows_train': 15, 'rows_test': 5,
  'metrics': {'auc': 0.9167, 'f1': 0.781, 'accuracy': 0.8},
  'model_dir': 'models/spam_clf',
  'train_time_sec': 3.606
}

```
### Упаковка модели 
```
sudo docker compose run --rm model \
  python scripts/pack_model.py \
    --model-dir models/spam_clf \
    --out dist/spam_clf.zip

ls -lh dist/
```

### Загрузка данных для инференса
```
sudo docker compose run --rm model \
  python scripts/push_to_redis.py \
    --csv data/infer.csv \
    --key lab6:input \
    --flush
```
### Применение модели
```
sudo docker compose run --rm model \
  python scripts/predict.py \
    --in-key lab6:input \
    --out-key lab6:predictions \
    --model-dir models/spam_clf

PREDICT OK: {
  'rows': 4,
  'out_key': 'lab6:predictions',
  'model_dir': 'models/spam_clf',
  'predict_time_sec': 4.043
}
```
### Проверка результатов
```
sudo docker exec -it lab6-redis redis-cli LRANGE lab6:predictions 0 -1
1) {"id": "101", "pred": 1, "prob": 0.9798, "ts": "..."}
2) {"id": "102", "pred": 1, "prob": 0.9999, "ts": "..."}
3) {"id": "103", "pred": 0, "prob": 0.3982, "ts": "..."}
4) {"id": "104", "pred": 1, "prob": 0.9992, "ts": "..."}
```
