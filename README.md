# Bottle Date Enhancer Service

FastAPI-микросервис для подготовки фото бутылок так, чтобы VLM / OCR лучше читали:

- даты розлива и сроки годности,
- лот-коды,
- короткие числовые маркировки,
- слабую лазерную или бледную печать по пластику / стеклу.

Сервис заточен именно под кейс **мелкие цифры + блики + слабый контраст**.

## Что изменено в этой версии

Вместо старого эвристического детектора по умолчанию используется **CRAFT** через `pycrafter`.

Новый пайплайн:

1. сначала ищет саму бутылку и выравнивает кадр по её оси;
2. при необходимости доворачивает бутылку, чтобы верх был сверху;
3. подавляет влияние бликов для этапа детекта;
4. нормализует освещённость;
5. запускает CRAFT или эвристику уже на выровненной бутылке;
6. выбирает наиболее date-like ROI внутри бутылки;
7. делает tight crop вокруг кода / даты;
8. отдельно доворачивает сам текстовый crop для VLM / OCR;
9. сильно увеличивает найденный ROI;
10. по HTTP отдает либо zip-архив с картинками, либо JSON-массив ссылок на сгенерированные изображения для Dify.

## Почему это лучше для бутылок

На бутылках дата часто:

- маленькая,
- напечатана в 1–2 строки,
- находится не на этикетке, а на теле бутылки,
- частично теряется из-за бликов и неравномерного фона.

CRAFT лучше ловит такие небольшие текстовые зоны, чем простая морфология, а последующий tight crop и апскейл делают символы заметно крупнее для VLM.

Дополнительно сервис теперь лучше работает с повернутыми бутылками: `debug_roi.jpg` формируется уже из выровненного кадра, на котором отдельно отмечаются бутылка и найденная дата.

## Стек

- Python 3.11
- FastAPI
- OpenCV
- pycrafter / CRAFT
- ONNX Runtime

## API

### `GET /health`

Проверка, что сервис жив.

### `POST /process`

Параметры:

- `file` — изображение (`jpeg`, `png`, `webp`)
- `response_format` — `zip` или `json_links`, по умолчанию `json_links`
- `crop_padding_ratio` — запас вокруг найденной даты / кода, по умолчанию `0.08`
- `detector_backend` — `craft` или `heuristic`, по умолчанию `craft`

Ответ:

Если `response_format=zip`, сервис возвращает `application/zip` только с изображениями:

- `crop_preview.jpg`
- `improved.jpg`
- `bw.png`
- `high_contrast.jpg`
- `debug_roi.jpg`

Если `response_format=json_links`, сервис возвращает `application/json` с массивом абсолютных ссылок на те же изображения. Этот режим удобен для Dify HTTP Request node.

`crop_preview.jpg` это исходный crop из найденного ROI. Если ROI не удалось уверенно локализовать, для crop используется полный кадр как fallback.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

Сборка:

```bash
docker build -t bottle-text-service .
```

Запуск:

```bash
docker run --rm -p 8000:8000 bottle-text-service
```

Через `docker compose`:

```bash
docker compose up --build -d
```

Остановка:

```bash
docker compose down
```

Результаты запросов из режима `json_links` сохраняются на хост в папку `./outputs/`.

## Примеры запросов

Ответ-архив:

```bash
curl -X POST "http://localhost:8000/process?detector_backend=craft&response_format=zip" \
  -H "accept: application/zip" \
  -F "file=@/path/to/bottle.jpg" \
  --output bottle_outputs.zip
```

Ответ для Dify:

```bash
curl -X POST "http://localhost:8000/process?detector_backend=craft&response_format=json_links" \
  -H "accept: application/json" \
  -F "file=@/path/to/bottle.jpg"
```

## Полезные переменные окружения

Через env можно подкрутить CRAFT без правки кода:

- `BOTTLE_CRAFT_LONG_SIZE` — размер длинной стороны для инференса CRAFT, по умолчанию `1280`
- `BOTTLE_CRAFT_TEXT_THRESHOLD` — по умолчанию `0.42`
- `BOTTLE_CRAFT_LINK_THRESHOLD` — по умолчанию `0.18`
- `BOTTLE_CRAFT_LOW_TEXT` — по умолчанию `0.18`
- `BOTTLE_CRAFT_REFINER` — `0` или `1`, по умолчанию `0`
- `BOTTLE_ENABLE_GRABCUT` — `0` или `1`, по умолчанию `0`; включает медленный дополнительный foreground-pass для поиска бутылки

В `docker-compose.yml` эти переменные уже вынесены в `environment`, поэтому их можно менять без пересборки образа.

На CPU самый полезный компромисс по скорости обычно даёт `BOTTLE_CRAFT_LONG_SIZE=1280`. Если нужен более агрессивный детект слабой печати, можно попробовать `1400–1600`, но это заметно медленнее.

## Smoke test

Синтетический тест:

```bash
PYTHONPATH=. python tests/smoke_test.py
```

Проверка на реальном изображении:

```bash
BOTTLE_TEST_IMAGE=/absolute/path/to/photo.jpg PYTHONPATH=. python tests/smoke_test.py
```

В `tests/_out/` будут сохранены:

- `improved.jpg`
- `bw.png`
- `high_contrast.jpg`
- `debug_roi.jpg`
- `crop_preview.jpg`

`POST /process` возвращает тот же набор файлов, что и `smoke_test.py`, без `metadata.json`.

## Ограничения текущей версии

- На CPU CRAFT не быстрый: для больших фото время может быть заметным.
- Если на одном кадре много бутылок, сервис выберет **самую вероятную** дату/маркировку, а не все сразу.
- Для industrial-scale сценария следующим шагом стоит добавить отдельный endpoint с top-N candidate crops.
