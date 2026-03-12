
from __future__ import annotations

import io
from pathlib import Path
import uuid
import zipfile
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.processor import encode_image, process_image

app = FastAPI(
    title="Bottle Date Enhancer",
    description=(
        "Микросервис для подготовки фото бутылок под лучшее чтение дат, сроков и цифр VLM-моделью. "
        "Сервис заточен под слабую печать, мелкие цифры и блики: находит вероятную область даты/кода, "
        "обрезает фон и возвращает набор картинок с вариантами обработанного crop."
    ),
    version="1.6.0",
)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}

MAX_FILE_SIZE_MB = 20
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_debug_roi_image(
    oriented_bgr,
    bottle_box: dict[str, int],
    crop_box: dict[str, int],
    bottle_found: bool,
    crop_found: bool,
):
    overlay = oriented_bgr.copy()
    if bottle_found:
        cv2.rectangle(
            overlay,
            (int(bottle_box["x1"]), int(bottle_box["y1"])),
            (int(bottle_box["x2"]), int(bottle_box["y2"])),
            (255, 160, 0),
            3,
        )
    if crop_found:
        cv2.rectangle(
            overlay,
            (int(crop_box["x1"]), int(crop_box["y1"])),
            (int(crop_box["x2"]), int(crop_box["y2"])),
            (0, 255, 0),
            3,
        )
    return overlay


def build_output_images(result) -> list[tuple[str, str, bytes]]:
    debug_roi_bgr = build_debug_roi_image(
        result.oriented_bgr,
        result.metadata.bottle_box,
        result.metadata.crop_box,
        result.metadata.bottle_found,
        result.metadata.crop_found,
    )
    return [
        ("crop_preview.jpg", "image/jpeg", encode_image(result.crop_bgr, ext=".jpg", quality=95)),
        ("improved.jpg", "image/jpeg", encode_image(result.improved_bgr, ext=".jpg", quality=96)),
        ("bw.png", "image/png", encode_image(result.bw, ext=".png")),
        ("high_contrast.jpg", "image/jpeg", encode_image(result.high_contrast, ext=".jpg", quality=98)),
        ("debug_roi.jpg", "image/jpeg", encode_image(debug_roi_bgr, ext=".jpg", quality=95)),
    ]


def save_output_images(images: list[tuple[str, str, bytes]]) -> str:
    batch_id = uuid.uuid4().hex
    batch_dir = OUTPUTS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    for filename, _, payload in images:
        (batch_dir / filename).write_bytes(payload)
    return batch_id


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process")
async def process_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Фото бутылки"),
    response_format: Literal["zip", "json_links"] = Query(
        default="json_links",
        description="Вернуть zip-архив или JSON-массив ссылок на изображения",
    ),
    crop_padding_ratio: float = Query(
        default=0.08,
        ge=0.0,
        le=0.25,
        description="Насколько расширять найденную область даты/маркировки",
    ),
    detector_backend: Literal["craft", "heuristic"] = Query(
        default="craft",
        description="Какой детектор использовать для поиска даты/кода",
    ),
):
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Неподдерживаемый тип файла: {file.content_type}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} MB")

    try:
        result = process_image(
            content,
            crop_padding_ratio=crop_padding_ratio,
            detector_backend=detector_backend,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {exc}") from exc

    images = build_output_images(result)

    if response_format == "json_links":
        batch_id = save_output_images(images)
        base_url = str(request.base_url).rstrip("/")
        return [f"{base_url}/outputs/{batch_id}/{filename}" for filename, _, _ in images]

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, _, payload in images:
            zf.writestr(filename, payload)

    archive.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="bottle_outputs.zip"'}
    return StreamingResponse(archive, media_type="application/zip", headers=headers)
