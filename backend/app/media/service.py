import base64
import html
import json
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

_OUTPUT_DIR = Path(settings.media_output_dir)
_JOB_FILE = _OUTPUT_DIR / "jobs.json"

_RUNWAY_IMAGE_RATIOS = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "4:3": "1104:832",
    "1:1": "960:960",
}

_STABILITY_IMAGE_RATIOS = {"16:9", "9:16", "4:3", "1:1", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21"}

_RUNWAY_SUCCESS = {"SUCCEEDED", "COMPLETED"}
_RUNWAY_FAILURE = {"FAILED", "CANCELLED", "CANCELED"}
_RUNWAY_PROCESSING = {"PENDING", "THROTTLED", "RUNNING"}


def _ensure_output_dir() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str, limit: int = 48) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip()).strip("-")
    return normalized[:limit] or "media"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_jobs() -> dict[str, Any]:
    _ensure_output_dir()
    if not _JOB_FILE.exists():
        return {}
    try:
        return json.loads(_JOB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_jobs(jobs: dict[str, Any]) -> None:
    _ensure_output_dir()
    _JOB_FILE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_job(job: dict[str, Any]) -> dict[str, Any]:
    jobs = _load_jobs()
    job["updated_at"] = _now_iso()
    jobs[job["job_id"]] = job
    _save_jobs(jobs)
    return job


async def get_media_job(job_id: str) -> dict[str, Any] | None:
    job = _load_jobs().get(job_id)
    if not job:
        return None
    return await refresh_media_job(job)


def _relative_url(path: Path) -> str:
    static_root = Path(settings.static_dir)
    return "/" + path.relative_to(static_root).as_posix()


def _guess_extension(content_type: str | None, fallback: str = "bin") -> str:
    if not content_type:
        return fallback
    base_type = content_type.split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_extension(base_type)
    if guessed:
        return guessed.lstrip(".")
    if base_type == "image/jpg":
        return "jpg"
    return fallback


def _write_text_file(prefix: str, suffix: str, content: str) -> str:
    _ensure_output_dir()
    file_name = f"{prefix}-{uuid.uuid4().hex[:8]}.{suffix}"
    output_path = _OUTPUT_DIR / file_name
    output_path.write_text(content, encoding="utf-8")
    return _relative_url(output_path)


def _write_binary_file(prefix: str, suffix: str, content: bytes) -> str:
    _ensure_output_dir()
    file_name = f"{prefix}-{uuid.uuid4().hex[:8]}.{suffix}"
    output_path = _OUTPUT_DIR / file_name
    output_path.write_bytes(content)
    return _relative_url(output_path)


def _write_svg_card(title: str, subtitle: str, prompt: str, accent: str) -> str:
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    wrapped_prompt = html.escape(prompt)
    lines: list[str] = []
    chunk = ""
    for ch in wrapped_prompt:
        chunk += ch
        if len(chunk) >= 22 and ch in {" ", "，", "。", "、", ",", ";", "；"}:
            lines.append(chunk.strip())
            chunk = ""
    if chunk.strip():
        lines.append(chunk.strip())
    preview = "".join(
        f'<text x="44" y="{180 + idx * 34}" font-size="24" fill="#F8FAFC">{line}</text>'
        for idx, line in enumerate(lines[:5])
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0F172A" />
      <stop offset="100%" stop-color="{accent}" />
    </linearGradient>
  </defs>
  <rect width="1280" height="720" fill="url(#bg)" rx="32" />
  <rect x="36" y="36" width="1208" height="648" rx="28" fill="rgba(15,23,42,0.35)" stroke="rgba(255,255,255,0.18)" />
  <text x="44" y="88" font-size="28" fill="#C7D2FE">AI 素材工作台</text>
  <text x="44" y="136" font-size="48" font-weight="700" fill="#FFFFFF">{safe_title}</text>
  <text x="44" y="180" font-size="24" fill="#CBD5E1">{safe_subtitle}</text>
  {preview}
</svg>'''
    return _write_text_file(_slugify(title), "svg", svg)


def _build_demo_storyboard(prompt: str) -> str:
    return "\n".join(
        [
            "镜头 1：展示课堂主体与核心人物，营造学习氛围。",
            "镜头 2：突出教学主题与活动亮点，加入品牌文案。",
            "镜头 3：补充节奏感转场与结尾 CTA，引导报名或参与。",
            f"创作提示：{prompt}",
        ]
    )


def _runway_ratio(aspect_ratio: str, *, is_video: bool) -> str:
    normalized = (aspect_ratio or "").strip()
    if normalized in _RUNWAY_IMAGE_RATIOS.values():
        return normalized
    if normalized in _RUNWAY_IMAGE_RATIOS:
        return _RUNWAY_IMAGE_RATIOS[normalized]
    return settings.runway_video_ratio if is_video else settings.runway_image_ratio


def _stability_ratio(aspect_ratio: str) -> str:
    normalized = (aspect_ratio or "").strip()
    if normalized in _STABILITY_IMAGE_RATIOS:
        return normalized
    reverse_map = {value: key for key, value in _RUNWAY_IMAGE_RATIOS.items()}
    return reverse_map.get(normalized, "16:9")


def _job_base(kind: str, prompt: str, provider: str, status: str = "completed") -> dict[str, Any]:
    return {
        "job_id": uuid.uuid4().hex,
        "kind": kind,
        "prompt": prompt,
        "provider": provider,
        "status": status,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _local_file_from_public_url(url: str | None) -> Path | None:
    if not url:
        return None
    stripped = url.strip()
    if not stripped.startswith("/"):
        return None
    candidate = Path(settings.static_dir) / stripped.lstrip("/")
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _as_data_uri_from_local_file(file_path: Path) -> str:
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    content = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{content}"


def _source_image_for_runway(source_image_url: str | None) -> str | None:
    if not source_image_url:
        return None
    stripped = source_image_url.strip()
    if not stripped:
        return None
    if stripped.startswith("https://") or stripped.startswith("runway://") or stripped.startswith("data:image/"):
        return stripped
    local_file = _local_file_from_public_url(stripped)
    if local_file:
        return _as_data_uri_from_local_file(local_file)
    return None


async def _download_to_local(asset_url: str, prefix: str, fallback_extension: str) -> str:
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        response = await client.get(asset_url)
        response.raise_for_status()
        extension = _guess_extension(response.headers.get("content-type"), fallback_extension)
        return _write_binary_file(prefix, extension, response.content)


async def _call_json_api(
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    method: str = "POST",
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        response = await client.request(method, url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


async def _call_multipart_api(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    files = {"none": (None, "")}
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        response = await client.post(url, data=data, files=files, headers=headers)
        response.raise_for_status()
        return response


def _runway_headers() -> dict[str, str]:
    api_key = settings.runway_api_key or settings.video_generation_api_key or settings.image_generation_api_key
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Runway-Version": settings.runway_api_version,
    }


def _stability_headers() -> dict[str, str]:
    api_key = settings.stability_api_key or settings.image_generation_api_key
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }


def _resolve_image_provider() -> str:
    provider = (settings.image_generation_provider or "demo").strip().lower()
    if provider == "demo":
        return "demo"
    if provider in {"stability", "stability_ai", "stable-diffusion", "stable_diffusion"}:
        return "stability"
    if provider == "runway":
        return "runway"
    return provider


def _resolve_video_provider() -> str:
    provider = (settings.video_generation_provider or "demo").strip().lower()
    if provider == "demo":
        return "demo"
    if provider == "runway":
        return "runway"
    return provider


def _demo_image_job(prompt: str, style: str, aspect_ratio: str) -> dict[str, Any]:
    preview_url = _write_svg_card(
        title="教学宣传图",
        subtitle=f"风格：{style} · 比例：{aspect_ratio}",
        prompt=prompt,
        accent="#7C3AED",
    )
    job = _job_base("image", prompt, "demo")
    job.update(
        {
            "image_url": preview_url,
            "preview_url": preview_url,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "message": "当前为演示模式，已生成宣传图预览。配置真实图片模型后可输出正式素材。",
        }
    )
    return _upsert_job(job)


async def _stability_image_job(prompt: str, style: str, aspect_ratio: str) -> dict[str, Any]:
    model_name = (settings.stability_image_model or "core").strip().lower()
    endpoint = f"{settings.stability_api_base_url.rstrip('/')}/v2beta/stable-image/generate/{model_name}"
    response = await _call_multipart_api(
        endpoint,
        {
            "prompt": f"{prompt}\n风格要求：{style}".strip(),
            "aspect_ratio": _stability_ratio(aspect_ratio),
            "output_format": settings.stability_output_format,
        },
        _stability_headers(),
    )
    image_url = _write_binary_file(
        _slugify("image"),
        settings.stability_output_format or _guess_extension(response.headers.get("content-type"), "png"),
        response.content,
    )
    job = _job_base("image", prompt, "stability")
    job.update(
        {
            "image_url": image_url,
            "preview_url": image_url,
            "style": style,
            "aspect_ratio": aspect_ratio,
            "message": f"Stability AI 图片生成完成，模型：{model_name}。",
            "provider_model": model_name,
        }
    )
    return _upsert_job(job)


async def _runway_image_job(prompt: str, style: str, aspect_ratio: str) -> dict[str, Any]:
    payload = {
        "model": settings.runway_image_model,
        "promptText": f"{prompt}\n风格要求：{style}".strip(),
        "ratio": _runway_ratio(aspect_ratio, is_video=False),
    }
    data = await _call_json_api(
        f"{settings.runway_api_base_url.rstrip('/')}/v1/text_to_image",
        payload,
        _runway_headers(),
    )
    provider_job_id = str(data.get("id") or data.get("task_id") or data.get("job_id") or "").strip()
    if not provider_job_id:
        raise ValueError("Runway 图片接口未返回任务 ID")
    job = _job_base("image", prompt, "runway", status="processing")
    job.update(
        {
            "style": style,
            "aspect_ratio": aspect_ratio,
            "provider_model": settings.runway_image_model,
            "provider_job_id": provider_job_id,
            "provider_status_url": f"{settings.runway_api_base_url.rstrip('/')}/v1/tasks/{provider_job_id}",
            "message": "Runway 图片任务已提交，正在生成中。",
        }
    )
    return _upsert_job(job)


async def create_image_job(prompt: str, style: str, aspect_ratio: str) -> dict[str, Any]:
    provider = _resolve_image_provider()
    if provider == "demo" or (provider == "stability" and not (settings.stability_api_key or settings.image_generation_api_key)):
        return _demo_image_job(prompt, style, aspect_ratio)
    if provider == "stability":
        return await _stability_image_job(prompt, style, aspect_ratio)
    if provider == "runway":
        if not (settings.runway_api_key or settings.image_generation_api_key):
            return _demo_image_job(prompt, style, aspect_ratio)
        return await _runway_image_job(prompt, style, aspect_ratio)
    if settings.image_generation_api_url and settings.image_generation_api_key:
        data = await _call_json_api(
            settings.image_generation_api_url,
            {
                "prompt": prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "model": settings.image_generation_model,
            },
            {"Authorization": f"Bearer {settings.image_generation_api_key}", "Content-Type": "application/json"},
        )
        image_url = data.get("image_url") or data.get("url") or data.get("data", {}).get("image_url")
        if not image_url:
            raise ValueError("图片接口未返回 image_url")
        job = _job_base("image", prompt, provider)
        job.update(
            {
                "image_url": image_url,
                "preview_url": image_url,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "provider_model": settings.image_generation_model,
                "message": data.get("message") or "图片生成完成。",
                "raw_response": data,
            }
        )
        return _upsert_job(job)
    return _demo_image_job(prompt, style, aspect_ratio)


def _demo_video_job(prompt: str, mode: str, duration_seconds: int, source_image_url: str | None) -> dict[str, Any]:
    preview_url = _write_svg_card(
        title="教学短视频",
        subtitle=f"模式：{mode} · 时长：{duration_seconds}s",
        prompt=prompt,
        accent="#0EA5E9",
    )
    storyboard = _build_demo_storyboard(prompt)
    storyboard_url = _write_text_file(_slugify(f"storyboard-{prompt}"), "txt", storyboard)
    job = _job_base("video", prompt, "demo", status="mock_ready")
    job.update(
        {
            "preview_url": preview_url,
            "poster_url": preview_url,
            "video_url": None,
            "storyboard": storyboard,
            "storyboard_url": storyboard_url,
            "mode": mode,
            "duration_seconds": duration_seconds,
            "source_image_url": source_image_url,
            "message": "当前为演示模式，已生成视频分镜和封面。配置真实视频接口后将输出真实视频。",
        }
    )
    return _upsert_job(job)


async def _runway_video_job(prompt: str, mode: str, duration_seconds: int, source_image_url: str | None) -> dict[str, Any]:
    endpoint = "/v1/text_to_video"
    payload: dict[str, Any] = {
        "model": settings.runway_video_model,
        "promptText": prompt,
        "duration": duration_seconds,
        "ratio": settings.runway_video_ratio,
    }
    if mode == "image-to-video":
        prompt_image = _source_image_for_runway(source_image_url)
        if not prompt_image:
            raise ValueError("图片转视频需要可访问的 HTTPS 图片、data URI，或先生成本地图片。")
        endpoint = "/v1/image_to_video"
        payload["promptImage"] = prompt_image
    data = await _call_json_api(
        f"{settings.runway_api_base_url.rstrip('/')}{endpoint}",
        payload,
        _runway_headers(),
    )
    provider_job_id = str(data.get("id") or data.get("task_id") or data.get("job_id") or "").strip()
    if not provider_job_id:
        raise ValueError("Runway 视频接口未返回任务 ID")
    job = _job_base("video", prompt, "runway", status="processing")
    job.update(
        {
            "mode": mode,
            "duration_seconds": duration_seconds,
            "source_image_url": source_image_url,
            "provider_model": settings.runway_video_model,
            "provider_job_id": provider_job_id,
            "provider_status_url": f"{settings.runway_api_base_url.rstrip('/')}/v1/tasks/{provider_job_id}",
            "message": "Runway 视频任务已提交，系统正在轮询生成结果。",
        }
    )
    return _upsert_job(job)


async def create_video_job(prompt: str, mode: str, duration_seconds: int, source_image_url: str | None) -> dict[str, Any]:
    provider = _resolve_video_provider()
    if provider == "demo" or (provider == "runway" and not (settings.runway_api_key or settings.video_generation_api_key)):
        return _demo_video_job(prompt, mode, duration_seconds, source_image_url)
    if provider == "runway":
        return await _runway_video_job(prompt, mode, duration_seconds, source_image_url)
    if settings.video_generation_api_url and settings.video_generation_api_key:
        data = await _call_json_api(
            settings.video_generation_api_url,
            {
                "prompt": prompt,
                "mode": mode,
                "duration_seconds": duration_seconds,
                "model": settings.video_generation_model,
                "source_image_url": source_image_url,
            },
            {"Authorization": f"Bearer {settings.video_generation_api_key}", "Content-Type": "application/json"},
        )
        job = _job_base("video", prompt, provider, status="processing")
        job.update(
            {
                "mode": mode,
                "duration_seconds": duration_seconds,
                "source_image_url": source_image_url,
                "provider_model": settings.video_generation_model,
                "provider_job_id": data.get("job_id") or data.get("task_id") or data.get("id"),
                "provider_status_url": data.get("status_url") or settings.video_generation_status_url,
                "video_url": data.get("video_url") or data.get("url"),
                "preview_url": data.get("preview_url") or data.get("poster_url"),
                "poster_url": data.get("poster_url") or data.get("preview_url"),
                "message": data.get("message") or "视频任务已创建。",
                "raw_response": data,
            }
        )
        if job.get("video_url"):
            job["status"] = "completed"
        return _upsert_job(job)
    return _demo_video_job(prompt, mode, duration_seconds, source_image_url)


async def _refresh_runway_job(job: dict[str, Any]) -> dict[str, Any]:
    provider_job_id = str(job.get("provider_job_id") or "").strip()
    if not provider_job_id:
        return job
    task = await _call_json_api(
        f"{settings.runway_api_base_url.rstrip('/')}/v1/tasks/{provider_job_id}",
        None,
        _runway_headers(),
        method="GET",
    )
    status = str(task.get("status") or "").upper()
    output = task.get("output") or []
    failure = task.get("failure") or task.get("error") or task.get("message")
    if status in _RUNWAY_SUCCESS:
        asset_url = output[0] if isinstance(output, list) and output else None
        if asset_url and job.get("kind") == "image":
            local_url = await _download_to_local(str(asset_url), _slugify("runway-image"), "png")
            job["image_url"] = local_url
            job["preview_url"] = local_url
        elif asset_url and job.get("kind") == "video":
            local_video_url = await _download_to_local(str(asset_url), _slugify("runway-video"), "mp4")
            job["video_url"] = local_video_url
            job["preview_url"] = local_video_url
            job["poster_url"] = local_video_url
        job["status"] = "completed"
        job["message"] = "Runway 素材生成完成，结果已转存到本地静态目录。"
    elif status in _RUNWAY_FAILURE:
        job["status"] = "failed"
        job["message"] = str(failure or "Runway 任务执行失败")
    elif status in _RUNWAY_PROCESSING:
        job["status"] = "processing"
        job["message"] = f"Runway 任务状态：{status.lower()}。"
    job["raw_status"] = task
    return _upsert_job(job)


async def _refresh_generic_video_job(job: dict[str, Any]) -> dict[str, Any]:
    status_url = str(job.get("provider_status_url") or "").strip()
    provider_job_id = str(job.get("provider_job_id") or "").strip()
    if not status_url:
        return job
    if "{job_id}" in status_url:
        status_url = status_url.replace("{job_id}", provider_job_id)
    data = await _call_json_api(
        status_url,
        None,
        {"Authorization": f"Bearer {settings.video_generation_api_key}", "Content-Type": "application/json"},
        method="GET",
    )
    normalized_status = str(data.get("status") or data.get("state") or job.get("status") or "processing").lower()
    if normalized_status in {"succeeded", "success", "done", "completed", "ready"}:
        job["status"] = "completed"
    elif normalized_status in {"failed", "error", "canceled", "cancelled"}:
        job["status"] = "failed"
    else:
        job["status"] = "processing"
    if data.get("video_url") or data.get("url"):
        job["video_url"] = data.get("video_url") or data.get("url")
    if data.get("preview_url") or data.get("poster_url"):
        job["preview_url"] = data.get("preview_url") or data.get("poster_url")
        job["poster_url"] = data.get("poster_url") or data.get("preview_url")
    job["raw_status"] = data
    return _upsert_job(job)


async def refresh_media_job(job: dict[str, Any]) -> dict[str, Any]:
    provider = str(job.get("provider") or "demo").strip().lower()
    if provider == "demo" or job.get("status") in {"completed", "failed", "mock_ready"}:
        return job
    if provider == "runway":
        return await _refresh_runway_job(job)
    return await _refresh_generic_video_job(job)
