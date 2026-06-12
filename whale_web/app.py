"""
AI护鲸使者 Web 演示系统 - Flask + ONNX Runtime

启动方式:
    cd whale_web
    python app.py

浏览器访问:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.onnx_inference import WhaleONNXPredictor


def resolve_project_path(path_value: str | Path) -> Path:
    """
    作用:
        将环境变量中的相对路径解析为项目根目录下的绝对路径。
    输入:
        path_value: 字符串或 Path 路径。
    输出:
        绝对 Path；如果输入已经是绝对路径则原样返回。
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "final_model_04"
ARTIFACT_DIR = resolve_project_path(os.getenv("WHALE_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
USE_ARTIFACT = os.getenv("WHALE_USE_ARTIFACT", "1") != "0"
ONNX_MODEL_PATH = resolve_project_path(os.getenv("WHALE_ONNX_PATH", PROJECT_ROOT / "whale_model.onnx"))
CLASS_MAP_PATH = resolve_project_path(os.getenv("WHALE_CLASS_MAP_PATH", PROJECT_ROOT / "outputs" / "class_to_idx.json"))
IMAGE_SIZE = int(os.getenv("WHALE_IMAGE_SIZE", "512"))
CONFIDENCE_THRESHOLD = float(os.getenv("WHALE_CONFIDENCE_THRESHOLD", "0.5"))

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

predictor: Optional[WhaleONNXPredictor] = None
predictor_error = ""


def create_predictor() -> Tuple[Optional[WhaleONNXPredictor], str]:
    """
    服务启动时尝试加载 ONNX 模型。
    如果模型文件或 onnxruntime 暂时不存在，Flask 仍然启动，便于前端展示健康状态。
    """
    errors = []
    if USE_ARTIFACT:
        try:
            model = WhaleONNXPredictor(
                artifact_dir=ARTIFACT_DIR,
                image_size=IMAGE_SIZE,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            return model, f"Artifact 模型加载成功: {ARTIFACT_DIR}"
        except Exception as exc:
            errors.append(f"artifact 加载失败: {exc}")

    try:
        model = WhaleONNXPredictor(
            onnx_path=ONNX_MODEL_PATH,
            class_map_path=CLASS_MAP_PATH,
            image_size=IMAGE_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        return model, "ONNX 模型加载成功。"
    except Exception as exc:
        errors.append(f"legacy 文件加载失败: {exc}")
        return None, "；".join(errors)


predictor, predictor_error = create_predictor()


def api_response(code: int = 200, msg: str = "success", data: Optional[Dict[str, Any]] = None, http_status: Optional[int] = None):
    """
    作用:
        生成统一 REST JSON 响应。
    输入:
        code: 业务状态码。
        msg: 响应消息。
        data: 响应数据字典。
        http_status: 可选 HTTP 状态码；为空时使用 code。
    输出:
        Flask response tuple: (jsonify(payload), http_status)。
    """
    return jsonify({"code": code, "msg": msg, "data": data or {}}), (http_status or code)


def is_allowed_file(filename: str) -> bool:
    """
    作用:
        判断上传文件后缀是否为允许的图片格式。
    输入:
        filename: 原始文件名。
    输出:
        True 表示允许上传，False 表示拒绝。
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in ALLOWED_EXTENSIONS


def validate_image_upload() -> Tuple[Optional[bytes], Optional[Tuple[Any, int]]]:
    """
    校验上传文件:
        1. 必须存在 image 字段。
        2. 后缀只能是 .png/.jpg/.jpeg。
        3. 请求体大小不能超过 10MB。
        4. 文件内容必须能被 PIL 识别为图片，避免伪造后缀。
    """
    if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
        return None, api_response(413, "图片大小不能超过 10MB。", http_status=413)

    if "image" not in request.files:
        return None, api_response(400, "请求中没有 image 文件字段。", http_status=400)

    file = request.files["image"]
    if file.filename == "":
        return None, api_response(400, "未选择图片文件。", http_status=400)

    # 后缀校验使用原始文件名，避免中文文件名被 secure_filename 清理后丢失扩展名。
    filename = file.filename
    if not is_allowed_file(filename):
        return None, api_response(400, "仅支持 .png、.jpg、.jpeg 图片。", http_status=400)

    image_bytes = file.read()
    if not image_bytes:
        return None, api_response(400, "上传文件为空。", http_status=400)
    if len(image_bytes) > MAX_FILE_SIZE:
        return None, api_response(413, "图片大小不能超过 10MB。", http_status=413)

    try:
        # verify 只校验图片结构，不完整解码；推理阶段会再次读取并转换为 RGB。
        Image.open(io.BytesIO(image_bytes)).verify()
    except (UnidentifiedImageError, OSError):
        return None, api_response(400, "文件内容不是有效图片。", http_status=400)

    return image_bytes, None


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_exc):
    """
    作用:
        处理 Flask 请求体超过 MAX_CONTENT_LENGTH 的异常。
    输入:
        _exc: RequestEntityTooLarge 异常对象。
    输出:
        统一 REST 错误响应。
    """
    return api_response(413, "图片大小不能超过 10MB。", http_status=413)


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    """
    作用:
        处理 Flask/Werkzeug 标准 HTTPException。
    输入:
        exc: HTTPException 异常对象。
    输出:
        统一 REST 错误响应。
    """
    return api_response(exc.code or 500, exc.description or "HTTP error", http_status=exc.code or 500)


@app.errorhandler(Exception)
def handle_unexpected_exception(exc: Exception):
    """
    作用:
        兜底处理未捕获异常，避免服务直接崩溃并记录日志。
    输入:
        exc: 任意异常对象。
    输出:
        统一 REST 500 错误响应。
    """
    app.logger.exception("Unhandled server error")
    return api_response(500, f"服务器内部错误: {exc}", http_status=500)


@app.route("/", methods=["GET"])
def index():
    """
    作用:
        返回 Web 单页演示界面。
    输入:
        无显式输入；使用全局 predictor 判断模型是否就绪。
    输出:
        渲染后的 HTML 页面。
    """
    return render_template("index.html", model_ready=predictor is not None)


@app.route("/health", methods=["GET"])
def health():
    """
    作用:
        返回服务健康状态和当前加载的模型版本信息。
    输入:
        无。
    输出:
        统一 REST JSON，包含 model_ready、artifact_dir、providers、confidence_threshold 等字段。
    """
    model_info = predictor.info() if predictor is not None else {}
    data = {
        "model_ready": predictor is not None,
        "engine": "onnxruntime",
        "onnx_runtime": predictor is not None,
        "artifact_dir": str(ARTIFACT_DIR),
        "use_artifact": USE_ARTIFACT,
        "model_name": model_info.get("model_name", ""),
        "version": model_info.get("version", ""),
        "num_classes": model_info.get("num_classes", 0),
        "onnx_model_path": str(ONNX_MODEL_PATH),
        "class_map_path": str(CLASS_MAP_PATH),
        "loaded_onnx_model_path": model_info.get("onnx_model_path", ""),
        "loaded_class_map_path": model_info.get("class_map_path", ""),
        "image_size": model_info.get("image_size", IMAGE_SIZE),
        "confidence_threshold": model_info.get("confidence_threshold", CONFIDENCE_THRESHOLD),
        "confidence_threshold_percent": round(float(model_info.get("confidence_threshold", CONFIDENCE_THRESHOLD)) * 100.0, 2),
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "providers": predictor.providers if predictor is not None else [],
        "metrics": model_info.get("metrics", {}),
        "message": "ready" if predictor is not None else predictor_error,
    }
    return api_response(200, "success", data=data)


@app.route("/predict", methods=["POST"])
def predict():
    """
    作用:
        接收上传图片，完成文件校验、ONNX 推理和结果格式化。
    输入:
        multipart/form-data 请求，文件字段名必须为 image。
    输出:
        统一 REST JSON，包含 top1、confidence、top3、不确定性判断和模型版本信息。
    """
    image_bytes, error_response = validate_image_upload()
    if error_response is not None:
        return error_response

    if predictor is None:
        return api_response(
            503,
            "ONNX 模型尚未就绪，请先导出 artifact 或 whale_model.onnx，并确认 onnxruntime 已安装。",
            data={"reason": predictor_error},
            http_status=503,
        )

    result = predictor.predict(image_bytes)
    model_info = predictor.info()
    data = {
        "top1": result["species"],
        "top1_display": result["species_display"],
        "species": result["species"],
        "species_display": result["species_display"],
        "confidence": result["confidence"],
        "confidence_percent": result["confidence_percent"],
        "confidence_threshold": result["confidence_threshold"],
        "confidence_threshold_percent": result["confidence_threshold_percent"],
        "is_uncertain": result["is_uncertain"],
        "decision": result["decision"],
        "message": result["message"],
        "top3": result["top3"],
        "providers": result.get("providers", []),
        "model": {
            "model_name": model_info.get("model_name", ""),
            "version": model_info.get("version", ""),
            "artifact_dir": model_info.get("artifact_dir", ""),
        },
    }
    return api_response(200, "success", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
