import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

# ── HTML 서빙 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ── Claude 프록시 ──
@app.post("/api/analyze")
async def analyze(request: Request):
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "서버에 ANTHROPIC_API_KEY가 설정되지 않았어요"}, status_code=500)
    body = await request.json()
    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json=body,
        )
    return JSONResponse(content=res.json(), status_code=res.status_code)

# ── Gemini 업로드 시작 ──
@app.post("/api/gemini/upload/start")
async def gemini_upload_start(request: Request):
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "서버에 GEMINI_API_KEY가 설정되지 않았어요"}, status_code=500)
    body = await request.body()
    headers_in = dict(request.headers)
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            f"https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=resumable&key={GEMINI_API_KEY}",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": headers_in.get("x-upload-content-length", "0"),
                "X-Goog-Upload-Header-Content-Type": headers_in.get("x-upload-content-type", "video/mp4"),
            },
            content=body,
        )
    upload_url = res.headers.get("x-goog-upload-url", "")
    return JSONResponse({"uploadUrl": upload_url, "status": res.status_code})

# ── Gemini 파일 업로드 ──
@app.post("/api/gemini/upload/data")
async def gemini_upload_data(request: Request):
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "서버에 GEMINI_API_KEY가 설정되지 않았어요"}, status_code=500)
    headers_in = dict(request.headers)
    upload_url = headers_in.get("x-upload-url", "")
    if not upload_url:
        return JSONResponse({"error": "x-upload-url 헤더가 없어요"}, status_code=400)
    body = await request.body()
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(
            upload_url,
            headers={
                "Content-Length": str(len(body)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=body,
        )
    return JSONResponse(content=res.json(), status_code=res.status_code)

# ── Gemini 파일 상태 조회 ──
@app.get("/api/gemini/files/{file_name}")
async def gemini_file_status(file_name: str):
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "서버에 GEMINI_API_KEY가 설정되지 않았어요"}, status_code=500)
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/files/{file_name}?key={GEMINI_API_KEY}"
        )
    return JSONResponse(content=res.json(), status_code=res.status_code)

# ── 사용 가능한 Gemini 모델 목록 조회 ──
@app.get("/api/gemini/models")
async def gemini_models():
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY 없음"}, status_code=500)
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        )
    return JSONResponse(content=res.json(), status_code=res.status_code)

# ── Gemini 영상 분석 ──
GEMINI_VIDEO_MODELS = [
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
]

@app.post("/api/gemini/analyze")
async def gemini_analyze(request: Request):
    if not GEMINI_API_KEY:
        return JSONResponse({"error": "서버에 GEMINI_API_KEY가 설정되지 않았어요"}, status_code=500)
    body = await request.json()

    # 사용 가능한 모델을 순서대로 시도
    async with httpx.AsyncClient(timeout=180.0) as client:
        # 먼저 모델 목록 조회
        models_res = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        )
        available = set()
        if models_res.status_code == 200:
            models_data = models_res.json()
            for m in models_data.get("models", []):
                name = m.get("name", "").replace("models/", "")
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    available.add(name)

        # 목록에서 첫 번째 사용 가능한 모델 선택
        chosen = next((m for m in GEMINI_VIDEO_MODELS if m in available), None)
        if not chosen:
            # 목록 조회 실패 시 기본값
            chosen = "gemini-1.5-flash-002"

        print(f"[Gemini] 선택된 모델: {chosen}, 사용 가능: {available}")

        res = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{chosen}:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=body,
        )
    return JSONResponse(content=res.json(), status_code=res.status_code)
