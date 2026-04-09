# Face Swap API (FastAPI)

A robust, GPU-accelerated face-swapping pipeline exposed as a FastAPI webservice. Designed to be run headlessly on Google Colab (using ONNX Runtime, InsightFace, and GFPGAN).

---

## How to Run on Google Colab

Follow these exact steps to deploy the API to a free Google Colab Tesla T4 GPU instance.

### Step 1: Clone and Setup
Open a new Colab Notebook, make sure the hardware accelerator is set to **T4 GPU**, and run this in the first cell:

```python
%cd /content
!git clone https://github.com/Prithwi0505/face_swap.git
%cd /content/face_swap
!bash setup_colab.sh
```

### Step 2: Restart Runtime (CRITICAL)
After the setup script finishes, you **must** restart the Python environment to load the correct dependencies.
* Go to the top menu: **Runtime → Restart runtime**.

### Step 3: Start the Server & Expose via Ngrok
Run this in a new cell. *(Make sure to replace `YOUR_NGROK_AUTH_TOKEN` with your actual token from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)).*

```python
%cd /content/face_swap

# 1. Start the FastAPI server in the background
!nohup uvicorn api:app --host 0.0.0.0 --port 7860 > /content/fastapi.log 2>&1 &

# Wait for server to boot and download models
import time
time.sleep(15)

# 2. Install ngrok via official apt
!curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
!echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
!sudo apt-get update || true
!sudo apt-get install -y ngrok

import time
import requests

# Kill any lingering ngrok tunnels
!pkill -f ngrok

# Start the native ngrok tunnel as a background job
!nohup ngrok http --authtoken YOUR_NGROK_AUTH_TOKEN --log=stdout 7860 > /content/ngrok.log 2>&1 &
time.sleep(4)

# Query ngrok's local management API to get the public URL
try:
    res = requests.get("http://localhost:4040/api/tunnels")
    public_url = res.json()["tunnels"][0]["public_url"]
    print(f"✅ FACE SWAP API IS ONLINE: {public_url}")
except Exception as e:
    print(f"❌ Failed to get Ngrok URL (check if your auth token is correct): {e}")
```

When startup is successful, the Colab log will print:
```
[DB] SQLite database ready at 'face_swap_requests.db'.
[API] Models pre-checked and ready.
[API] Face-swap API is ready.
✅ FACE SWAP API IS ONLINE: https://YOUR_NGROK_URL.ngrok-free.app
```

---

## Testing the API

Once your Colab prints the `✅ FACE SWAP API IS ONLINE` URL, you can send requests from your local terminal.

> **Windows users:** Use `curl.exe` instead of `curl` in PowerShell.

---

### 1. Swap Face in a Video

Send a face-swap request. When complete, the terminal **immediately prints a JSON response** with the request details and a download link.

**Windows (PowerShell):**
```powershell
curl.exe -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face" -F "source_image=@source.jpg" -F "target_video=@target.mp4"
```

**Mac/Linux:**
```bash
curl -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face" \
  -F "source_image=@source.jpg" \
  -F "target_video=@target.mp4"
```

**Response printed in terminal:**
```json
{
  "request_id": "895cdc77-f5b4-426d-8ecf-bc4c358e6b91",
  "timestamp": 63.802,
  "status": "completed",
  "request_type": "video",
  "created_at": "2026-04-09T06:35:15.597752+00:00",
  "download_url": "/download/895cdc77-f5b4-426d-8ecf-bc4c358e6b91",
  "message": "Face swap completed. Use the download_url to fetch your result file."
}
```

Then **download the result file** using the `request_id` from the response:

**Windows (PowerShell):**
```powershell
curl.exe "https://YOUR_NGROK_URL.ngrok-free.app/download/895cdc77-f5b4-426d-8ecf-bc4c358e6b91" --output my_new_video.mp4
```

**Mac/Linux:**
```bash
curl "https://YOUR_NGROK_URL.ngrok-free.app/download/895cdc77-f5b4-426d-8ecf-bc4c358e6b91" --output my_new_video.mp4
```

---

### 2. Swap Face in an Image (Image-to-Image)

**Windows (PowerShell):**
```powershell
curl.exe -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" -F "source_image=@source.jpg" -F "target_image=@target.jpg"
```

**Mac/Linux:**
```bash
curl -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg"
```

**Response printed in terminal:**
```json
{
  "request_id": "cbdae007-a377-42ca-b6b7-7552f49888c3",
  "timestamp": 4.417,
  "status": "completed",
  "request_type": "image",
  "created_at": "2026-04-09T05:30:18.899865+00:00",
  "download_url": "/download/cbdae007-a377-42ca-b6b7-7552f49888c3",
  "message": "Face swap completed. Use the download_url to fetch your result file."
}
```

Then **download the result image**:

**Windows (PowerShell):**
```powershell
curl.exe "https://YOUR_NGROK_URL.ngrok-free.app/download/cbdae007-a377-42ca-b6b7-7552f49888c3" --output my_swapped_image.jpg
```

---

## Request Tracking (SQLite Database)

Every face-swap request is automatically tracked in a SQLite database (`face_swap_requests.db`) on the server. The database is created automatically when the API starts — no setup needed.

### What is stored per request

| Field          | Description                                               |
|----------------|-----------------------------------------------------------|
| `request_id`   | Unique UUID assigned to every request                     |
| `status`       | `not_completed` → `in_progress` → `completed` / `failed`  |
| `timestamp`    | Time taken to process the request (in seconds)            |
| `request_type` | `image` or `video`                                        |
| `created_at`   | ISO-8601 UTC timestamp of when the request arrived        |

### Status lifecycle

```
Request arrives  →  not_completed
Pipeline starts  →  in_progress
Pipeline done    →  completed  (timestamp recorded)
Any error        →  failed     (timestamp recorded)
```

### Query the database via API

#### Get all requests (newest first)
```powershell
curl.exe -X GET "https://YOUR_NGROK_URL.ngrok-free.app/requests"
```

Example response:
```json
{
  "total": 2,
  "requests": [
    {
      "request_id": "895cdc77-f5b4-426d-8ecf-bc4c358e6b91",
      "timestamp": 63.802,
      "status": "completed",
      "request_type": "video",
      "created_at": "2026-04-09T06:35:15.597752+00:00"
    },
    {
      "request_id": "cbdae007-a377-42ca-b6b7-7552f49888c3",
      "timestamp": 4.417,
      "status": "completed",
      "request_type": "image",
      "created_at": "2026-04-09T05:30:18.899865+00:00"
    }
  ]
}
```

#### Get a specific request by ID
```powershell
curl.exe -X GET "https://YOUR_NGROK_URL.ngrok-free.app/request/895cdc77-f5b4-426d-8ecf-bc4c358e6b91"
```

---

## Using Python

```python
import requests

# Step 1: Trigger face swap — prints JSON response
with open("source.jpg", "rb") as src, open("target.mp4", "rb") as tgt:
    response = requests.post(
        "https://YOUR_NGROK_URL.ngrok-free.app/swap-face",
        files={"source_image": src, "target_video": tgt},
    )

data = response.json()
print(data)
# {
#   "request_id": "895cdc77-...",
#   "status": "completed",
#   "timestamp": 63.802,
#   "download_url": "/download/895cdc77-...",
#   ...
# }

# Step 2: Download the result file using the request_id
request_id = data["request_id"]
file_response = requests.get(
    f"https://YOUR_NGROK_URL.ngrok-free.app/download/{request_id}"
)

with open("my_new_video.mp4", "wb") as f:
    f.write(file_response.content)

print("Download complete!")
```

---

## API Endpoints Summary

| Method | Endpoint                    | Description                                        |
|--------|-----------------------------|----------------------------------------------------|
| `GET`  | `/health`                   | Health check — confirms API is running             |
| `POST` | `/swap-face`                | Swap face in a video — returns JSON with DB record |
| `POST` | `/swap-face-image`          | Swap face in an image — returns JSON with DB record|
| `GET`  | `/download/{request_id}`    | Download the result file for a completed request   |
| `GET`  | `/request/{request_id}`     | Get one specific request record by ID              |
| `GET`  | `/requests`                 | List all request records (newest first)            |

The interactive API docs (Swagger UI) are available at:
```
https://YOUR_NGROK_URL.ngrok-free.app/docs
```

---

## Server Log Tracing

On the Colab server side, each request prints its ID at the start:
```
[API] swap-face request_id=895cdc77-f5b4-426d-8ecf-bc4c358e6b91
[API] swap-face-image request_id=cbdae007-a377-42ca-b6b7-7552f49888c3
```
This lets you match a client-side request ID with the server logs for debugging.
