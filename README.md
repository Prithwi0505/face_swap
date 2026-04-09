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

Once your Colab prints the `✅ FACE SWAP API IS ONLINE` URL, you can send requests to it from your local computer's terminal.
*Note for Windows users: If you are using PowerShell, you must type `curl.exe` instead of `curl`!*

### 1. Swap Face in Video

**Mac/Linux:**
```bash
curl -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face" \
  -F "source_image=@my_source_face.jpg" \
  -F "target_video=@cool_video.mp4" \
  --output my_new_video.mp4
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face" -F "source_image=@my_source_face.jpg" -F "target_video=@cool_video.mp4" --output my_new_video.mp4
```

### 2. Swap Face in Image (Image-to-Image)

**Mac/Linux:**
```bash
curl -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" \
  -F "source_image=@my_source_face.jpg" \
  -F "target_image=@target_person.jpg" \
  --output my_swapped_face.jpg
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" -F "source_image=@my_source_face.jpg" -F "target_image=@target_person.jpg" --output my_swapped_face.jpg
```

---

## Request Tracking (SQLite Database)

Every face-swap request is automatically tracked in a SQLite database (`face_swap_requests.db`) on the server. No setup is needed — the database is created automatically when the API starts.

### What is stored per request

| Field          | Description                                              |
|----------------|----------------------------------------------------------|
| `request_id`   | Unique UUID assigned to every request                    |
| `status`       | `not_completed` → `in_progress` → `completed` / `failed` |
| `timestamp`    | Time taken to process the request (in seconds)          |
| `request_type` | `image` or `video`                                       |
| `created_at`   | ISO-8601 UTC timestamp of when the request arrived       |

### Status lifecycle

```
Request arrives  →  not_completed
Pipeline starts  →  in_progress
Pipeline done    →  completed  (timestamp recorded)
Any error        →  failed     (timestamp recorded)
```

### Reading the database via API endpoints

#### Get all requests (newest first)
```powershell
curl.exe -X GET "https://YOUR_NGROK_URL.ngrok-free.app/requests"
```

Example response:
```json
{
  "total": 1,
  "requests": [
    {
      "request_id": "a26ed03f-b66d-4634-88db-ab285455fd56",
      "timestamp": 65.983,
      "status": "completed",
      "request_type": "video",
      "created_at": "2026-04-09T05:15:26.410689+00:00"
    }
  ]
}
```

#### Get a specific request by ID
```powershell
curl.exe -X GET "https://YOUR_NGROK_URL.ngrok-free.app/request/YOUR_REQUEST_ID"
```

Example response:
```json
{
  "request_id": "a26ed03f-b66d-4634-88db-ab285455fd56",
  "timestamp": 65.983,
  "status": "completed",
  "request_type": "video",
  "created_at": "2026-04-09T05:15:26.410689+00:00"
}
```

### Reading request info from response headers

Every successful face-swap response includes the database record directly in the HTTP response headers. Add `-i` to your curl command to see them:

**Windows (PowerShell):**
```powershell
curl.exe -i -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face" -F "source_image=@source.jpg" -F "target_video=@target.mp4" --output my_new_video.mp4
```

The terminal will print headers before saving the file:
```
HTTP/1.1 200 OK
X-Request-ID:   a26ed03f-b66d-4634-88db-ab285455fd56
X-Status:        completed
X-Time-Taken:    65.983s
X-Request-Type:  video
X-Created-At:    2026-04-09T05:15:26.410689+00:00
```

### Reading request info via Python

```python
import requests

with open("source.jpg", "rb") as src, open("target.mp4", "rb") as tgt:
    response = requests.post(
        "https://YOUR_NGROK_URL.ngrok-free.app/swap-face",
        files={"source_image": src, "target_video": tgt},
    )

# Save the output video
with open("my_new_video.mp4", "wb") as f:
    f.write(response.content)

# Read the DB record from headers
print("Request ID:  ", response.headers["X-Request-ID"])
print("Status:      ", response.headers["X-Status"])
print("Time taken:  ", response.headers["X-Time-Taken"])
print("Request type:", response.headers["X-Request-Type"])
print("Created at:  ", response.headers["X-Created-At"])
```

---

## API Endpoints Summary

| Method | Endpoint                    | Description                              |
|--------|-----------------------------|------------------------------------------|
| `GET`  | `/health`                   | Health check — confirms API is running   |
| `POST` | `/swap-face`                | Swap face in a video                     |
| `POST` | `/swap-face-image`          | Swap face in an image                    |
| `GET`  | `/requests`                 | List all request records (newest first)  |
| `GET`  | `/request/{request_id}`     | Get one specific request record by ID    |

The interactive API docs (Swagger UI) are available at:
```
https://YOUR_NGROK_URL.ngrok-free.app/docs
```

---

## Server Log Tracing

On the Colab server side, each request prints its ID at the start:
```
[API] swap-face request_id=a26ed03f-b66d-4634-88db-ab285455fd56
[API] swap-face-image request_id=7e9d0c1a-4b2f-4e8c-a3d1-e5f6a7b8c9d0
```
This lets you match a client-side request ID with the server logs for debugging.
