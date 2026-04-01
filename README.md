# Face Swap API (FastAPI)

A robust, GPU-accelerated face-swapping pipeline exposed as a FastAPI webservice. Designed to be run headlessly on Google Colab (using ONNX Runtime, InsightFace, and GFPGAN).

## How to Run on Google Colab

Follow these exact steps to deploy the API to a free Google Colab Tesla T4 GPU instance.

### Step 1: Clone and Setup
Open a new Colab Notebook, make sure the hardware accelerator is set to **T4 GPU**, and run this in the first cell:

```python
%cd /content
!git clone -b Fastapi_models https://github.com/Agrawalenis/AigetaiModelsBackend.git
%cd /content/AigetaiModelsBackend/face_swapping/fastApi_version
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

# 2. Start the secure Ngrok tunnel
# 2. Start the secure Ngrok tunnel
# Install ngrok via official apt, ignoring any errors from broken third-party Colab PPAs
!curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
!echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
!sudo apt-get update || true
!sudo apt-get install -y ngrok

# Lock pyngrok explicitly to the exact installation path we found ngrok in
# Actually, let's just bypass the bug-ridden `pyngrok` Python library completely.
# We will use the native `ngrok` binary directly.
import time
import requests

# Kill any lingering ngrok tunnels
!pkill -f ngrok

# Start the native ngrok tunnel as a background job
# We explicitly use --log=stdout to prevent the binary from instantly crashing while attempting to paint its TUI in a headless environment
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

---

## Testing the API Locally

Once your Colab prints the `✅ FACE SWAP API IS ONLINE` URL, you can send requests to it from your local computer's terminal. 
*Note for Windows users: If you are using PowerShell, you must type `curl.exe` instead of `curl` and put the command on a single line!*

### 1. Swap Face in Video
To swap a face in a target video using a source face image:

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
*Depending on the length of the video, processing may take a minute or two. The `my_new_video.mp4` will be saved in the directory where you ran the command.*

### 2. Swap Face in Image (Image-to-Image)
To swap a face in a target image using a source face image:

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

## Unique Request ID

Every request — whether image-to-image or image-to-video — is assigned a **unique UUID** (`X-Request-ID` response header). This ID is also printed to the server log at the start of every request and automatically appended to a `request_ids.txt` file on the server, making it easy to trace or audit a specific job.

### Server-side Request Log File

In addition to being returned via headers, every generated unique request ID is securely stored line-by-line in a dedicated local file on the server: `request_ids.txt` (located in the `fastApi_version` folder). This provides a simple, persistent record of all processed requests.

### Reading the Request ID via curl

Add the `-i` flag to print response headers along with the output:

**Mac/Linux:**
```bash
curl -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" \
  -F "source_image=@my_source_face.jpg" \
  -F "target_image=@target_person.jpg" \
  --output my_swapped_face.jpg \
  -i 2>/dev/null | grep -i x-request-id
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image" -F "source_image=@my_source_face.jpg" -F "target_image=@target_person.jpg" --output my_swapped_face.jpg -D headers.txt
type headers.txt
```

Or save headers to a file and inspect after:
```powershell
curl.exe -X POST "..." -F "source_image=@source.jpg" -F "target_image=@target.jpg" --output result.jpg -D headers.txt
findstr /i "x-request-id" headers.txt
```

Example output:
```
x-request-id: 3f4a1b2c-8d6e-4f9a-b1c2-d3e4f5a6b7c8
```

### Reading the Request ID via Python

```python
import requests

with open("my_source_face.jpg", "rb") as src, open("target_person.jpg", "rb") as tgt:
    response = requests.post(
        "https://YOUR_NGROK_URL.ngrok-free.app/swap-face-image",
        files={"source_image": src, "target_image": tgt},
    )

request_id = response.headers["X-Request-ID"]
print(f"Request ID: {request_id}")

with open("my_swapped_face.jpg", "wb") as f:
    f.write(response.content)
```

### Server Log Tracing

On the Colab server side, each request prints its ID at the start:
```
[API] swap-face-image request_id=3f4a1b2c-8d6e-4f9a-b1c2-d3e4f5a6b7c8
[API] swap-face request_id=7e9d0c1a-4b2f-4e8c-a3d1-e5f6a7b8c9d0
```
This allows you to match a client-side request ID with the server logs for debugging.
