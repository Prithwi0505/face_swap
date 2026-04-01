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
%cd /content/AigetaiModelsBackend/face_swapping/fastApi_version

# 1. Start the FastAPI server in the background
!nohup uvicorn api:app --host 0.0.0.0 --port 7860 > /content/fastapi.log 2>&1 &

# Wait for server to boot and download models
import time
time.sleep(15)

# 2. Start the secure Ngrok tunnel
# Download ngrok perfectly explicitly to circumvent `apt` PPA broken signatures in Colab
!wget -q -O ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
!tar -xf ngrok.tgz
!chmod +x ngrok

!pip install pyngrok -q
from pyngrok import ngrok, conf

# Lock pyngrok explicitly directly to our freshly extracted binary
pyngrok_config = conf.PyngrokConfig(ngrok_path="/content/ngrok")

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN", pyngrok_config=pyngrok_config)
public_url = ngrok.connect(7860, pyngrok_config=pyngrok_config)

print(f"✅ FACE SWAP API IS ONLINE: {public_url}")
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
