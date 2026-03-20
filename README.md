# Face Swap API (FastAPI)

A robust, GPU-accelerated face-swapping pipeline exposed as a FastAPI webservice. Designed to be run headlessly on Google Colab (using ONNX Runtime, InsightFace, and GFPGAN).

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

# 2. Start the secure Ngrok tunnel
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(7860)

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
