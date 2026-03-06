#Face swapping
%cd face_swapping
!pip install numpy==1.24.3 --force-reinstall
!pip install onnxruntime-gpu==1.18.0
!pip install -r requirements.txt
!pip install opennsfw2 keras --upgrade
!apt-get update --yes
!apt install nvidia-cuda-toolkit --yes

#API
!pip install fastapi uvicorn python-multipart


!pip uninstall -y jax jaxlib
!pip install numpy==1.24.3 --force-reinstall
import shutil
shutil.rmtree("/root/.cache", ignore_errors=True)


%cd /content/GenaibackenD/face_swapping
!nohup uvicorn api:app --host 0.0.0.0 --port 7860 --reload > /content/fastapi.log 2>&1 &

%cd /content/GenaibackenD/face_swapping
!nohup uvicorn api:app --host 0.0.0.0 --port 7860 --reload > /content/fastapi.log 2>&1 &

!ngrok config add-authtoken 2zww0HAKzeb2giggQqJbTexIzzn_4oHmU8VnQzD8A8sWPqQVU
!pkill -f ngrok

!pip install pyngrok
from pyngrok import ngrok

public_url = ngrok.connect(7860)
print("Public API URL:", public_url)
