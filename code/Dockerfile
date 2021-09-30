FROM python:3.8

RUN pip install --no-cache-dir rasterio
RUN pip install --no-cache-dir torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir pillow
RUN pip install --no-cache-dir notebook
RUN pip install --no-cache-dir pycocotools
RUN pip install --no-cache-dir fiona
RUN pip install --no-cache-dir shapely
RUN pip install --no-cache-dir geojson
RUN pip install --no-cache-dir tqdm

RUN pip install --no-cache-dir opencv-python
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work

WORKDIR /work


RUN chmod +x train.sh
RUN chmod +x test.sh