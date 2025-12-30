FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /exp

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY train.py .
COPY Controller.py .
COPY Utils/ Utils/

CMD ["python", "train.py"]
