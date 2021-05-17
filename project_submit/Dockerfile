# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3
# FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
COPY ./requirements.txt .

RUN rm /etc/apt/sources.list.d/*
ADD . /
WORKDIR /

RUN apt-get update && apt-get install -y curl
# RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ onnx
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ onnxruntime-gpu==1.3.0
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ ai-hub
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ flask
# # RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ pytorch_pretrained_bert==0.6.2
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch_pretrained_bert==0.6.2
# # RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ tokenizers==0.7.0
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ tokenizers==0.10.2
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn==0.24.1


CMD ["sh", "run.sh"]