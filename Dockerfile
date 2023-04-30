from python:3.9.6-slim-buster
ENV TZ Asia/Shanghai
ENV PROD yes
EXPOSE 5000

COPY index.py vegetable_model_v1.h5 /app/
#拷贝mqtt依赖包
COPY paho /app/paho
WORKDIR /app/

#RUN pip install -r requirements.txt -i https://pypi.douban.com/simple

ENTRYPOINT python index.py
