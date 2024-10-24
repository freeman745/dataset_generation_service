FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip install "flask[async]"

RUN apt-get update

RUN apt-get install libgl1-mesa-glx -y

RUN apt-get install libglib2.0-0 -y

RUN pip install -e thirdparty/mongodb

EXPOSE 4001

CMD ["python", "-m", "app"]