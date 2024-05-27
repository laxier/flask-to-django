FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MONGODB_HOST="host.docker.internal"
ENV MONGODB_PORT="27017"
ENV MONGODB_DATABASE="blog"
ENV MONGODB_USERNAME="root"
ENV MONGODB_PASSWORD="rootroot"

ENV PYTHONUTF8=1

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
