FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "train.py", "--agent", "rule", "--episodes", "10"]
