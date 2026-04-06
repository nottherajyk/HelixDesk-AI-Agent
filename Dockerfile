FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# Verify environment import works (required for grading)
RUN python -c "from helixdesk import HelixDeskEnv; print('HelixDeskEnv import OK')"
# Required for hackathon gate
RUN openenv validate
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
