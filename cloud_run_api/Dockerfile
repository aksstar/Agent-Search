FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_ADC=true \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY config.py \
     setup_gcs_datastore.py \
     setup_5stores_bq_and_vertex.py \
     evaluate_all_queries.py \
     eval_5stores_nlp_collapse_html.py \
     api_server.py \
     /app/

COPY search_studio /app/search_studio

EXPOSE 8080

CMD ["python3", "-u", "/app/api_server.py"]
