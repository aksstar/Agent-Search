FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_ADC=true \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python3 -c "import shutil, pathlib; from huggingface_hub import hf_hub_download; d=pathlib.Path('/app/minilm_model'); d.mkdir(parents=True, exist_ok=True); shutil.copyfile(hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json'), d/'tokenizer.json'); shutil.copyfile(hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx'), d/'model.onnx')"

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
