FROM node:24-bookworm-slim AS frontend
WORKDIR /app/web/frontend
COPY web/frontend/package*.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS service
WORKDIR /app
COPY web/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY web/backend/ web/backend/
COPY data/fitted/ data/fitted/
COPY --from=frontend /app/web/frontend/dist web/frontend/dist
COPY --from=frontend /app/web/frontend/public/replays web/frontend/public/replays
ENV APSIS_LIVE=0 APSIS_CACHE_DIR=/tmp/apsis-cache
USER 65534:65534
EXPOSE 8000
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
