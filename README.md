# VisionGuard (23L-0654_A2_Ops)

![CI](https://github.com/MSabihkhan/VisionGuard/actions/workflows/ci.yml/badge.svg)

Web service that tells whether an uploaded image is AI-generated or real.

## Run with Docker

```bash
cp .env.example .env
python download_champion.py
docker compose up --build
```

Open http://localhost:5000. Health check: http://localhost:5000/health

## Layout

- `train_*.py`, `evaluate.py`, `tune.py`: model training and evaluation
- `train_tracked.py`, `register_model.py`: MLflow tracking and model registry on DagsHub
- `app.py`, `inference.py`, `db.py`: Flask app, model loading and history database
- `Dockerfile`, `docker-compose.yml`: web app and Postgres stack
