"""
serve.py — ML Inference FastAPI microservice
=============================================
Subscribes to `engine/telemetry`, runs the inference pipeline, and publishes
results to `engine/alerts`.  Also exposes REST endpoints for ad-hoc inference.

Startup order:
  1. Load all models (anomaly, classifier, RUL).
  2. Connect to MQTT.
  3. Start uvicorn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.anomaly_detection import AnomalyDetector
from src.fault_classifier import FaultClassifier
from src.rul_model import RULModel
from src.explainability import SHAPExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ML] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_TELEMETRY = os.getenv("MQTT_TOPIC_TELEMETRY", "engine/telemetry")
TOPIC_ALERTS = os.getenv("MQTT_TOPIC_ALERTS", "engine/alerts")

# Anomaly threshold — tune after training
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.6"))

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

anomaly = AnomalyDetector()
classifier = FaultClassifier()
rul = RULModel()
explainer = SHAPExplainer()

_mqtt_client: mqtt.Client | None = None
_loop: asyncio.AbstractEventLoop | None = None

# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------


def _on_connect(client, userdata, flags, rc, properties):
    logger.info("ML service MQTT connected (rc=%s)", rc)
    client.subscribe(TOPIC_TELEMETRY)


def _on_message(client, userdata, msg: mqtt.MQTTMessage):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return
    if _loop:
        asyncio.run_coroutine_threadsafe(_infer_and_publish(payload), _loop)


async def _infer_and_publish(sample: dict) -> None:
    """Run the full inference pipeline for one telemetry sample."""
    # 1. Anomaly detection
    anomaly_score = anomaly.score(sample)
    if anomaly_score < ANOMALY_THRESHOLD:
        return   # healthy — do not publish an alert

    # 2. Fault classification
    fault_label, confidence = classifier.predict(sample)

    # 3. RUL estimation
    rul.update(sample)
    rul_hours = rul.predict()

    # 4. Explainability
    top_features = explainer.top_features(sample)

    # 5. Publish alert
    alert = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "fault_type":        fault_label,
        "confidence":        round(confidence, 4),
        "rul_hours":         rul_hours,
        "shap_top_features": top_features,
    }
    _mqtt_client.publish(TOPIC_ALERTS, json.dumps(alert), qos=0)
    logger.info("Alert published: %s (conf=%.2f, RUL=%s h)", fault_label, confidence, rul_hours)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mqtt_client, _loop
    _loop = asyncio.get_event_loop()

    # Load models (best-effort — service stays up even without trained models)
    anomaly.load()
    if classifier.load() and classifier.model is not None:
        explainer.setup(classifier.model, None)
    rul.load()

    # Connect MQTT
    _mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="dt-ml-serve")
    _mqtt_client.on_connect = _on_connect
    _mqtt_client.on_message = _on_message
    _mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    _mqtt_client.loop_start()

    logger.info("ML service ready")
    yield

    _mqtt_client.loop_stop()
    _mqtt_client.disconnect()
    logger.info("ML service shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Digital Twin — ML Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "service": "ml"}


@app.post("/infer", tags=["inference"])
async def infer(sample: dict):
    """
    Ad-hoc inference endpoint.
    Accepts a telemetry JSON body and returns anomaly score + fault prediction.
    """
    anomaly_score = anomaly.score(sample)
    fault_label, confidence, prob_map = classifier.predict(sample, return_probs=True)
    rul.update(sample)
    rul_hours = rul.predict()
    top_features = explainer.top_features(sample)

    return {
        "anomaly_score":     anomaly_score,
        "fault_label":       fault_label,
        "confidence":        confidence,
        "rul_hours":         rul_hours,
        "shap_top_features": top_features,
        "probabilities":     prob_map,
    }
