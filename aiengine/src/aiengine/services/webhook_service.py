#!/usr/bin/env python3
"""
AI Engine Webhook Service - Receives alerts and forwards to AI Engine
"""

import json
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import os

# Configuration
KAFKA_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'prometheus-alerts'
AI_ENGINE_URL = 'http://localhost:8085/api/process_alert'
WEBHOOK_PORT = int(os.getenv('PORT', '9095'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Kafka producer
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVERS],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    logger.info("✅ Kafka producer initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Kafka producer: {e}")
    producer = None

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        logger.info(f"📨 Received webhook: {data}")

        # Send to Kafka
        if producer:
            producer.send(KAFKA_TOPIC, data)
            logger.info("✅ Sent to Kafka")

        # Send directly to AI Engine
        response = requests.post(AI_ENGINE_URL, json=data, timeout=10)
        logger.info(f"✅ Sent to AI Engine: {response.status_code}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({"webhook_port": WEBHOOK_PORT, "ai_engine": AI_ENGINE_URL, "kafka": KAFKA_SERVERS})

if __name__ == '__main__':
    logger.info(f"🚀 AI Webhook starting on port {WEBHOOK_PORT}")
    app.run(host='0.0.0.0', port=WEBHOOK_PORT, debug=False)
