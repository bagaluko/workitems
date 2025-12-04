#!/usr/bin/env python3
"""
Comprehensive Prometheus Alert Receiver for Universal AI Engine
Advanced alert processing, correlation analysis, auto-remediation, and learning system
Integrates with the existing prometheus_alert_processor.py and Universal AI ecosystem
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import hashlib
import subprocess
import requests
import signal
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, asdict, field
from enum import Enum
import sqlite3
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths for imports
sys.path.insert(0, '/aiengine/src/aiengine')
sys.path.insert(0, '/aiengine/src/aiengine/monitoring')
sys.path.insert(0, '/aiengine/src/aiengine/core')

# Setup comprehensive logging
os.makedirs('/aiengine/src/aiengine/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/aiengine/src/aiengine/logs/prometheus_alert_receiver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PrometheusAlertReceiver')

try:
    from flask import Flask, request, jsonify, render_template_string
    from werkzeug.serving import make_server
    FLASK_AVAILABLE = True
    logger.info("✅ Flask available for alert receiver")
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("⚠️ Flask not available - Prometheus alert receiver disabled")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('/aiengine/src/aiengine/.env')
    logger.info("✅ Environment variables loaded")
except ImportError:
    logger.warning("⚠️ python-dotenv not available")

# Import existing alert processor components
try:
    from prometheus_alert_processor import (
        EnhancedAlertProcessor, AIAlertProcessor, AlertSeverity,
        RemediationStatus, PrometheusAlert, RemediationAction,
        RemediationResult, convert_deque_to_list, DequeEncoder
    )
    EXISTING_PROCESSOR_AVAILABLE = True
    logger.info("✅ Existing alert processor components loaded successfully")
except ImportError as e:
    EXISTING_PROCESSOR_AVAILABLE = False
    logger.warning(f"⚠️ Existing alert processor not available: {e}")

# Try to import Universal AI system
try:
    sys.path.insert(0, '/aiengine/src/aiengine')
    from main import (
        UniversalTask, UniversalSolution, DomainType, TaskType,
        WorldClassUniversalNeuralSystem
    )
    UNIVERSAL_AI_AVAILABLE = True
    logger.info("✅ Universal AI system components loaded")
except ImportError as e:
    UNIVERSAL_AI_AVAILABLE = False
    logger.warning(f"⚠️ Universal AI system not available: {e}")

# Try to import precheck engine
try:
    from core.precheck_engine import PrecheckEngine, PrecheckFailure, IngredientType
    PRECHECK_ENGINE_AVAILABLE = True
    logger.info("✅ Precheck engine loaded")
except ImportError as e:
    PRECHECK_ENGINE_AVAILABLE = False
    logger.warning(f"⚠️ Precheck engine not available: {e}")

# Configuration classes
@dataclass
class AlertReceiverConfig:
    """Configuration for the alert receiver"""
    host: str = '0.0.0.0'
    port: int = 8052
    max_workers: int = 10
    alert_retention_days: int = 30
    correlation_window_minutes: int = 5
    throttle_duration_seconds: int = 60
    max_alerts_per_minute: int = 5
    enable_auto_remediation: bool = True
    enable_learning: bool = True
    enable_notifications: bool = True
    database_path: str = '/aiengine/src/aiengine/monitoring/alert_receiver.db'
    knowledge_base_path: str = '/aiengine/src/aiengine/monitoring/knowledge_base'
    backup_retention_days: int = 7
    integration_mode: str = 'enhanced'  # 'basic', 'enhanced', 'full'

@dataclass
class AlertContext:
    """Extended context for alert processing"""
    alert_id: str
    fingerprint: str
    received_at: datetime
    source_ip: str
    processing_start: datetime
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    child_alert_ids: List[str] = field(default_factory=list)
    processing_history: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationRule:
    """Rule for alert correlation"""
    rule_id: str
    name: str
    alert_patterns: List[str]
    time_window_minutes: int
    correlation_type: str
    confidence_threshold: float
    action_recommendation: str
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    channel_type: str  # email, slack, webhook, sms
    endpoint: str
    credentials: Dict[str, str]
    alert_filters: List[str]
    enabled: bool = True
    rate_limit_per_hour: int = 100
    last_notification: Optional[datetime] = None
    notification_count: int = 0

class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ProcessingStatus(Enum):
    """Alert processing status"""
    RECEIVED = "received"
    QUEUED = "queued"
    PROCESSING = "processing"
    CORRELATED = "correlated"
    REMEDIATED = "remediated"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FAILED = "failed"

class DatabaseManager:
    """Manages SQLite database for alert storage and analytics"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        fingerprint TEXT,
                        alertname TEXT,
                        severity TEXT,
                        status TEXT,
                        instance TEXT,
                        job TEXT,
                        labels TEXT,
                        annotations TEXT,
                        starts_at TEXT,
                        ends_at TEXT,
                        received_at TEXT,
                        processed_at TEXT,
                        processing_status TEXT,
                        correlation_id TEXT,
                        remediation_id TEXT,
                        ai_confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Correlations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS correlations (
                        id TEXT PRIMARY KEY,
                        correlation_type TEXT,
                        alert_ids TEXT,
                        confidence REAL,
                        recommended_action TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TEXT
                    )
                ''')

                # Performance tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_type TEXT,
                        duration_seconds REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_alertname ON alerts(alertname)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_received_at ON alerts(received_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(fingerprint)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlations_type ON correlations(correlation_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_operation ON performance_tracking(operation_type)')

                conn.commit()
                logger.info("✅ Alert receiver database initialized successfully")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    def store_alert(self, alert_data: Dict, context: AlertContext) -> bool:
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts
                    (id, fingerprint, alertname, severity, status, instance, job,
                     labels, annotations, starts_at, ends_at, received_at,
                     processing_status, correlation_id, ai_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    context.alert_id,
                    context.fingerprint,
                    alert_data.get('labels', {}).get('alertname', 'Unknown'),
                    alert_data.get('labels', {}).get('severity', 'unknown'),
                    alert_data.get('status', 'unknown'),
                    alert_data.get('labels', {}).get('instance', 'unknown'),
                    alert_data.get('labels', {}).get('job', 'unknown'),
                    json.dumps(alert_data.get('labels', {})),
                    json.dumps(alert_data.get('annotations', {})),
                    alert_data.get('startsAt', ''),
                    alert_data.get('endsAt', ''),
                    context.received_at.isoformat(),
                    ProcessingStatus.RECEIVED.value,
                    context.correlation_id,
                    0.0
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to store alert: {e}")
            return False

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                since = (datetime.now() - timedelta(hours=hours)).isoformat()
                cursor.execute('''
                    SELECT * FROM alerts
                    WHERE received_at > ?
                    ORDER BY received_at DESC
                    LIMIT 1000
                ''', (since,))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get recent alerts: {e}")
            return []

    def store_correlation(self, correlation_data: Dict) -> bool:
        """Store correlation data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO correlations
                    (id, correlation_type, alert_ids, confidence, recommended_action)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    correlation_data['id'],
                    correlation_data['type'],
                    json.dumps(correlation_data['alert_ids']),
                    correlation_data['confidence'],
                    correlation_data['recommended_action']
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to store correlation: {e}")
            return False

    def store_performance_metric(self, operation_type: str, duration: float, success: bool, error_message: str = None):
        """Store performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_tracking
                    (operation_type, duration_seconds, success, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (operation_type, duration, success, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to store performance metric: {e}")

    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                since = (datetime.now() - timedelta(hours=hours)).isoformat()

                # Get operation statistics
                cursor.execute('''
                    SELECT operation_type,
                           COUNT(*) as total_operations,
                           AVG(duration_seconds) as avg_duration,
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations
                    FROM performance_tracking
                    WHERE timestamp > ?
                    GROUP BY operation_type
                ''', (since,))

                stats = {}
                for row in cursor.fetchall():
                    operation_type, total, avg_duration, successful = row
                    stats[operation_type] = {
                        'total_operations': total,
                        'avg_duration_seconds': avg_duration,
                        'successful_operations': successful,
                        'success_rate': successful / total if total > 0 else 0
                    }

                return stats
        except Exception as e:
            logger.error(f"❌ Failed to get performance stats: {e}")
            return {}

    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()

                # Clean up old alerts
                cursor.execute('DELETE FROM alerts WHERE received_at < ?', (cutoff,))
                alerts_deleted = cursor.rowcount

                # Clean up old correlations
                cursor.execute('DELETE FROM correlations WHERE created_at < ?', (cutoff,))
                correlations_deleted = cursor.rowcount

                # Clean up old performance data
                cursor.execute('DELETE FROM performance_tracking WHERE timestamp < ?', (cutoff,))
                performance_deleted = cursor.rowcount

                conn.commit()
                logger.info(f"🧹 Cleaned up old data: {alerts_deleted} alerts, {correlations_deleted} correlations, {performance_deleted} performance records")

        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")

class AdvancedCorrelationEngine:
    """Advanced correlation engine for alert analysis"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.correlation_rules = []
        self.create_default_rules()
        self.pattern_cache = {}

    def create_default_rules(self):
        """Create default correlation rules"""
        default_rules = [
            CorrelationRule(
                rule_id="cascade_failure",
                name="Cascade Failure Detection",
                alert_patterns=["*Down", "*Unreachable", "*Failed"],
                time_window_minutes=5,
                correlation_type="cascade",
                confidence_threshold=0.8,
                action_recommendation="investigate_root_cause"
            ),
            CorrelationRule(
                rule_id="resource_pressure",
                name="Resource Pressure Detection",
                alert_patterns=["HighCPUUsage", "HighMemoryUsage", "DiskSpaceLow"],
                time_window_minutes=10,
                correlation_type="resource_pressure",
                confidence_threshold=0.7,
                action_recommendation="scale_resources"
            ),
            CorrelationRule(
                rule_id="ai_system_degradation",
                name="AI System Degradation",
                alert_patterns=["LowAIConfidence", "AISystemHealthDegraded", "LowDomainMastery"],
                time_window_minutes=15,
                correlation_type="ai_degradation",
                confidence_threshold=0.75,
                action_recommendation="retrain_models"
            )
        ]
        self.correlation_rules = default_rules

    def find_correlations(self, current_alerts: List[Dict], time_window_minutes: int = 5) -> List[Dict]:
        """Find correlations using advanced algorithms"""
        correlations = []

        try:
            # Get recent alerts from database
            recent_alerts = self.db_manager.get_recent_alerts(hours=time_window_minutes/60*2)

            # Rule-based correlation
            rule_correlations = self.find_rule_based_correlations(current_alerts, recent_alerts)
            correlations.extend(rule_correlations)

            # Temporal correlation
            temporal_correlations = self.find_temporal_correlations(current_alerts, recent_alerts)
            correlations.extend(temporal_correlations)

            # Remove duplicates and sort by confidence
            unique_correlations = self.deduplicate_correlations(correlations)
            return sorted(unique_correlations, key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {e}")
            return []

    def find_rule_based_correlations(self, current_alerts: List[Dict], recent_alerts: List[Dict]) -> List[Dict]:
        """Find correlations based on predefined rules"""
        correlations = []

        for rule in self.correlation_rules:
            if not rule.enabled:
                continue

            try:
                matching_alerts = []

                # Check current alerts
                for alert in current_alerts:
                    alert_name = alert.get('labels', {}).get('alertname', '')
                    if self.matches_pattern(alert_name, rule.alert_patterns):
                        matching_alerts.append(alert)

                # Check recent alerts within time window
                cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
                for alert in recent_alerts:
                    try:
                        alert_time = datetime.fromisoformat(alert['received_at'])
                        if alert_time > cutoff_time:
                            alert_name = alert['alertname']
                            if self.matches_pattern(alert_name, rule.alert_patterns):
                                matching_alerts.append(alert)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse alert time: {e}")

                # If we have multiple matching alerts, create correlation
                if len(matching_alerts) >= 2:
                    correlation = {
                        'id': str(uuid.uuid4()),
                        'type': rule.correlation_type,
                        'rule_id': rule.rule_id,
                        'alert_ids': [alert.get('id', str(uuid.uuid4())) for alert in matching_alerts],
                        'alert_names': [alert.get('labels', {}).get('alertname', alert.get('alertname', '')) for alert in matching_alerts],
                        'confidence': rule.confidence_threshold,
                        'recommended_action': rule.action_recommendation,
                        'time_window': rule.time_window_minutes,
                        'created_at': datetime.now().isoformat(),
                        'rule_name': rule.name,
                        'matching_alerts_count': len(matching_alerts)
                    }
                    correlations.append(correlation)

                    # Update rule statistics
                    rule.last_triggered = datetime.now()
                    rule.trigger_count += 1

                    logger.info(f"🔗 Rule-based correlation found: {rule.name} with {len(matching_alerts)} alerts")

            except Exception as e:
                logger.error(f"❌ Rule-based correlation failed for rule {rule.rule_id}: {e}")

        return correlations

    def find_temporal_correlations(self, current_alerts: List[Dict], recent_alerts: List[Dict]) -> List[Dict]:
        """Find temporal correlations between alerts"""
        correlations = []

        try:
            # Group alerts by time windows
            time_windows = self.group_alerts_by_time_windows(recent_alerts, window_minutes=2)

            for window_start, window_alerts in time_windows.items():
                if len(window_alerts) >= 3:  # Multiple alerts in short time window
                    correlation = {
                        'id': str(uuid.uuid4()),
                        'type': 'temporal_burst',
                        'alert_ids': [alert.get('id', str(uuid.uuid4())) for alert in window_alerts if 'id' in alert],
                        'alert_names': [alert.get('alertname', 'Unknown') for alert in window_alerts],
                        'confidence': min(0.9, len(window_alerts) * 0.2),
                        'recommended_action': 'investigate_burst',
                        'time_window': window_start,
                        'alert_count': len(window_alerts),
                        'created_at': datetime.now().isoformat(),
                        'burst_intensity': len(window_alerts)
                    }
                    correlations.append(correlation)
                    logger.info(f"🔗 Temporal burst correlation found: {len(window_alerts)} alerts in 2-minute window")

        except Exception as e:
            logger.error(f"❌ Temporal correlation failed: {e}")

        return correlations

    def matches_pattern(self, alert_name: str, patterns: List[str]) -> bool:
        """Check if alert name matches any of the patterns"""
        for pattern in patterns:
            if pattern.startswith('*') and pattern.endswith('*'):
                # Contains pattern
                if pattern[1:-1].lower() in alert_name.lower():
                    return True
            elif pattern.startswith('*'):
                # Ends with pattern
                if alert_name.lower().endswith(pattern[1:].lower()):
                    return True
            elif pattern.endswith('*'):
                # Starts with pattern
                if alert_name.lower().startswith(pattern[:-1].lower()):
                    return True
            else:
                # Exact match
                if alert_name.lower() == pattern.lower():
                    return True
        return False

    def group_alerts_by_time_windows(self, alerts: List[Dict], window_minutes: int = 2) -> Dict[str, List[Dict]]:
        """Group alerts by time windows"""
        windows = defaultdict(list)

        for alert in alerts:
            try:
                alert_time_str = alert.get('received_at', alert.get('startsAt', ''))
                if alert_time_str:
                    alert_time = datetime.fromisoformat(alert_time_str.replace('Z', '+00:00'))
                    # Round down to nearest window
                    window_start = alert_time.replace(minute=(alert_time.minute // window_minutes) * window_minutes, second=0, microsecond=0)
                    windows[window_start.isoformat()].append(alert)
            except Exception as e:
                logger.warning(f"⚠️ Failed to group alert by time: {e}")

        return dict(windows)

    def deduplicate_correlations(self, correlations: List[Dict]) -> List[Dict]:
        """Remove duplicate correlations"""
        seen = set()
        unique_correlations = []

        for correlation in correlations:
            # Create a signature for the correlation
            alert_ids = tuple(sorted(correlation.get('alert_ids', [])))
            correlation_type = correlation.get('type', '')
            signature = (alert_ids, correlation_type)

            if signature not in seen:
                seen.add(signature)
                unique_correlations.append(correlation)

        return unique_correlations

class NotificationManager:
    """Manages alert notifications across multiple channels"""

    def __init__(self, config: AlertReceiverConfig):
        self.config = config
        self.channels = []
        self.create_default_channels()
        self.rate_limits = defaultdict(list)

    def create_default_channels(self):
        """Create default notification channels"""
        default_channels = [
            NotificationChannel(
                channel_id="log_all",
                channel_type="log",
                endpoint="/aiengine/src/aiengine/logs/alert_notifications.log",
                credentials={},
                alert_filters=["*"],
                rate_limit_per_hour=1000
            )
        ]
        self.channels = default_channels

    def send_notification(self, alert_data: Dict, correlation_data: Optional[Dict] = None) -> bool:
        """Send notification through appropriate channels"""
        try:
            alert_severity = alert_data.get('labels', {}).get('severity', 'unknown')
            alert_name = alert_data.get('labels', {}).get('alertname', 'Unknown')

            notifications_sent = 0

            for channel in self.channels:
                if not channel.enabled:
                    continue

                # Check if alert matches channel filters
                if not self.matches_filters(alert_severity, alert_name, channel.alert_filters):
                    continue

                # Check rate limits
                if not self.check_rate_limit(channel):
                    logger.warning(f"⚠️ Rate limit exceeded for channel {channel.channel_id}")
                    continue

                # Send notification
                success = self.send_channel_notification(channel, alert_data, correlation_data)
                if success:
                    notifications_sent += 1
                    self.update_rate_limit(channel)

            logger.info(f"📧 Sent {notifications_sent} notifications for alert {alert_name}")
            return notifications_sent > 0

        except Exception as e:
            logger.error(f"❌ Notification sending failed: {e}")
            return False

    def send_channel_notification(self, channel: NotificationChannel, alert_data: Dict, correlation_data: Optional[Dict] = None) -> bool:
        """Send notification through specific channel"""
        try:
            if channel.channel_type == 'log':
                return self.send_log_notification(channel, alert_data, correlation_data)
            else:
                logger.warning(f"⚠️ Unknown channel type: {channel.channel_type}")
                return False
        except Exception as e:
            logger.error(f"❌ Channel notification failed for {channel.channel_id}: {e}")
            return False

    def send_log_notification(self, channel: NotificationChannel, alert_data: Dict, correlation_data: Optional[Dict] = None) -> bool:
        """Send log notification"""
        try:
            alert_name = alert_data.get('labels', {}).get('alertname', 'Unknown')
            severity = alert_data.get('labels', {}).get('severity', 'unknown')
            instance = alert_data.get('labels', {}).get('instance', 'unknown')

            log_message = f"[{datetime.now().isoformat()}] ALERT: {alert_name} | Severity: {severity} | Instance: {instance}"

            if correlation_data:
                log_message += f" | Correlation: {correlation_data.get('type', 'unknown')} (confidence: {correlation_data.get('confidence', 0):.2f})"

            # Ensure log directory exists
            os.makedirs(os.path.dirname(channel.endpoint), exist_ok=True)

            with open(channel.endpoint, 'a') as f:
                f.write(log_message + '\n')

            return True

        except Exception as e:
            logger.error(f"❌ Log notification failed: {e}")
            return False

    def matches_filters(self, severity: str, alert_name: str, filters: List[str]) -> bool:
        """Check if alert matches channel filters"""
        if '*' in filters:
            return True

        for filter_pattern in filters:
            if filter_pattern.lower() == severity.lower():
                return True
            if filter_pattern.lower() in alert_name.lower():
                return True

        return False

    def check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)

        # Clean old entries
        self.rate_limits[channel.channel_id] = [
            timestamp for timestamp in self.rate_limits[channel.channel_id]
            if timestamp > hour_ago
        ]

        return len(self.rate_limits[channel.channel_id]) < channel.rate_limit_per_hour

    def update_rate_limit(self, channel: NotificationChannel):
        """Update rate limit tracking"""
        self.rate_limits[channel.channel_id].append(datetime.now())
        channel.last_notification = datetime.now()
        channel.notification_count += 1

class PrometheusAlertReceiver:
    """Comprehensive Prometheus Alert Receiver with integration to existing processor"""

    def __init__(self, config: Optional[AlertReceiverConfig] = None, universal_system=None):
        self.config = config or AlertReceiverConfig()
        self.universal_system = universal_system
        self.app = None
        self.server = None
        self.server_thread = None
        self.running = False
        self.shutdown_event = threading.Event()

        # Initialize components
        self.db_manager = DatabaseManager(self.config.database_path)
        self.correlation_engine = AdvancedCorrelationEngine(self.db_manager)
        self.notification_manager = NotificationManager(self.config)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Integration with existing processors
        self.enhanced_processor = None
        self.ai_alert_processor = None
        self.precheck_engine = None

        # Initialize processors based on availability
        self.init_processors()

        # Metrics and statistics
        self.metrics = {
            'alerts_received': 0,
            'alerts_processed': 0,
            'correlations_found': 0,
            'notifications_sent': 0,
            'processing_errors': 0,
            'integration_calls': 0,
            'performance_avg_ms': 0.0
        }

        # Setup Flask app
        if FLASK_AVAILABLE:
            self.setup_flask_app()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Start background tasks
        self.start_background_tasks()

        logger.info(f"🚀 Comprehensive Prometheus Alert Receiver initialized (Integration Mode: {self.config.integration_mode})")

    def init_processors(self):
        """Initialize alert processors based on availability"""
        try:
            if EXISTING_PROCESSOR_AVAILABLE:
                # Initialize enhanced processor from existing system
                self.enhanced_processor = EnhancedAlertProcessor()
                logger.info("✅ Enhanced alert processor from existing system initialized")

                # Try to initialize full AI alert processor if available
                if self.config.integration_mode in ['enhanced', 'full']:
                    try:
                        # Use a different port to avoid conflicts
                        ai_port = self.config.port + 100
                        self.ai_alert_processor = AIAlertProcessor(port=ai_port)
                        logger.info(f"✅ AI alert processor initialized on port {ai_port}")
                    except Exception as e:
                        logger.warning(f"⚠️ AI alert processor not available: {e}")

            if PRECHECK_ENGINE_AVAILABLE and self.config.integration_mode == 'full':
                self.precheck_engine = PrecheckEngine()
                logger.info("✅ Precheck engine initialized")

            if UNIVERSAL_AI_AVAILABLE:
                if not self.universal_system:
                    try:
                        self.universal_system = WorldClassUniversalNeuralSystem()
                        logger.info("✅ Universal AI system initialized")
                    except Exception as e:
                        logger.warning(f"⚠️ Universal AI system initialization failed: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize processors: {e}")

    def setup_flask_app(self):
        """Setup comprehensive Flask application"""
        self.app = Flask(__name__)

        # Main webhook endpoints
        @self.app.route('/webhook', methods=['POST'])
        @self.app.route('/alerts', methods=['POST'])
        def receive_alerts():
            return self.handle_webhook_request()

        # Health and status endpoints
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.get_health_status()

        @self.app.route('/status', methods=['GET'])
        def get_status():
            return self.get_comprehensive_status()

        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            return self.get_prometheus_metrics()

        # Integration endpoints
        @self.app.route('/integration/status', methods=['GET'])
        def integration_status():
            return self.get_integration_status()

        @self.app.route('/integration/test', methods=['POST'])
        def test_integration():
            return self.test_integration_endpoint()

        # Test endpoints
        @self.app.route('/test/alert', methods=['POST'])
        def test_alert_processing():
            return self.test_alert_processing_endpoint()

        # Dashboard endpoint
        @self.app.route('/dashboard', methods=['GET'])
        def dashboard():
            return self.render_dashboard()

    def handle_webhook_request(self):
        """Handle incoming webhook requests with comprehensive processing"""
        start_time = time.time()

        try:
            # Get request data
            data = request.get_json()
            if not data:
                self.metrics['processing_errors'] += 1
                return jsonify({'error': 'No JSON data received'}), 400

            # Extract alerts
            alerts = data.get('alerts', [])
            if not alerts:
                return jsonify({'status': 'no_alerts', 'message': 'No alerts in request'}), 200

            # Get client information
            client_ip = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')

            logger.info(f"📨 Received {len(alerts)} alerts from {client_ip}")
            self.metrics['alerts_received'] += len(alerts)

            # Process alerts based on integration mode
            if self.config.integration_mode == 'basic':
                result = self.process_alerts_basic(alerts, client_ip, user_agent)
            elif self.config.integration_mode == 'enhanced':
                result = self.process_alerts_enhanced(alerts, client_ip, user_agent)
            else:  # full
                result = self.process_alerts_comprehensive(alerts, client_ip, user_agent)

            processing_time = time.time() - start_time
            self.metrics['performance_avg_ms'] = (self.metrics['performance_avg_ms'] + processing_time * 1000) / 2

            # Store performance metric
            self.db_manager.store_performance_metric(
                'webhook_processing',
                processing_time,
                result.get('status') != 'error',
                result.get('error')
            )

            result['processing_time_seconds'] = processing_time
            result['timestamp'] = datetime.now().isoformat()
            result['integration_mode'] = self.config.integration_mode

            return jsonify(result), 200

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Webhook request handling failed: {e}")
            self.metrics['processing_errors'] += 1

            # Store error metric
            self.db_manager.store_performance_metric(
                'webhook_processing',
                processing_time,
                False,
                str(e)
            )

            return jsonify({
                'error': str(e),
                'processing_time_seconds': processing_time,
                'integration_mode': self.config.integration_mode
            }), 500

    def process_alerts_basic(self, alerts: List[Dict], client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Basic alert processing with minimal features"""
        processing_start = datetime.now()
        results = {
            'status': 'processed',
            'mode': 'basic',
            'alerts_received': len(alerts),
            'alerts_processed': 0,
            'errors': []
        }

        try:
            for alert_data in alerts:
                try:
                    # Create basic context
                    context = AlertContext(
                        alert_id=str(uuid.uuid4()),
                        fingerprint=self.generate_alert_fingerprint(alert_data),
                        received_at=processing_start,
                        source_ip=client_ip,
                        processing_start=datetime.now(),
                        metadata={'user_agent': user_agent, 'mode': 'basic'}
                    )

                    # Store alert in database
                    self.db_manager.store_alert(alert_data, context)

                    # Basic processing
                    alert_name = alert_data.get('labels', {}).get('alertname', 'Unknown')
                    logger.info(f"📝 Basic processing: {alert_name}")

                    results['alerts_processed'] += 1
                    self.metrics['alerts_processed'] += 1

                except Exception as e:
                    error_msg = f"Failed to process alert: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    self.metrics['processing_errors'] += 1

            processing_time = (datetime.now() - processing_start).total_seconds()
            results['processing_time_seconds'] = processing_time

            logger.info(f"✅ Basic processing completed: {results['alerts_processed']}/{len(alerts)} alerts in {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Basic alert processing failed: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results

    def process_alerts_enhanced(self, alerts: List[Dict], client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Enhanced alert processing with correlation and existing processor integration"""
        processing_start = datetime.now()
        results = {
            'status': 'processed',
            'mode': 'enhanced',
            'alerts_received': len(alerts),
            'alerts_processed': 0,
            'correlations_found': 0,
            'notifications_sent': 0,
            'integration_results': {},
            'errors': []
        }

        try:
            # Process each alert
            for alert_data in alerts:
                try:
                    # Create alert context
                    context = AlertContext(
                        alert_id=str(uuid.uuid4()),
                        fingerprint=self.generate_alert_fingerprint(alert_data),
                        received_at=processing_start,
                        source_ip=client_ip,
                        processing_start=datetime.now(),
                        metadata={'user_agent': user_agent, 'mode': 'enhanced'}
                    )

                    # Store alert in database
                    self.db_manager.store_alert(alert_data, context)

                    results['alerts_processed'] += 1
                    self.metrics['alerts_processed'] += 1

                except Exception as e:
                    error_msg = f"Failed to process alert: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    self.metrics['processing_errors'] += 1

            # Find correlations
            try:
                correlations = self.correlation_engine.find_correlations(alerts, self.config.correlation_window_minutes)
                results['correlations_found'] = len(correlations)
                results['correlations'] = correlations
                self.metrics['correlations_found'] += len(correlations)

                # Store correlations in database
                for correlation in correlations:
                    self.db_manager.store_correlation(correlation)

            except Exception as e:
                error_msg = f"Correlation analysis failed: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)

            # Send notifications
            try:
                notifications_sent = 0
                for i, alert_data in enumerate(alerts):
                    correlation_data = correlations[i] if i < len(correlations) else None
                    if self.notification_manager.send_notification(alert_data, correlation_data):
                        notifications_sent += 1

                results['notifications_sent'] = notifications_sent
                self.metrics['notifications_sent'] += notifications_sent

            except Exception as e:
                error_msg = f"Notification sending failed: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)

            # Integration with existing enhanced processor
            if self.enhanced_processor:
                try:
                    self.metrics['integration_calls'] += 1
                    enhanced_results = self.enhanced_processor.process_alerts_enhanced(alerts)
                    results['integration_results']['enhanced_processor'] = enhanced_results
                    results['enhanced_statistics'] = self.enhanced_processor.get_processing_statistics()
                    logger.info("✅ Enhanced processor integration successful")
                except Exception as e:
                    error_msg = f"Enhanced processor integration failed: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)

            processing_time = (datetime.now() - processing_start).total_seconds()
            results['processing_time_seconds'] = processing_time

            logger.info(f"✅ Enhanced processing completed: {results['alerts_processed']}/{len(alerts)} alerts in {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Enhanced alert processing failed: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results

    def process_alerts_comprehensive(self, alerts: List[Dict], client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Comprehensive alert processing with all features"""
        processing_start = datetime.now()
        results = {
            'status': 'processed',
            'mode': 'comprehensive',
            'alerts_received': len(alerts),
            'alerts_processed': 0,
            'correlations_found': 0,
            'notifications_sent': 0,
            'integration_results': {},
            'errors': []
        }

        try:
            # Process each alert
            for alert_data in alerts:
                try:
                    # Create alert context
                    context = AlertContext(
                        alert_id=str(uuid.uuid4()),
                        fingerprint=self.generate_alert_fingerprint(alert_data),
                        received_at=processing_start,
                        source_ip=client_ip,
                        processing_start=datetime.now(),
                        metadata={'user_agent': user_agent, 'mode': 'comprehensive'}
                    )

                    # Store alert in database
                    self.db_manager.store_alert(alert_data, context)

                    results['alerts_processed'] += 1
                    self.metrics['alerts_processed'] += 1

                except Exception as e:
                    error_msg = f"Failed to process alert: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    self.metrics['processing_errors'] += 1

            # Find correlations
            try:
                correlations = self.correlation_engine.find_correlations(alerts, self.config.correlation_window_minutes)
                results['correlations_found'] = len(correlations)
                results['correlations'] = correlations
                self.metrics['correlations_found'] += len(correlations)

                # Store correlations in database
                for correlation in correlations:
                    self.db_manager.store_correlation(correlation)

            except Exception as e:
                error_msg = f"Correlation analysis failed: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)

            # Send notifications
            try:
                notifications_sent = 0
                for i, alert_data in enumerate(alerts):
                    correlation_data = correlations[i] if i < len(correlations) else None
                    if self.notification_manager.send_notification(alert_data, correlation_data):
                        notifications_sent += 1

                results['notifications_sent'] = notifications_sent
                self.metrics['notifications_sent'] += notifications_sent

            except Exception as e:
                error_msg = f"Notification sending failed: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)

            # Integration with all available processors
            integration_results = {}

            # Enhanced processor integration
            if self.enhanced_processor:
                try:
                    self.metrics['integration_calls'] += 1
                    enhanced_results = self.enhanced_processor.process_alerts_enhanced(alerts)
                    integration_results['enhanced_processor'] = enhanced_results
                    results['enhanced_statistics'] = self.enhanced_processor.get_processing_statistics()
                    logger.info("✅ Enhanced processor integration successful")
                except Exception as e:
                    error_msg = f"Enhanced processor integration failed: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)

            # AI processor integration
            if self.ai_alert_processor:
                try:
                    self.metrics['integration_calls'] += 1
                    ai_results = self.ai_alert_processor.process_alerts_enhanced(alerts)
                    integration_results['ai_processor'] = ai_results
                    logger.info("✅ AI processor integration successful")
                except Exception as e:
                    error_msg = f"AI processor integration failed: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)

            # Universal AI integration
            if self.universal_system:
                try:
                    self.metrics['integration_calls'] += 1
                    universal_results = self.process_with_universal_ai(alerts)
                    integration_results['universal_ai'] = universal_results
                    logger.info("✅ Universal AI integration successful")
                except Exception as e:
                    error_msg = f"Universal AI integration failed: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)

            results['integration_results'] = integration_results

            processing_time = (datetime.now() - processing_start).total_seconds()
            results['processing_time_seconds'] = processing_time

            logger.info(f"✅ Comprehensive processing completed: {results['alerts_processed']}/{len(alerts)} alerts in {processing_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"❌ Comprehensive alert processing failed: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results

    def process_with_universal_ai(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Process alerts with Universal AI system"""
        try:
            if not UNIVERSAL_AI_AVAILABLE or not self.universal_system:
                return {'status': 'unavailable', 'message': 'Universal AI system not available'}

            results = []
            for alert in alerts:
                try:
                    alert_name = alert.get('labels', {}).get('alertname', 'Unknown')
                    task_description = f"Analyze and handle Prometheus alert: {alert_name}"
                    solution = self.universal_system.solve_task(task_description)

                    results.append({
                        'alert': alert_name,
                        'solution': str(solution),  # Convert to string for JSON serialization
                        'status': 'processed'
                    })
                except Exception as e:
                    results.append({
                        'alert': alert.get('labels', {}).get('alertname', 'Unknown'),
                        'error': str(e),
                        'status': 'error'
                    })

            return {
                'status': 'processed',
                'results': results,
                'total_alerts': len(alerts),
                'successful_processing': len([r for r in results if r['status'] == 'processed'])
            }

        except Exception as e:
            logger.error(f"❌ Universal AI processing failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def generate_alert_fingerprint(self, alert_data: Dict) -> str:
        """Generate unique fingerprint for alert"""
        labels = alert_data.get('labels', {})
        key_fields = [
            labels.get('alertname', ''),
            labels.get('instance', ''),
            labels.get('job', ''),
            labels.get('severity', '')
        ]
        fingerprint_string = '|'.join(key_fields)
        return hashlib.md5(fingerprint_string.encode()).hexdigest()

    def start_background_tasks(self):
        """Start background maintenance tasks"""
        def cleanup_task():
            while not self.shutdown_event.is_set():
                try:
                    # Cleanup old data
                    self.db_manager.cleanup_old_data(self.config.alert_retention_days)

                    # Sleep for 1 hour
                    self.shutdown_event.wait(3600)
                except Exception as e:
                    logger.error(f"❌ Background cleanup task failed: {e}")
                    self.shutdown_event.wait(300)  # Wait 5 minutes on error

        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown()

    def start(self):
        """Start the alert receiver server"""
        if not FLASK_AVAILABLE:
            logger.error("❌ Flask not available - cannot start alert receiver")
            return False

        if self.running:
            logger.warning("⚠️ Alert receiver already running")
            return True

        try:
            self.running = True

            # Create server
            self.server = make_server(
                self.config.host,
                self.config.port,
                self.app,
                threaded=True
            )

            # Start server in thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True
            )
            self.server_thread.start()

            logger.info(f"🚀 Comprehensive Prometheus Alert Receiver started on {self.config.host}:{self.config.port}")
            logger.info(f"   Integration Mode: {self.config.integration_mode}")
            logger.info(f"   Enhanced Processor: {'✅' if self.enhanced_processor else '❌'}")
            logger.info(f"   AI Processor: {'✅' if self.ai_alert_processor else '❌'}")
            logger.info(f"   Universal AI: {'✅' if self.universal_system else '❌'}")
            logger.info(f"   Precheck Engine: {'✅' if self.precheck_engine else '❌'}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to start alert receiver: {e}")
            self.running = False
            return False

    def shutdown(self):
        """Shutdown the alert receiver gracefully"""
        logger.info("🛑 Shutting down Prometheus Alert Receiver...")

        self.running = False
        self.shutdown_event.set()

        # Shutdown server
        if self.server:
            self.server.shutdown()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("✅ Prometheus Alert Receiver shutdown complete")

    def is_running(self) -> bool:
        """Check if the receiver is running"""
        return self.running and (self.server_thread and self.server_thread.is_alive())

    # Flask endpoint implementations
    def get_health_status(self):
        """Get health status"""
        try:
            health_data = {
                'status': 'healthy' if self.running else 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'integration_mode': self.config.integration_mode,
                'components': {
                    'database': 'connected',
                    'correlation_engine': 'active' if self.correlation_engine else 'inactive',
                    'notification_manager': 'active' if self.notification_manager else 'inactive',
                    'enhanced_processor': 'active' if self.enhanced_processor else 'inactive',
                    'ai_processor': 'active' if self.ai_alert_processor else 'inactive',
                    'universal_ai': 'active' if self.universal_system else 'inactive',
                    'precheck_engine': 'active' if self.precheck_engine else 'inactive'
                },
                'metrics': self.metrics
            }
            return jsonify(health_data)
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    def get_comprehensive_status(self):
        """Get comprehensive status"""
        try:
            status_data = {
                'running': self.running,
                'config': asdict(self.config),
                'metrics': self.metrics,
                'components': {
                    'database_manager': bool(self.db_manager),
                    'correlation_engine': bool(self.correlation_engine),
                    'notification_manager': bool(self.notification_manager),
                    'enhanced_processor': bool(self.enhanced_processor),
                    'ai_alert_processor': bool(self.ai_alert_processor),
                    'universal_system': bool(self.universal_system),
                    'precheck_engine': bool(self.precheck_engine)
                },
                'correlation_rules': len(self.correlation_engine.correlation_rules) if self.correlation_engine else 0,
                'notification_channels': len(self.notification_manager.channels) if self.notification_manager else 0,
                'performance_stats': self.db_manager.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }

            # Add enhanced processor stats if available
            if self.enhanced_processor:
                status_data['enhanced_stats'] = self.enhanced_processor.get_processing_statistics()

            return jsonify(status_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def get_prometheus_metrics(self):
        """Get Prometheus metrics"""
        try:
            metrics_lines = [
                "# HELP prometheus_alert_receiver_alerts_received_total Total alerts received",
                "# TYPE prometheus_alert_receiver_alerts_received_total counter",
                f"prometheus_alert_receiver_alerts_received_total {self.metrics['alerts_received']}",
                "",
                "# HELP prometheus_alert_receiver_alerts_processed_total Total alerts processed",
                "# TYPE prometheus_alert_receiver_alerts_processed_total counter",
                f"prometheus_alert_receiver_alerts_processed_total {self.metrics['alerts_processed']}",
                "",
                "# HELP prometheus_alert_receiver_correlations_found_total Total correlations found",
                "# TYPE prometheus_alert_receiver_correlations_found_total counter",
                f"prometheus_alert_receiver_correlations_found_total {self.metrics['correlations_found']}",
                "",
                "# HELP prometheus_alert_receiver_notifications_sent_total Total notifications sent",
                "# TYPE prometheus_alert_receiver_notifications_sent_total counter",
                f"prometheus_alert_receiver_notifications_sent_total {self.metrics['notifications_sent']}",
                "",
                "# HELP prometheus_alert_receiver_processing_errors_total Total processing errors",
                "# TYPE prometheus_alert_receiver_processing_errors_total counter",
                f"prometheus_alert_receiver_processing_errors_total {self.metrics['processing_errors']}",
                "",
                "# HELP prometheus_alert_receiver_integration_calls_total Total integration calls",
                "# TYPE prometheus_alert_receiver_integration_calls_total counter",
                f"prometheus_alert_receiver_integration_calls_total {self.metrics['integration_calls']}",
                "",
                "# HELP prometheus_alert_receiver_performance_avg_ms Average processing time in milliseconds",
                "# TYPE prometheus_alert_receiver_performance_avg_ms gauge",
                f"prometheus_alert_receiver_performance_avg_ms {self.metrics['performance_avg_ms']}",
                "",
                "# HELP prometheus_alert_receiver_running Receiver running status",
                "# TYPE prometheus_alert_receiver_running gauge",
                f"prometheus_alert_receiver_running {1 if self.running else 0}",
                ""
            ]

            return '\n'.join(metrics_lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}

        except Exception as e:
            return f"# Error generating metrics: {e}\n", 500, {'Content-Type': 'text/plain'}

    def get_integration_status(self):
        """Get integration status with existing processors"""
        try:
            integration_status = {
                'integration_mode': self.config.integration_mode,
                'available_integrations': {
                    'existing_processor_available': EXISTING_PROCESSOR_AVAILABLE,
                    'universal_ai_available': UNIVERSAL_AI_AVAILABLE,
                    'precheck_engine_available': PRECHECK_ENGINE_AVAILABLE,
                    'flask_available': FLASK_AVAILABLE
                },
                'active_integrations': {
                    'enhanced_processor': self.enhanced_processor is not None,
                    'ai_alert_processor': self.ai_alert_processor is not None,
                    'universal_system': self.universal_system is not None,
                    'precheck_engine': self.precheck_engine is not None
                },
                'integration_metrics': {
                    'total_integration_calls': self.metrics['integration_calls'],
                    'avg_performance_ms': self.metrics['performance_avg_ms']
                },
                'timestamp': datetime.now().isoformat()
            }

            return jsonify(integration_status)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def test_integration_endpoint(self):
        """Test integration with existing processors"""
        try:
            test_alert = {
                'labels': {
                    'alertname': 'IntegrationTestAlert',
                    'severity': 'warning',
                    'instance': 'localhost:8000',
                    'job': 'test-service'
                },
                'annotations': {
                    'description': 'Integration test alert',
                    'summary': 'Test alert for integration validation'
                },
                'status': 'firing',
                'startsAt': datetime.now().isoformat()
            }

            test_results = {}

            # Test enhanced processor integration
            if self.enhanced_processor:
                try:
                    enhanced_result = self.enhanced_processor.process_alerts_enhanced([test_alert])
                    test_results['enhanced_processor'] = {
                        'status': 'success',
                        'result': enhanced_result
                    }
                except Exception as e:
                    test_results['enhanced_processor'] = {
                        'status': 'error',
                        'error': str(e)
                    }

            # Test Universal AI integration
            if self.universal_system:
                try:
                    universal_result = self.process_with_universal_ai([test_alert])
                    test_results['universal_ai'] = {
                        'status': 'success',
                        'result': universal_result
                    }
                except Exception as e:
                    test_results['universal_ai'] = {
                        'status': 'error',
                        'error': str(e)
                    }

            return jsonify({
                'status': 'completed',
                'test_results': test_results,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    def test_alert_processing_endpoint(self):
        """Test alert processing endpoint"""
        try:
            test_alerts = [{
                'labels': {
                    'alertname': 'TestAlert',
                    'severity': 'warning',
                    'instance': 'localhost:8000',
                    'job': 'test-service'
                },
                'annotations': {
                    'description': 'Test alert for validation',
                    'summary': 'Test alert'
                },
                'status': 'firing',
                'startsAt': datetime.now().isoformat()
            }]

            # Process based on integration mode
            if self.config.integration_mode == 'basic':
                result = self.process_alerts_basic(test_alerts, '127.0.0.1', 'test-client')
            elif self.config.integration_mode == 'enhanced':
                result = self.process_alerts_enhanced(test_alerts, '127.0.0.1', 'test-client')
            else:
                result = self.process_alerts_comprehensive(test_alerts, '127.0.0.1', 'test-client')

            return jsonify({
                'status': 'success',
                'message': 'Test alert processed successfully',
                'result': result
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    def render_dashboard(self):
        """Render web dashboard"""
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prometheus Alert Receiver Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric h3 { margin-top: 0; color: #2c3e50; }
        .status-ok { color: #27ae60; font-weight: bold; }
        .status-error { color: #e74c3c; font-weight: bold; }
        .status-warning { color: #f39c12; font-weight: bold; }
        .integration-badge { display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 12px; margin: 2px; }
        .badge-active { background: #27ae60; color: white; }
        .badge-inactive { background: #95a5a6; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Prometheus Alert Receiver Dashboard</h1>
            <p>Comprehensive alert processing with AI integration</p>
        </div>

        <div class="metric-grid">
            <div class="metric">
                <h3>System Status</h3>
                <p>Status: <span class="{{ 'status-ok' if running else 'status-error' }}">{{ 'Running' if running else 'Stopped' }}</span></p>
                <p>Integration Mode: <span class="status-ok">{{ integration_mode }}</span></p>
                <p>Port: {{ port }}</p>
            </div>

            <div class="metric">
                <h3>Processing Metrics</h3>
                <p>Alerts Received: <strong>{{ metrics.alerts_received }}</strong></p>
                <p>Alerts Processed: <strong>{{ metrics.alerts_processed }}</strong></p>
                <p>Processing Errors: <strong>{{ metrics.processing_errors }}</strong></p>
                <p>Avg Performance: <strong>{{ "%.2f"|format(metrics.performance_avg_ms) }}ms</strong></p>
            </div>

            <div class="metric">
                <h3>Correlation & Notifications</h3>
                <p>Correlations Found: <strong>{{ metrics.correlations_found }}</strong></p>
                <p>Notifications Sent: <strong>{{ metrics.notifications_sent }}</strong></p>
                <p>Integration Calls: <strong>{{ metrics.integration_calls }}</strong></p>
            </div>

            <div class="metric">
                <h3>Active Integrations</h3>
                <div>
                    <span class="integration-badge {{ 'badge-active' if enhanced_processor else 'badge-inactive' }}">Enhanced Processor</span>
                    <span class="integration-badge {{ 'badge-active' if ai_processor else 'badge-inactive' }}">AI Processor</span>
                    <span class="integration-badge {{ 'badge-active' if universal_ai else 'badge-inactive' }}">Universal AI</span>
                    <span class="integration-badge {{ 'badge-active' if precheck_engine else 'badge-inactive' }}">Precheck Engine</span>
                </div>
            </div>

            <div class="metric">
                <h3>Quick Actions</h3>
                <p><a href="/test/alert" style="color: #3498db;">Test Alert Processing</a></p>
                <p><a href="/integration/status" style="color: #3498db;">Integration Status</a></p>
                <p><a href="/metrics" style="color: #3498db;">Prometheus Metrics</a></p>
                <p><a href="/health" style="color: #3498db;">Health Check</a></p>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return render_template_string(dashboard_html,
                                    running=self.running,
                                    integration_mode=self.config.integration_mode,
                                    port=self.config.port,
                                    metrics=self.metrics,
                                    enhanced_processor=self.enhanced_processor is not None,
                                    ai_processor=self.ai_alert_processor is not None,
                                    universal_ai=self.universal_system is not None,
                                    precheck_engine=self.precheck_engine is not None)

# Test and utility functions
def test_comprehensive_receiver():
    """Test the comprehensive alert receiver"""
    print("🧪 Testing Comprehensive Prometheus Alert Receiver...")

    config = AlertReceiverConfig(port=8054, integration_mode='enhanced')  # Use different port for testing
    receiver = PrometheusAlertReceiver(config)

    if not FLASK_AVAILABLE:
        print("❌ Flask not available for testing")
        return False

    # Test alert processing
    test_alerts = [{
        'labels': {
            'alertname': 'TestAlert',
            'severity': 'warning',
            'instance': 'localhost:8000',
            'job': 'test-service'
        },
        'annotations': {
            'description': 'This is a comprehensive test alert',
            'summary': 'Test alert for validation'
        },
        'status': 'firing',
        'startsAt': datetime.now().isoformat(),
        'fingerprint': 'test123'
    }]

    try:
        result = receiver.process_alerts_enhanced(test_alerts, '127.0.0.1', 'test-agent')
        print(f"✅ Test alerts processed: {result['status']}")
        print(f"   Mode: {result['mode']}")
        print(f"   Alerts processed: {result['alerts_processed']}/{result['alerts_received']}")
        print(f"   Correlations found: {result['correlations_found']}")
        print(f"   Processing time: {result.get('processing_time_seconds', 0):.2f}s")

        if result['errors']:
            print(f"⚠️ Errors encountered: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")

        # Test integration status
        if receiver.enhanced_processor:
            print("✅ Enhanced processor integration working")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    # Run comprehensive test
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_comprehensive_receiver()
    else:
        # Start the receiver
        config = AlertReceiverConfig(integration_mode='enhanced')  # Default to enhanced mode
        receiver = PrometheusAlertReceiver(config)

        try:
            if receiver.start():
                logger.info("🚀 Prometheus Alert Receiver started successfully")
                # Keep running until interrupted
                while receiver.is_running():
                    time.sleep(1)
            else:
                logger.error("❌ Failed to start Prometheus Alert Receiver")
                sys.exit(1)
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        finally:
            receiver.shutdown()
