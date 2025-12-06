
#!/usr/bin/env python3
"""
Real-Time Intel Precheck Monitoring System
Automatically detects ingredients and processes them through Neural AI
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import signal
import psutil
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingredient.ingredient_management import IngredientManager


# Add your aiengine to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import your components
try:
    from enhanced_precheck_neural import EnhancedPrecheckEngine, NeuralPrecheckProcessor
    from core.precheck_engine import PrecheckFailure, IngredientType
    from main import WorldClassUniversalNeuralSystem
    NEURAL_SYSTEM_AVAILABLE = True
    print("✅ Neural precheck system imported successfully")
except ImportError as e:
    print(f"❌ Could not import neural system: {e}")
    NEURAL_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_precheck.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# REAL-TIME INGREDIENT DETECTION
# ============================================================================

@dataclass
class IngredientEvent:
    """Represents an ingredient event for processing"""
    ingredient_id: str
    ingredient_name: str
    ingredient_type: str  # 'intel' or 'tpv'
    file_path: str
    milestone: str
    timestamp: float
    metadata: Dict[str, Any]
    priority: str = "medium"  # low, medium, high, critical

    def to_dict(self):
        return asdict(self)

class IngredientDetector:
    """Detects new ingredients in real-time"""

    def __init__(self, watch_directories: List[str], callback: Callable):
        self.watch_directories = watch_directories
        self.callback = callback
        self.observers = []
        self.processed_files = set()
        self.logger = logging.getLogger(f"{__name__}.IngredientDetector")

        # File patterns to watch
        self.ingredient_patterns = [
            '*.dll', '*.exe', '*.sys', '*.inf', '*.cat', '*.msi',
            '*.zip', '*.cab', '*.pkg', '*.rpm', '*.deb'
        ]

        # Ingredient type detection rules
        self.intel_indicators = [
            'intel', 'igfx', 'gfx', 'graphics', 'wifi', 'bluetooth',
            'chipset', 'audio', 'network', 'storage'
        ]

        self.tpv_indicators = [
            'nvidia', 'amd', 'realtek', 'broadcom', 'qualcomm',
            'marvell', 'atheros', 'synaptics', 'elan'
        ]

    def start_monitoring(self):
        """Start real-time monitoring"""
        self.logger.info(f"🔍 Starting ingredient monitoring on {len(self.watch_directories)} directories")

        for directory in self.watch_directories:
            if os.path.exists(directory):
                event_handler = IngredientFileHandler(self)
                observer = Observer()
                observer.schedule(event_handler, directory, recursive=True)
                observer.start()
                self.observers.append(observer)
                self.logger.info(f"   📁 Monitoring: {directory}")
            else:
                self.logger.warning(f"   ⚠️ Directory not found: {directory}")

        # Initial scan of existing files
        self._initial_scan()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.logger.info("🛑 Stopping ingredient monitoring")
        for observer in self.observers:
            observer.stop()
            observer.join()

    def _initial_scan(self):
        """Scan existing files on startup"""
        self.logger.info("🔍 Performing initial ingredient scan...")

        for directory in self.watch_directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_ingredient_file(file_path):
                            self._process_ingredient_file(file_path, is_initial_scan=True)

    def _is_ingredient_file(self, file_path: str) -> bool:
        """Check if file is an ingredient"""
        file_ext = Path(file_path).suffix.lower()
        return any(pattern.replace('*', '') == file_ext for pattern in self.ingredient_patterns)

    def _process_ingredient_file(self, file_path: str, is_initial_scan: bool = False):
        """Process detected ingredient file"""
        try:
            # Avoid duplicate processing
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                return

            self.processed_files.add(file_hash)

            # Extract ingredient information
            ingredient_info = self._extract_ingredient_info(file_path)

            # Create ingredient event
            event = IngredientEvent(
                ingredient_id=f"ing_{int(time.time())}_{file_hash[:8]}",
                ingredient_name=ingredient_info['name'],
                ingredient_type=ingredient_info['type'],
                file_path=file_path,
                milestone=ingredient_info['milestone'],
                timestamp=time.time(),
                metadata={
                    'file_size': os.path.getsize(file_path),
                    'file_hash': file_hash,
                    'detection_method': 'initial_scan' if is_initial_scan else 'real_time',
                    'file_extension': Path(file_path).suffix,
                    'directory': os.path.dirname(file_path)
                },
                priority=ingredient_info['priority']
            )

            # Trigger callback
            self.callback(event)

        except Exception as e:
            self.logger.error(f"❌ Error processing ingredient file {file_path}: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read(1024)).hexdigest()  # Hash first 1KB for speed
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()

    def _extract_ingredient_info(self, file_path: str) -> Dict[str, Any]:
        """Extract ingredient information from file"""
        file_name = Path(file_path).name.lower()

        # Determine ingredient type
        ingredient_type = 'unknown'
        if any(indicator in file_name for indicator in self.intel_indicators):
            ingredient_type = 'intel'
        elif any(indicator in file_name for indicator in self.tpv_indicators):
            ingredient_type = 'tpv'
        else:
            # Default based on directory structure
            if 'intel' in file_path.lower():
                ingredient_type = 'intel'
            elif any(vendor in file_path.lower() for vendor in self.tpv_indicators):
                ingredient_type = 'tpv'
            else:
                ingredient_type = 'intel'  # Default to Intel

        # Determine milestone from path
        milestone = 'development'
        path_lower = file_path.lower()
        if 'alpha' in path_lower:
            milestone = 'alpha'
        elif 'beta' in path_lower:
            milestone = 'beta'
        elif 'rtm' in path_lower or 'release' in path_lower:
            milestone = 'rtm'
        elif 'production' in path_lower:
            milestone = 'production'

        # Determine priority
        priority = 'medium'
        if 'critical' in file_name or 'security' in file_name:
            priority = 'critical'
        elif 'driver' in file_name or 'firmware' in file_name:
            priority = 'high'
        elif 'test' in file_name or 'debug' in file_name:
            priority = 'low'

        return {
            'name': Path(file_path).name,
            'type': ingredient_type,
            'milestone': milestone,
            'priority': priority
        }

class IngredientFileHandler(FileSystemEventHandler):
    """File system event handler for ingredient detection"""

    def __init__(self, detector: IngredientDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.IngredientFileHandler")

    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and self.detector._is_ingredient_file(event.src_path):
            self.logger.info(f"📦 New ingredient detected: {event.src_path}")
            # Small delay to ensure file is fully written
            threading.Timer(2.0, lambda: self.detector._process_ingredient_file(event.src_path)).start()

    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and self.detector._is_ingredient_file(event.src_path):
            self.logger.debug(f"📝 Ingredient modified: {event.src_path}")
            # Process modified files as they might be updated ingredients
            threading.Timer(1.0, lambda: self.detector._process_ingredient_file(event.src_path)).start()

# ============================================================================
# REAL-TIME PRECHECK PROCESSOR
# ============================================================================

class RealTimePrecheckProcessor:
    """Real-time precheck processor with neural AI"""

    def __init__(self, universal_system=None):
        self.logger = logging.getLogger(f"{__name__}.RealTimePrecheckProcessor")

        # Initialize neural precheck engine
        if NEURAL_SYSTEM_AVAILABLE and universal_system:
            self.precheck_engine = EnhancedPrecheckEngine(universal_system)
            self.neural_available = True
        else:
            self.precheck_engine = None
            self.neural_available = False

        # Processing queue
        self.processing_queue = asyncio.Queue()
        self.processing_active = False

        # Results storage
        self.results_db = RealTimeResultsDB()

        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'auto_approved': 0,
            'auto_rejected': 0,
            'escalated': 0,
            'manual_review': 0,
            'processing_errors': 0,
            'average_processing_time': 0.0,
            'start_time': time.time()
        }

        # Notification system
        self.notification_system = RealTimeNotificationSystem()

        self.logger.info(f"🧠 Real-time precheck processor initialized (Neural: {'Available' if self.neural_available else 'Unavailable'})")

    async def start_processing(self):
        """Start real-time processing"""
        self.processing_active = True
        self.logger.info("🚀 Starting real-time precheck processing")

        # Start processing loop
        asyncio.create_task(self._processing_loop())


    async def stop_processing(self):
        """Stop processing"""
        self.processing_active = False
        self.logger.info("🛑 Stopping real-time precheck processing")

    async def queue_ingredient(self, ingredient_event: IngredientEvent):
        """Queue ingredient for processing"""
        await self.processing_queue.put(ingredient_event)
        self.logger.info(f"📥 Queued ingredient: {ingredient_event.ingredient_name} (Priority: {ingredient_event.priority})")

    async def _processing_loop(self):
        """Main processing loop"""
        self.logger.info("🔄 Processing loop started")

        while self.processing_active:
            try:
                # Get ingredient from queue (with timeout)
                try:
                    ingredient_event = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process ingredient
                await self._process_ingredient_event(ingredient_event)

            except Exception as e:
                self.logger.error(f"❌ Processing loop error: {e}")
                await asyncio.sleep(1)

    async def _process_ingredient_event(self, ingredient_event: IngredientEvent):
        """Process individual ingredient event"""
        start_time = time.time()

        try:
            self.logger.info(f"🔍 Processing ingredient: {ingredient_event.ingredient_name}")

            # Run precheck simulations
            precheck_results = await self._run_prechecks(ingredient_event)

            # Process each precheck result
            final_results = []
            for precheck_result in precheck_results:
                if precheck_result['has_failure']:
                    # Create precheck failure
                    failure = self._create_precheck_failure(ingredient_event, precheck_result)

                    # Process with neural AI
                    if self.neural_available:
                        decision_result = await self.precheck_engine.analyze_precheck_failure_with_neural_ai(failure)
                    else:
                        decision_result = await self._fallback_processing(failure)

                    # Store result
                    result_record = self._create_result_record(ingredient_event, precheck_result, decision_result)
                    final_results.append(result_record)

                    # Send notifications
                    await self._send_notifications(ingredient_event, decision_result)
                else:
                    # Precheck passed
                    result_record = self._create_pass_record(ingredient_event, precheck_result)
                    final_results.append(result_record)

            # Store all results
            for result in final_results:
                self.results_db.store_result(result)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(final_results, processing_time)

            self.logger.info(f"✅ Completed processing {ingredient_event.ingredient_name} in {processing_time:.3f}s")

        except Exception as e:
            self.logger.error(f"❌ Error processing ingredient {ingredient_event.ingredient_name}: {e}")
            self.stats['processing_errors'] += 1

    async def _run_prechecks(self, ingredient_event: IngredientEvent) -> List[Dict[str, Any]]:
        """Simulate running prechecks on ingredient"""
        # This simulates various precheck types that might run
        precheck_types = [
            'MsftSignChk', 'SignChk', 'SDLeChk', 'BATChk', 'VersionChk',
            'PathLengthChk', 'INFCompChk', 'DriverChk', 'BinaryScanChk'
        ]

        results = []

        for precheck_name in precheck_types:
            # Simulate precheck execution
            await asyncio.sleep(0.1)  # Simulate processing time

            # Determine if precheck passes or fails (simulation)
            has_failure = self._simulate_precheck_result(ingredient_event, precheck_name)

            result = {
                'precheck_name': precheck_name,
                'has_failure': has_failure,
                'execution_time': 0.1,
                'error_message': self._generate_error_message(precheck_name) if has_failure else None,
                'severity': self._determine_severity(precheck_name, ingredient_event) if has_failure else None
            }

            results.append(result)

        return results

    def _simulate_precheck_result(self, ingredient_event: IngredientEvent, precheck_name: str) -> bool:
        """Simulate precheck result (in real system, this would be actual precheck execution)"""
        # Simulation logic - in real system, this would call actual precheck tools

        # Higher failure rate for certain conditions
        base_failure_rate = 0.1  # 10% base failure rate

        # Adjust based on ingredient type
        if ingredient_event.ingredient_type == 'tpv':
            failure_rate = base_failure_rate * 0.5  # TPV components fail less often
        else:
            failure_rate = base_failure_rate

        # Adjust based on precheck type
        if precheck_name in ['MsftSignChk', 'SignChk']:
            failure_rate *= 2.0  # Signature checks fail more often
        elif precheck_name in ['BATChk', 'VersionChk']:
            failure_rate *= 0.5  # These checks are more lenient

        # Adjust based on milestone
        if ingredient_event.milestone in ['rtm', 'production']:
            failure_rate *= 1.5  # Stricter checks for release milestones

        # Random failure based on calculated rate
        import random
        return random.random() < failure_rate

    def _generate_error_message(self, precheck_name: str) -> str:
        """Generate realistic error message for failed precheck"""
        error_messages = {
            'MsftSignChk': 'Microsoft signature verification failed - certificate not found',
            'SignChk': 'Digital signature validation failed - signature invalid or expired',
            'SDLeChk': 'Security Development Lifecycle check failed - banned API detected',
            'BATChk': 'Binary Analysis Tool detected potential security vulnerability',
            'VersionChk': 'Version information missing or inconsistent',
            'PathLengthChk': 'File path exceeds maximum allowed length (260 characters)',
            'INFCompChk': 'INF file compatibility check failed - unsupported OS version',
            'DriverChk': 'Driver package validation failed - missing required files',
            'BinaryScanChk': 'Binary scan detected suspicious code patterns'
        }

        return error_messages.get(precheck_name, f'{precheck_name} validation failed')

    def _determine_severity(self, precheck_name: str, ingredient_event: IngredientEvent) -> str:
        """Determine severity of precheck failure"""
        # Critical severity for security-related checks
        if precheck_name in ['MsftSignChk', 'SignChk', 'SDLeChk', 'BinaryScanChk']:
            return 'critical'

        # High severity for driver-related issues in production
        if precheck_name == 'DriverChk' and ingredient_event.milestone in ['rtm', 'production']:
            return 'high'

        # Medium severity for most other issues
        if precheck_name in ['INFCompChk', 'VersionChk']:
            return 'medium'

        # Low severity for path and formatting issues
        return 'low'

    def _create_precheck_failure(self, ingredient_event: IngredientEvent, precheck_result: Dict[str, Any]) -> 'PrecheckFailure':
        """Create PrecheckFailure object from ingredient event and precheck result"""

        ingredient_type = IngredientType.INTEL if ingredient_event.ingredient_type == 'intel' else IngredientType.TPV

        return PrecheckFailure(
            failure_id=f"fail_{ingredient_event.ingredient_id}_{precheck_result['precheck_name']}",
            precheck_name=precheck_result['precheck_name'],
            ingredient_type=ingredient_type,
            ingredient_name=ingredient_event.ingredient_name,
            failure_details={
                'file_path': ingredient_event.file_path,
                'file_size': ingredient_event.metadata.get('file_size', 0),
                'detection_method': ingredient_event.metadata.get('detection_method', 'unknown'),
                'precheck_execution_time': precheck_result['execution_time']
            },
            milestone=ingredient_event.milestone,
            timestamp=ingredient_event.timestamp,
            severity=precheck_result['severity'],
            error_message=precheck_result['error_message'],
            metadata={
                'ingredient_event': ingredient_event.to_dict(),
                'real_time_processing': True,
                'priority': ingredient_event.priority
            }
        )

    async def _fallback_processing(self, failure: 'PrecheckFailure'):
        """Fallback processing when neural AI is not available"""
        # Simple rule-based processing
        if failure.ingredient_type == IngredientType.TPV:
            decision = 'AUTO_APPROVE'
            confidence = 0.8
        elif failure.severity == 'critical':
            decision = 'ESCALATE_BA'
            confidence = 0.9
        else:
            decision = 'MANUAL_REVIEW'
            confidence = 0.6

        # Create mock decision result
        class MockDecisionResult:
            def __init__(self):
                self.decision = type('Decision', (), {'value': decision})()
                self.confidence = confidence
                self.reasoning = f"Fallback processing: {decision}"
                self.ai_analysis = {'neural_network_used': False, 'fallback_used': True}

        return MockDecisionResult()

    def _create_result_record(self, ingredient_event: IngredientEvent, precheck_result: Dict[str, Any], decision_result) -> Dict[str, Any]:
        """Create result record for storage"""
        return {
            'timestamp': time.time(),
            'ingredient_id': ingredient_event.ingredient_id,
            'ingredient_name': ingredient_event.ingredient_name,
            'ingredient_type': ingredient_event.ingredient_type,
            'file_path': ingredient_event.file_path,
            'milestone': ingredient_event.milestone,
            'priority': ingredient_event.priority,
            'precheck_name': precheck_result['precheck_name'],
            'precheck_passed': False,
            'decision': decision_result.decision.value,
            'confidence': decision_result.confidence,
            'reasoning': decision_result.reasoning,
            'severity': precheck_result['severity'],
            'error_message': precheck_result['error_message'],
            'neural_used': decision_result.ai_analysis.get('neural_network_used', False),
            'processing_type': 'real_time_failure'
        }

    def _create_pass_record(self, ingredient_event: IngredientEvent, precheck_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create record for passed precheck"""
        return {
            'timestamp': time.time(),
            'ingredient_id': ingredient_event.ingredient_id,
            'ingredient_name': ingredient_event.ingredient_name,
            'ingredient_type': ingredient_event.ingredient_type,
            'file_path': ingredient_event.file_path,
            'milestone': ingredient_event.milestone,
            'priority': ingredient_event.priority,
            'precheck_name': precheck_result['precheck_name'],
            'precheck_passed': True,
            'decision': 'PASSED',
            'confidence': 1.0,
            'reasoning': f'{precheck_result["precheck_name"]} validation passed successfully',
            'severity': None,
            'error_message': None,
            'neural_used': False,
            'processing_type': 'real_time_pass'
        }

    async def _send_notifications(self, ingredient_event: IngredientEvent, decision_result):
        """Send notifications for critical decisions"""
        if hasattr(decision_result.decision, 'value'):
            decision_value = decision_result.decision.value
        else:
            decision_value = str(decision_result.decision)

        # Send notifications for critical decisions
        if decision_value in ['ESCALATE_BA', 'AUTO_REJECT'] or ingredient_event.priority == 'critical':
            await self.notification_system.send_notification(
                f"🚨 Critical Precheck Decision: {ingredient_event.ingredient_name}",
                f"Decision: {decision_value} (Confidence: {decision_result.confidence:.1%})\n"
                f"Reasoning: {decision_result.reasoning}"
            )

    def _update_stats(self, results: List[Dict[str, Any]], processing_time: float):
        """Update processing statistics"""
        self.stats['total_processed'] += len(results)

        for result in results:
            decision = result['decision']
            if decision == 'AUTO_APPROVE' or decision == 'PASSED':
                self.stats['auto_approved'] += 1
            elif decision == 'AUTO_REJECT':
                self.stats['auto_rejected'] += 1
            elif 'ESCALATE' in decision:
                self.stats['escalated'] += 1
            elif decision == 'MANUAL_REVIEW':
                self.stats['manual_review'] += 1

        # Update average processing time
        total_processed = self.stats['total_processed']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg * (total_processed - len(results)) + processing_time) / total_processed
        )

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics"""
        uptime = time.time() - self.stats['start_time']

        return {
            **self.stats,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'processing_rate': self.stats['total_processed'] / max(uptime / 60, 1),  # per minute
            'neural_available': self.neural_available,
            'queue_size': self.processing_queue.qsize() if hasattr(self.processing_queue, 'qsize') else 0
        }

# ============================================================================
# RESULTS DATABASE
# ============================================================================

class RealTimeResultsDB:
    """SQLite database for storing real-time results"""

    def __init__(self, db_path: str = "realtime_precheck_results.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.RealTimeResultsDB")
        self._initialize_db()

    def _initialize_db(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precheck_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    ingredient_id TEXT,
                    ingredient_name TEXT,
                    ingredient_type TEXT,
                    file_path TEXT,
                    milestone TEXT,
                    priority TEXT,
                    precheck_name TEXT,
                    precheck_passed BOOLEAN,
                    decision TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    severity TEXT,
                    error_message TEXT,
                    neural_used BOOLEAN,
                    processing_type TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON precheck_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_id ON precheck_results(ingredient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decision ON precheck_results(decision)')

            conn.commit()
            conn.close()

            self.logger.info(f"✅ Results database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")

    def store_result(self, result: Dict[str, Any]):
        """Store precheck result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO precheck_results (
                    timestamp, ingredient_id, ingredient_name, ingredient_type,
                    file_path, milestone, priority, precheck_name, precheck_passed,
                    decision, confidence, reasoning, severity, error_message,
                    neural_used, processing_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'], result['ingredient_id'], result['ingredient_name'],
                result['ingredient_type'], result['file_path'], result['milestone'],
                result['priority'], result['precheck_name'], result['precheck_passed'],
                result['decision'], result['confidence'], result['reasoning'],
                result['severity'], result['error_message'], result['neural_used'],
                result['processing_type']
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"❌ Failed to store result: {e}")

    def get_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent results"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM precheck_results
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            results = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return results

        except Exception as e:
            self.logger.error(f"❌ Failed to get recent results: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total results
            cursor.execute('SELECT COUNT(*) FROM precheck_results')
            total_results = cursor.fetchone()[0]

            # Results by decision
            cursor.execute('''
                SELECT decision, COUNT(*)
                FROM precheck_results
                GROUP BY decision
            ''')
            decision_counts = dict(cursor.fetchall())

            # Results by ingredient type
            cursor.execute('''
                SELECT ingredient_type, COUNT(*)
                FROM precheck_results
                GROUP BY ingredient_type
            ''')
            type_counts = dict(cursor.fetchall())

            # Neural usage
            cursor.execute('''
                SELECT neural_used, COUNT(*)
                FROM precheck_results
                GROUP BY neural_used
            ''')
            neural_counts = dict(cursor.fetchall())

            conn.close()

            return {
                'total_results': total_results,
                'decision_distribution': decision_counts,
                'ingredient_type_distribution': type_counts,
                'neural_usage': neural_counts
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {}

# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================

class RealTimeNotificationSystem:
    """Real-time notification system"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealTimeNotificationSystem")
        self.email_enabled = True  # Configure as needed
        self.slack_enabled = False  # Configure as needed

    async def send_notification(self, subject: str, message: str):
        """Send notification through available channels"""
        try:
            # Log notification
            self.logger.info(f"📢 Notification: {subject}")

            # Email notification (if configured)
            if self.email_enabled:
                await self._send_email_notification(subject, message)

            # Slack notification (if configured)
            if self.slack_enabled:
                await self._send_slack_notification(subject, message)

            # Console notification (always)
            print(f"\n🚨 NOTIFICATION: {subject}")
            print(f"   {message}")
            print()

        except Exception as e:
            self.logger.error(f"❌ Failed to send notification: {e}")

    async def _send_email_notification(self, subject: str, message: str):
        """Send email notification"""
        # Implement email sending logic here
        # For now, just log
        self.logger.info(f"📧 Email notification: {subject}")

    async def _send_slack_notification(self, subject: str, message: str):
        """Send Slack notification"""
        # Implement Slack webhook logic here
        # For now, just log
        self.logger.info(f"💬 Slack notification: {subject}")

# ============================================================================
# MAIN REAL-TIME SYSTEM
# ============================================================================

class RealTimePrecheckSystem:
    """Main real-time precheck system"""

    def __init__(self, watch_directories: List[str]):
        self.ingredient_manager = IngredientManager()
        self.logger = logging.getLogger(f"{__name__}.RealTimePrecheckSystem")

        # Initialize universal neural system
        if NEURAL_SYSTEM_AVAILABLE:
            self.logger.info("🧠 Initializing Universal Neural System...")
            self.universal_system = WorldClassUniversalNeuralSystem()
        else:
            self.universal_system = None

        # Initialize components
        self.processor = RealTimePrecheckProcessor(self.universal_system)
        self.detector = IngredientDetector(watch_directories, self._on_ingredient_detected)

        # System state
        self.running = False

        self.logger.info("🚀 Real-time precheck system initialized")

    def _on_ingredient_detected(self, ingredient_event: IngredientEvent):
        """Callback for when ingredient is detected"""
        # Queue ingredient for processing
        asyncio.create_task(self.processor.queue_ingredient(ingredient_event))

    async def start(self):
        """Start the real-time system"""
        self.logger.info("🚀 Starting real-time precheck system")

        self.running = True

        # Start components
        self.detector.start_monitoring()
        await self.processor.start_processing()

        self.logger.info("✅ Real-time precheck system started")

    async def _process_detected_file(self, file_path):
        # Create ingredient first
        ingredient = await self.ingredient_manager.create_ingredient_from_file(file_path)

        # Process with neural AI
        if ingredient:
            await self.processor.queue_ingredient_object(ingredient)


    async def stop(self):
        """Stop the real-time system"""
        self.logger.info("🛑 Stopping real-time precheck system")

        self.running = False

        # Stop components
        self.detector.stop_monitoring()
        await self.processor.stop_processing()

        self.logger.info("✅ Real-time precheck system stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_running': self.running,
            'neural_system_available': self.universal_system is not None,
            'processor_stats': self.processor.get_real_time_stats(),
            'database_stats': self.processor.results_db.get_statistics(),
            'monitored_directories': self.detector.watch_directories,
            'processed_files_count': len(self.detector.processed_files)
        }

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def main():
    """Main function"""
    print("🧠 Intel Real-Time Precheck System with Neural AI")
    print("=" * 80)

    # Configuration
    watch_directories = [
        "/tmp/ingredients",  # Example directory
        "/opt/intel/ingredients",
        "/home/ingredients",
        # Add your actual ingredient directories here
    ]

    # Create directories if they don't exist (for demo)
    for directory in watch_directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Monitoring directory: {directory}")

    # Initialize system
    system = RealTimePrecheckSystem(watch_directories)

    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal")
        asyncio.create_task(system.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start system
        await system.start()

        print("\n✅ Real-time precheck system is running!")
        print("📋 System Status:")
        status = system.get_system_status()
        for key, value in status.items():
            print(f"   {key}: {value}")

        print("\n💡 To test the system:")
        print("   1. Copy ingredient files to monitored directories")
        print("   2. Watch the logs for real-time processing")
        print("   3. Check the database for results")
        print("\n🔍 Press Ctrl+C to stop the system")

        # Keep running
        while system.running:
            await asyncio.sleep(10)

            # Print periodic status
            stats = system.processor.get_real_time_stats()
            if stats['total_processed'] > 0:
                print(f"\n📊 Processing Stats: {stats['total_processed']} processed, "
                      f"{stats['processing_rate']:.1f}/min, "
                      f"Neural: {'✅' if stats['neural_available'] else '❌'}")

    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await system.stop()

if __name__ == "__main__":
    # Create demo ingredients for testing
    def create_demo_ingredients():
        """Create demo ingredient files for testing"""
        demo_dir = "/tmp/ingredients"
        os.makedirs(demo_dir, exist_ok=True)

        demo_files = [
            "intel_graphics_driver.dll",
            "nvidia_display_driver.sys",
            "realtek_audio.inf",
            "intel_wifi_firmware.bin",
            "test_component.exe"
        ]

        for filename in demo_files:
            filepath = os.path.join(demo_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Demo ingredient file: {filename}\n")
                f.write(f"Created at: {datetime.now()}\n")

        print(f"📦 Created {len(demo_files)} demo ingredients in {demo_dir}")

    if len(sys.argv) > 1 and sys.argv[1] == '--create-demo':
        create_demo_ingredients()
    else:
        asyncio.run(main())
