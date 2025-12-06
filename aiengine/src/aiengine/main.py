# main.py - WORLD CLASS UNIVERSAL NEURAL NETWORK SYSTEM
import sys
import os
import gc
import time
import signal
import threading
import asyncio
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque
import logging
from pathlib import Path

MODULAR_SYSTEM_AVAILABLE = True

# Add project root to path early
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PIL Compatibility Fix for Transformers
try:
    from PIL import Image
    import PIL
    # Fix PIL.Image.Resampling compatibility for older Pillow versions
    if not hasattr(Image, 'Resampling'):
        class Resampling:
            NEAREST = getattr(Image, 'NEAREST', 0)
            LANCZOS = getattr(Image, 'LANCZOS', 1)
            BILINEAR = getattr(Image, 'BILINEAR', 2)
            BICUBIC = getattr(Image, 'BICUBIC', 3)
            BOX = getattr(Image, 'BOX', 4)
            HAMMING = getattr(Image, 'HAMMING', 5)
        Image.Resampling = Resampling

        # Also patch PIL.Image if accessible
        try:
            PIL.Image.Resampling = Resampling
        except:
            pass

        print(f"✅ PIL compatibility fix applied (PIL version: {PIL.__version__})")
except ImportError as e:
    print(f"⚠️ PIL not available: {e}")
    # Create minimal mock for transformers compatibility
    class MockImage:
        class Resampling:
            NEAREST = 0
            LANCZOS = 1
            BILINEAR = 2
            BICUBIC = 3
    sys.modules['PIL'] = type('MockPIL', (), {'Image': MockImage})()
    print("⚠️ PIL not available - transformers image processing may not work")

try:
    from dotenv import load_dotenv

    # Load .env file as single source of truth
    env_loaded = load_dotenv('/aiengine/src/aiengine/.env')

    if env_loaded:
        print("✅ Environment variables loaded from .env file")
        print(f"   PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'not configured')}")
        print(f"   PostgreSQL Port: {os.getenv('POSTGRES_PORT', 'not configured')}")
        print(f"   PostgreSQL DB: {os.getenv('POSTGRES_DB', 'not configured')}")
        print(f"   PostgreSQL User: {os.getenv('POSTGRES_USER', 'not configured')}")
        print(f"   PostgreSQL Password: {'configured' if os.getenv('POSTGRES_PASSWORD') else 'not configured'}")
    else:
        print("⚠️ .env file not found at /aiengine/src/aiengine/.env")

except ImportError:
    print("⚠️ python-dotenv not installed")
    print("   Install with: pip install python-dotenv")
    print("   Using system environment variables only")


# Ensure PORTS_AVAILABLE is defined
# if 'PORTS_AVAILABLE' not in globals():
#    PORTS_AVAILABLE = False
#    PORTS = {
#        'AI_ENGINE_MAIN': 8000,
#        'AI_ENGINE_API': 8090,
#        'POSTGRESQL': 5432,
#        'REDIS': 6379,
#        'PROMETHEUS': 9090,
#        'GRAFANA': 3000,
#        'ENHANCED_DASHBOARD': 5000,
#        'ALERT_PROCESSOR': 8052
#    }
#    logger.warning("⚠️ PORTS_AVAILABLE not set, using fallback configuration")


# Validate critical PostgreSQL configuration
def validate_postgres_config():
    """Validate PostgreSQL configuration from .env"""
    required_vars = ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️ Missing PostgreSQL configuration: {missing_vars}")
        print("   PostgreSQL will not be available - falling back to SQLite")
        return False
    else:
        print("✅ PostgreSQL configuration complete")
        return True

# Validate configuration on startup
postgres_config_valid = validate_postgres_config()

try:
    from core.universal_types import DomainType, TaskType, UniversalTask, UniversalSolution
    print("✅ Universal types loaded from core.universal_types")
except ImportError as e:
    print(f"❌ Failed to import universal types: {e}")
    print("📁 Checking if core/universal_types.py exists...")
    if os.path.exists('/aiengine/src/aiengine/core/universal_types.py'):
        print("✅ File exists, checking Python path...")
        if '/aiengine/src/aiengine' not in sys.path:
            sys.path.insert(0, '/aiengine/src/aiengine')
            print("✅ Added /aiengine/src/aiengine to Python path")
            try:
                from core.universal_types import DomainType, TaskType, UniversalTask, UniversalSolution
                print("✅ Universal types loaded after path fix")
            except ImportError as e2:
                print(f"❌ Still failed after path fix: {e2}")
                sys.exit(1)
    else:
        print("❌ core/universal_types.py file not found!")
        sys.exit(1)

try:
    import shap
    import lime
    from lime.lime_text import LimeTextExplainer
    from lime.lime_tabular import LimeTabularExplainer
    EXPLAINABILITY_AVAILABLE = True
    print("✅ SHAP & LIME loaded for explainability")
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    print("⚠️ Explainability libraries not available - install with: pip install shap lime")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available")

# For attention visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prometheus integration with delayed import to avoid circular dependency
PROMETHEUS_INTEGRATION_AVAILABLE = False

def initialize_azure_integration():
    """Proper Azure integration initialization"""
    try:
        from core.azure_connect import AzureConnection
        return AzureConnection()
    except ImportError as e:
        logger.warning(f"Azure integration not available: {e}")
        return None

def initialize_prometheus_integration():
    """Initialize Prometheus integration (disabled due to circular import)"""
    global PROMETHEUS_INTEGRATION_AVAILABLE
    logger.info("🔧 Prometheus integration disabled - circular import with prometheus_alert_receiver.py")
    logger.info("   prometheus_alert_receiver.py tries to import from main.py")
    logger.info("   This creates a circular dependency that causes initialization errors")
    PROMETHEUS_INTEGRATION_AVAILABLE = False
    return None

# Import utilities from shared module to avoid circular imports
try:
    from config.ai_utils import (
        safe_json_serialize,
        validate_input_data,
        create_task_id,
        setup_directories,
        cleanup_temp_files,
        get_system_resources,
        format_duration,
        format_bytes,
        health_check,
        convert_deque_to_list,
        parse_port_config
    )
    AI_UTILITIES_AVAILABLE = True
    print("✅ AI utility functions imported successfully")
except ImportError as e:
    AI_UTILITIES_AVAILABLE = False
    print(f"⚠️ AI utils not available: {e}")
    # Minimal fallbacks
    def safe_json_serialize(obj):
        return obj
    def validate_input_data(data, max_size_mb=10):
        return True, "Valid"
    def create_task_id(prefix="task"):
        return f"{prefix}_{int(time.time())}"

    def convert_deque_to_list(obj):
        return obj

    def cleanup_memory_periodically(self):
        """Periodic memory cleanup to prevent leaks"""
        if hasattr(self, '_projection_cache'):
            if len(self._projection_cache) > 100:
                recent_keys = list(self._projection_cache.keys())[-50:]
                self._projection_cache = {k: self._projection_cache[k] for k in recent_keys}

        gc.collect()

        current_time = time.time()
        old_tasks = [
            task_id for task_id, data in self.completed_tasks.items()
            if current_time - data['task'].timestamp > 3600
        ]
        for task_id in old_tasks:
            del self.completed_tasks[task_id]

    def parse_port_config(port_config):
        return 8000
    def setup_directories():
        return []
    def cleanup_temp_files(max_age_hours=24):
        return 0
    def get_system_resources():
        return {}
    def format_duration(seconds):
        return f"{seconds:.2f}s"
    def format_bytes(bytes_value):
        return f"{bytes_value}B"
    def health_check():
        return {'status': 'unknown'}

# Modular System Imports - Enhanced domain and task management
try:
    from core.domains.domain_registry import (
        get_domain_config,
        get_domain_feature_weight,
        is_task_compatible_with_domain,
        DOMAIN_METADATA,
        get_all_domains,
        get_domain_stats
    )
    from core.tasks.task_registry import (
        get_task_config,
        validate_task_input,
        get_processing_steps,
        get_confidence_factors,
        TASK_METADATA,
        get_all_task_types,
        get_task_stats
    )

    MODULAR_SYSTEM_AVAILABLE = True
    print("✅ Modular domain/task system loaded successfully")
    print(f"   📊 Domains configured: {len(DOMAIN_METADATA)}")
    print(f"   📊 Tasks configured: {len(TASK_METADATA)}")

except ImportError as e:
    print(f"⚠️ Modular system not available, using legacy system: {e}")

class PrecheckManager:
    def __init__(self):
        self.processor = None
        self.engine = None
        self.azure_connection = None
        self._initialize()

    def _initialize(self):
        try:
            from core.precheck_engine import PrecheckEngine
            self.engine = PrecheckEngine()
            self.processor = PrecheckProcessorWrapper(self.engine)
            print("✅ Precheck manager with engine initialized")
        except ImportError as e:
            print(f"❌ Precheck engine import failed: {e}")
        except Exception as e:
            print(f"❌ Precheck initialization failed: {e}")

    def get_processor(self):
        return self.processor

    def get_engine(self):
        return self.engine

    def is_available(self):
        return self.engine is not None

# Create a fallback PrecheckEngine
class FallbackPrecheckEngine:
    def __init__(self):
        self.available = False
        print("⚠️ Using fallback PrecheckEngine due to import issues")

    def process_sync(self, task):
        return {
            'decision': 'APPROVE_WITH_CONDITIONS',
            'confidence': 0.6,
            'explanation': 'Using fallback precheck processor',
            'fallback_mode': True
        }

    def ai_analyze_failure(self, failure_data):
        return self.process_sync(failure_data)

    def get_health_status(self):
        return {
            'engine_available': False,
            'status': 'fallback_mode',
            'last_check': time.time()
        }

    def integrate_with_universal_system(self, universal_system):
        """Fallback integration method"""
        return True

class PrecheckProcessorWrapper:
    """Wrapper to make PrecheckEngine compatible with processor interface"""
    def __init__(self, engine):
        self.precheck_engine = engine

    def process_task_sync(self, task):
        if hasattr(self.precheck_engine, 'process_sync'):
            return self.precheck_engine.process_sync(task)
        else:
            return self.precheck_engine.ai_analyze_failure(task.input_data)

    def get_health_status(self):
        if hasattr(self.precheck_engine, 'get_health_status'):
            return self.precheck_engine.get_health_status()
        else:
            return {
                'engine_available': True,
                'status': 'healthy',
                'last_check': time.time()
            }

# Initialize global precheck manager
print("🔧 Initializing global precheck manager...")
try:
    precheck_manager = PrecheckManager()
    print("✅ Global precheck manager initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize precheck manager: {e}")
    class MinimalPrecheckManager:
        def __init__(self):
            self.engine = FallbackPrecheckEngine()
            self.processor = PrecheckProcessorWrapper(self.engine)
        def get_processor(self):
            return self.processor
        def get_engine(self):
            return self.engine
        def is_available(self):
            return True
    precheck_manager = MinimalPrecheckManager()
    print("✅ Fallback precheck manager created")

# Fallback: Define minimal compatibility functions
def get_domain_feature_weight(domain, feature_name):
    """Fallback domain feature weight function"""
    domain_weights = {
        'infrastructure': {'cpu': 1.0, 'memory': 0.9, 'network': 0.8},
        'finance': {'price': 1.0, 'volume': 0.8, 'risk': 0.9},
        'healthcare': {'temperature': 0.9, 'heart_rate': 0.85}
    }
    domain_dict = domain_weights.get(domain.value if hasattr(domain, 'value') else str(domain), {})
    for key, weight in domain_dict.items():
        if key.lower() in feature_name.lower():
            return weight
    return 0.5

def get_domain_config(domain):
    """Fallback domain config function"""
    return {'description': f'Legacy {domain} domain', 'confidence_threshold': 0.6}

def validate_task_input(task_type, input_data):
    """Fallback task validation function"""
    return True, "Legacy validation - no schema available"

precheck_processor = None

# Core system imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch loaded safely")
    gc.collect()
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback implementations")

# Skip TensorFlow to prevent segmentation faults
TENSORFLOW_AVAILABLE = False
print("⚠️ TensorFlow disabled to prevent memory conflicts")

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer
    import transformers.pipelines
    from transformers.pipelines import pipeline
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_PIPELINE_AVAILABLE = True
    print(f"✅ Transformers {transformers.__version__} with pipeline loaded successfully")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_PIPELINE_AVAILABLE = False
    print(f"⚠️ Transformers not available: {e}")
    def pipeline(task, model=None, **kwargs):
        return lambda x: [{"label": "POSITIVE", "score": 0.9}]

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Basic logging setup first
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration and port management
try:
    from config.port_registry import port_registry
    from config.settings import setup_logging, logger as settings_logger, FLASK_AVAILABLE, security_config

    # Use the settings logger if available
    if settings_logger:
        logger = settings_logger
    # Create PORTS dictionary for backward compatibility
    PORTS = {name: service.port for name, service in port_registry.services.items()}
    PORTS_AVAILABLE = True

    ai_service = port_registry.get_service("AI_ENGINE_MAIN")
    AI_ENGINE_PORT = ai_service.port if ai_service else 8000
    AI_ENGINE_HOST = ai_service.host if ai_service else "localhost"

    print("✅ Port registry loaded successfully")
    print(f"📊 Services configured: {len(port_registry.services)}")
    print(f"🎯 AI_ENGINE_MAIN: {AI_ENGINE_HOST}:{AI_ENGINE_PORT}")

    # Ensure AI_ENGINE_API is available
    if 'AI_ENGINE_API' not in PORTS:
        # Add AI_ENGINE_API if not present
        PORTS['AI_ENGINE_API'] = 8090
        print(f"⚠️ AI_ENGINE_API not found in registry, using fallback port 8090")
    else:
        print(f"🎯 AI_ENGINE_API: {PORTS['AI_ENGINE_API']}")

    # Log key services
    key_services = ['AI_ENGINE_MAIN', 'AI_ENGINE_API', 'PROMETHEUS', 'GRAFANA']
    for service in key_services:
        service_config = port_registry.get_service(service)
        if service_config:
            print(f"🎯 {service}: {service_config.address}")

except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    FLASK_AVAILABLE = True
    PORTS_AVAILABLE = False
    AI_ENGINE_PORT = 8000
    AI_ENGINE_HOST = "localhost"

    # Fallback PORTS definition
    PORTS = {
        'AI_ENGINE_MAIN': 8000,
        'AI_ENGINE_API': 8090,
        'POSTGRESQL': 5432,
        'REDIS': 6379,
        'PROMETHEUS': 9090,
        'GRAFANA': 3000,
        'ENHANCED_DASHBOARD': 5000,
        'ALERT_PROCESSOR': 8052
    }

    logger.warning(f"Port registry not available, using fallback: {e}")

# Setup logging
if 'setup_logging' in globals():
    setup_logging()

# Enhanced service URL functions
def get_service_url(service_name: str) -> str:
    """Get full service URL from registry"""
    try:
        if PORTS_AVAILABLE:
            return port_registry.get_url(service_name)
        else:
            port = PORTS.get(service_name, 8000)
            return f"http://localhost:{port}"
    except (ImportError, NameError):
        return f"http://localhost:8000"

def get_service_host_port(service_name: str) -> tuple:
    """Get service host and port"""
    try:
        if PORTS_AVAILABLE:
            return port_registry.get_host_port(service_name)
        else:
            port = PORTS.get(service_name, 8000)
            return 'localhost', port
    except (ImportError, NameError):
        return 'localhost', 8000

def get_service_port(service_name: str) -> int:
    """Get service port"""
    try:
        if PORTS_AVAILABLE:
            return port_registry.get_port(service_name)
        else:
            return PORTS.get(service_name, 8000)
    except (ImportError, NameError):
        return 8000

def get_service_host(service_name: str) -> str:
    """Get service host"""
    try:
        if PORTS_AVAILABLE:
            return port_registry.get_host(service_name)
        else:
            return 'localhost'
    except (ImportError, NameError):
        return 'localhost'

@dataclass
class UniversalTask:
    """Universal task representation"""
    task_id: str
    domain: DomainType
    task_type: TaskType
    input_data: Any
    expected_output: Optional[Any] = None
    metadata: Dict[str, Any] = None
    priority: int = 1
    timestamp: float = None
    user_id: str = "system"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UniversalSolution:
    """Universal solution representation with explainability"""
    task_id: str
    solution: Any
    confidence: float
    reasoning: str
    execution_time: float
    model_used: str
    domain_adapted: bool = False
    learned_patterns: List[str] = None
    # Explainability fields
    explanation: Dict[str, Any] = None
    feature_importance: Dict[str, float] = None
    attention_weights: List[float] = None
    decision_path: List[str] = None
    counterfactuals: List[Dict] = None
    uncertainty_analysis: Dict[str, float] = None

    def __post_init__(self):
        if self.learned_patterns is None:
            self.learned_patterns = []
        if self.explanation is None:
            self.explanation = {}
        if self.feature_importance is None:
            self.feature_importance = {}
        if self.attention_weights is None:
            self.attention_weights = []
        if self.decision_path is None:
            self.decision_path = []
        if self.counterfactuals is None:
            self.counterfactuals = []
        if self.uncertainty_analysis is None:
            self.uncertainty_analysis = {}

class UniversalDatabase:
    """Enhanced database integration supporting multiple backends"""

    def __init__(self, db_type="auto", connection_string=None):
        self.db_type = db_type if db_type != "auto" else os.getenv('DB_TYPE', 'auto')
        self.connection_string = connection_string or os.getenv('DB_CONNECTION_STRING', os.getenv('DB_NAME', 'universal_ai_prod'))
        self.lock = threading.Lock()
        self.conn = None
        self.db_available = False

        if self.db_type == "auto":
            self.db_type = self._detect_available_database()

        validation = self.validate_configuration()
        if not validation['valid'] and self.db_type == "postgresql":
            logger.warning("⚠️PostgreSQL configuration invalid, falling back to SQLite")
            self.db_type = "sqlite"
            try:
                self._initialize_database()
            except Exception as e:
                logger.warning(f"⚠️SQLite fallback failed: {e}, using JSON storage")
                self.db_type = "json"

        self._initialize_database()
        self.log_connection_info()

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        info = {
            'type': self.db_type,
            'available': self.db_available,
            'connection_string': self.connection_string
        }

        if self.db_type == "postgresql":
            info.update({
                'host': os.getenv('POSTGRES_HOST'),
                'port': os.getenv('POSTGRES_PORT'),
                'database': os.getenv('POSTGRES_DB'),
                'user': os.getenv('POSTGRES_USER'),
                'password_configured': bool(os.getenv('POSTGRES_PASSWORD')),
                'ssl_mode': os.getenv('DB_SSL_MODE'),
                'schema': os.getenv('DB_SCHEMA', 'universal_ai'),
                'environment': os.getenv('ENVIRONMENT', os.getenv('ENV')),
                'deployment_id': os.getenv('DEPLOYMENT_ID'),
                'db_id': os.getenv('INTEL_DBAAS_DB_ID', os.getenv('DB_ID')),
            })
            info['connection_status'] = 'connected' if self.db_available else 'disconnected'

            if info['host'] and info['port'] and info['database'] and info['user']:
                info['connection_url'] = f"postgresql://{info['user']}:***@{info['host']}:{info['port']}/{info['database']}?sslmode={info['ssl_mode']}"
            else:
                info['connection_url'] = "incomplete_configuration"

        return info

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate database configuration"""
        validation_result = {
            'valid': False,
            'database_type': self.db_type,
            'issues': [],
            'recommendations': []
        }

        if self.db_type == "postgresql":
            required_env_vars = {
                'POSTGRES_HOST': 'PostgreSQL server hostname/IP',
                'POSTGRES_PORT': 'PostgreSQL server port',
                'POSTGRES_DB': 'Database name',
                'POSTGRES_USER': 'Database username',
                'POSTGRES_PASSWORD': 'Database password'
            }

            missing_vars = []
            for var, description in required_env_vars.items():
                if not os.getenv(var):
                    missing_vars.append(f"{var} ({description})")

            if missing_vars:
                validation_result['issues'].extend(missing_vars)
                validation_result['recommendations'].append("Add missing variables to /aiengine/src/aiengine/.env file")
            else:
                validation_result['valid'] = True

            if validation_result['valid']:
                try:
                    connection_test = self.test_connection()
                    if not connection_test:
                        validation_result['valid'] = False
                        validation_result['issues'].append("Connection test failed")
                        validation_result['recommendations'].append("Verify PostgreSQL server is running and accessible")
                except Exception as e:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Connection error: {str(e)}")

        elif self.db_type == "sqlite":
            validation_result['valid'] = True
            validation_result['recommendations'].append("Consider PostgreSQL for production use")
        elif self.db_type == "json":
            validation_result['valid'] = True
            validation_result['recommendations'].append("JSON storage is not recommended for production")

        return validation_result

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if self.db_type == "postgresql":
                import psycopg2
                host = os.getenv('POSTGRES_HOST')
                port = int(os.getenv('POSTGRES_PORT', '5432'))
                database = os.getenv('POSTGRES_DB')
                user = os.getenv('POSTGRES_USER')
                password = os.getenv('POSTGRES_PASSWORD')

                test_conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    connect_timeout=10
                )
                test_conn.close()
                return True
            elif self.db_type == "sqlite":
                import sqlite3
                db_path = os.getenv('SQLITE_DB_PATH', f"{os.getenv('DB_NAME', 'universal_ai_prod')}.db")
                test_conn = sqlite3.connect(db_path, timeout=5)
                test_conn.close()
                return True
            elif self.db_type == "json":
                return True
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _detect_available_database(self) -> str:
        """Auto-detect the best available database backend"""
        try:
            import psycopg2
            pg_config = {
                'host': os.getenv('POSTGRES_HOST'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB'),
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD')
            }

            required_params = ['host', 'database', 'user', 'password']
            missing_params = [param for param in required_params if not pg_config.get(param)]

            if missing_params:
                logger.warning(f"⚠️ PostgreSQL config incomplete - missing: {missing_params}")
                raise ValueError(f"Missing PostgreSQL configuration: {missing_params}")

            test_conn = psycopg2.connect(**pg_config)
            test_conn.close()
            logger.info("🔍 PostgreSQL detected and connection verified")
            return "postgresql"
        except ImportError as e:
            logger.warning("⚠️ PostgreSQL adapter (psycopg2) not available")
        except ValueError as e:
            logger.warning(f"⚠️ PostgreSQL configuration error: {e}")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL connection failed: {e}")

        try:
            import sqlite3
            logger.info("🔍 SQLite3 detected and available")
            return "sqlite"
        except ImportError:
            logger.warning("⚠️ SQLite3 not available")

        logger.info("🔍 Falling back to JSON file storage")
        return "json"

    def _initialize_database(self):
        """Initialize the selected database backend"""
        try:
            if self.db_type == "postgresql":
                self._setup_postgresql()
            elif self.db_type == "sqlite":
                self._setup_sqlite()
            else:
                self._setup_json()
            self.db_available = True
            logger.info(f"✅ Database initialized: {self.db_type}")
        except Exception as e:
            logger.error(f"❌ Database initialization failed for {self.db_type}: {e}")
            if self.db_type != "json":
                logger.info("🔄 Falling back to JSON storage")
                self.db_type = "json"
                self._setup_json()
                self.db_available = True
            else:
                raise

    def _setup_postgresql(self):
        """Setup PostgreSQL database using configuration from .env file only"""
        import psycopg2
        from psycopg2.extras import RealDictCursor

        host = os.getenv('POSTGRES_HOST')
        port = int(os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432')))
        database = os.getenv('POSTGRES_DB')
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        ssl_mode = os.getenv('DB_SSL_MODE', os.getenv('SSL_MODE', 'prefer'))

        logger.info(f"🔍 Connecting to database: {host}:{port}/{database}")

        required_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables in .env: {missing}")

        try:
            connection_params = {
                'host': host,
                'port': port,
                'database': database,
                'user': user,
                'password': password,
                'sslmode': ssl_mode,
                'connect_timeout': 30,
                'application_name': f"universal_ai_{os.getenv('DEPLOYMENT_ID', 'system')}"
            }

            logger.info("🔄 Establishing PostgreSQL connection using .env configuration...")

            self.conn = psycopg2.connect(**connection_params)
            self.conn.autocommit = True

            cursor = self.conn.cursor()
            cursor.execute("SELECT version(), current_database(), current_user")
            result = cursor.fetchone()

            schema_name = os.getenv('DB_SCHEMA', 'universal_ai')
            cursor.execute("""
                SELECT schema_name FROM information_schema.schemata
                WHERE schema_name = %s
            """, (schema_name,))
            schema_exists = cursor.fetchone()

            if not schema_exists:
                logger.info(f"🔧 Creating schema: {schema_name}")
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

            self._create_postgresql_tables_with_schema()

            logger.info(f"🐘 PostgreSQL connected successfully!")
            logger.info(f"   Database: {result[1]} (User: {result[2]})")
            logger.info(f"   Version: {result[0].split(',')[0]}")
            logger.info(f"   Schema: {schema_name}")

        except psycopg2.OperationalError as e:
            error_msg = str(e)
            logger.error(f"❌ Database connection failed: {error_msg}")
            raise ConnectionError(f"Database connection failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            raise ConnectionError(f"Database connection error: {e}")

    def _create_postgresql_tables_with_schema(self):
        """Create PostgreSQL tables using schema and settings from .env"""
        cursor = self.conn.cursor()

        schema_name = os.getenv('DB_SCHEMA', 'universal_ai')
        environment = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development'))
        deployment_id = os.getenv('DEPLOYMENT_ID', 'universal-ai')

        cursor.execute(f"SET search_path TO {schema_name}, public")

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {schema_name}.tasks (
                task_id VARCHAR(255) PRIMARY KEY,
                domain VARCHAR(100) NOT NULL,
                task_type VARCHAR(100) NOT NULL,
                input_data TEXT,
                solution TEXT,
                confidence REAL,
                execution_time REAL,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                environment VARCHAR(50) DEFAULT %s,
                deployment_id VARCHAR(100) DEFAULT %s
            )
        ''', (environment, deployment_id))

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {schema_name}.performance_metrics (
                id SERIAL PRIMARY KEY,
                domain VARCHAR(100),
                task_type VARCHAR(100),
                avg_confidence REAL,
                avg_execution_time REAL,
                total_tasks INTEGER,
                successful_tasks INTEGER,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                environment VARCHAR(50) DEFAULT %s,
                deployment_id VARCHAR(100) DEFAULT %s
            )
        ''', (environment, deployment_id))

        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {schema_name}.system_status (
                id SERIAL PRIMARY KEY,
                system_id VARCHAR(255),
                uptime_seconds REAL,
                total_tasks_processed INTEGER,
                successful_resolutions INTEGER,
                average_confidence REAL,
                average_execution_time REAL,
                domains_mastered TEXT,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                environment VARCHAR(50) DEFAULT %s,
                deployment_id VARCHAR(100) DEFAULT %s,
                db_id VARCHAR(100) DEFAULT %s
            )
        ''', (environment, deployment_id, os.getenv('DB_ID', 'postgresql')))

        index_queries = [
            f'CREATE INDEX IF NOT EXISTS idx_tasks_domain ON {schema_name}.tasks(domain)',
            f'CREATE INDEX IF NOT EXISTS idx_tasks_timestamp ON {schema_name}.tasks(timestamp)',
            f'CREATE INDEX IF NOT EXISTS idx_tasks_confidence ON {schema_name}.tasks(confidence)',
            f'CREATE INDEX IF NOT EXISTS idx_tasks_environment ON {schema_name}.tasks(environment)',
        ]

        for query in index_queries:
            cursor.execute(query)

        logger.info(f"✅ PostgreSQL tables created in {schema_name} schema")
        logger.info(f"🔧 Environment: {environment}")
        logger.info(f"🔧 Deployment ID: {deployment_id}")

    def _setup_sqlite(self):
        """Setup SQLite database using .env configuration"""
        import sqlite3

        db_path = os.getenv('SQLITE_DB_PATH', f"{os.getenv('DB_NAME', 'universal_ai_prod')}.db")
        db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else os.getenv('DB_STORAGE_PATH', 'data')
        timeout = int(os.getenv('DB_CONNECTION_TIMEOUT', '30'))

        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"🔧 Setting up SQLite database: {db_path}")

        try:
            self.conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
                timeout=timeout
            )

            journal_mode = os.getenv('SQLITE_JOURNAL_MODE', 'WAL')
            synchronous = os.getenv('SQLITE_SYNCHRONOUS', 'NORMAL')
            cache_size = os.getenv('SQLITE_CACHE_SIZE', '2000')

            self.conn.execute(f"PRAGMA journal_mode={journal_mode}")
            self.conn.execute(f"PRAGMA synchronous={synchronous}")
            self.conn.execute(f"PRAGMA cache_size={cache_size}")

            self._create_sqlite_tables()

            logger.info(f"📁 SQLite database initialized: {db_path}")
            logger.info(f"🔧 Journal mode: {journal_mode}")
            logger.info(f"🔧 Cache size: {cache_size}")
        except Exception as e:
            logger.error(f"❌ SQLite setup failed: {e}")
            raise

    def _create_sqlite_tables(self):
        """Create SQLite tables using .env configuration"""
        cursor = self.conn.cursor()

        environment = os.getenv('ENVIRONMENT', 'development')
        deployment_id = os.getenv('DEPLOYMENT_ID', 'universal-ai')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                task_type TEXT NOT NULL,
                input_data TEXT,
                solution TEXT,
                confidence REAL,
                execution_time REAL,
                timestamp REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                environment TEXT DEFAULT ?,
                deployment_id TEXT DEFAULT ?
            )
        ''', (environment, deployment_id))

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                task_type TEXT,
                avg_confidence REAL,
                avg_execution_time REAL,
                total_tasks INTEGER,
                successful_tasks INTEGER,
                timestamp REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                environment TEXT DEFAULT ?,
                deployment_id TEXT DEFAULT ?
            )
        ''', (environment, deployment_id))

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_id TEXT,
                uptime_seconds REAL,
                total_tasks_processed INTEGER,
                successful_resolutions INTEGER,
                average_confidence REAL,
                average_execution_time REAL,
                domains_mastered TEXT,
                timestamp REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                environment TEXT DEFAULT ?,
                deployment_id TEXT DEFAULT ?,
                db_id TEXT DEFAULT ?
            )
        ''', (environment, deployment_id, os.getenv('DB_ID', 'sqlite')))

        index_queries = [
            'CREATE INDEX IF NOT EXISTS idx_tasks_domain ON tasks(domain)',
            'CREATE INDEX IF NOT EXISTS idx_tasks_timestamp ON tasks(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_tasks_confidence ON tasks(confidence)',
            'CREATE INDEX IF NOT EXISTS idx_tasks_environment ON tasks(environment)',
            'CREATE INDEX IF NOT EXISTS idx_perf_domain ON performance_metrics(domain)',
            'CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_status_system_id ON system_status(system_id)',
            'CREATE INDEX IF NOT EXISTS idx_status_timestamp ON system_status(timestamp)'
        ]

        for query in index_queries:
            cursor.execute(query)

        self.conn.commit()
        logger.info("✅ SQLite tables and indexes created")

    def _setup_json(self):
        """Setup JSON file-based storage using .env configuration"""
        storage_path = os.getenv('JSON_STORAGE_PATH', os.getenv('DB_STORAGE_PATH', 'data'))
        db_name = os.getenv('DB_NAME', 'universal_ai_prod')
        self.data_dir = os.path.join(storage_path, f"{db_name}_json")
        os.makedirs(self.data_dir, exist_ok=True)

        self.tasks_file = os.path.join(self.data_dir, "tasks.json")
        self.metrics_file = os.path.join(self.data_dir, "metrics.json")
        self.status_file = os.path.join(self.data_dir, "system_status.json")

        initial_metadata = {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'deployment_id': os.getenv('DEPLOYMENT_ID', 'universal-ai'),
            'db_id': os.getenv('DB_ID', 'json'),
            'created_at': time.time(),
            'version': os.getenv('DB_VERSION', '1.0')
        }

        files_to_init = [
            (self.tasks_file, {'metadata': initial_metadata, 'tasks': []}),
            (self.metrics_file, {'metadata': initial_metadata, 'metrics': {}}),
            (self.status_file, {'metadata': initial_metadata, 'status': []})
        ]

        for file_path, initial_data in files_to_init:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump(initial_data, f, indent=2)

        logger.info(f"📄 JSON storage initialized: {self.data_dir}")
        logger.info(f"🔧 Environment: {initial_metadata['environment']}")
        logger.info(f"🔧 Deployment ID: {initial_metadata['deployment_id']}")

    def save_task_result(self, task: 'UniversalTask', solution: 'UniversalSolution'):
        """Save task result to database"""
        try:
            if self.db_type in ["postgresql", "sqlite"]:
                self._save_task_sql(task, solution)
            elif self.db_type == "json":
                self._save_task_json(task, solution)
        except Exception as e:
            logger.error(f"Failed to save task result: {e}")

    def _save_task_sql(self, task, solution):
        """Save task to SQL database using .env configuration"""
        cursor = self.conn.cursor()

        schema_name = os.getenv('DB_SCHEMA', 'universal_ai')
        environment = os.getenv('ENVIRONMENT', os.getenv('ENV', 'development'))
        deployment_id = os.getenv('DEPLOYMENT_ID', 'universal-ai')

        if self.db_type == "postgresql":
            cursor.execute(f"SET search_path TO {schema_name}, public")

        try:
            safe_input_data = task.input_data
            safe_solution_data = solution.solution

            if self.db_type == "postgresql":
                query = f'''
                    INSERT INTO {schema_name}.tasks
                    (task_id, domain, task_type, input_data, solution, confidence, execution_time, timestamp, environment, deployment_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO UPDATE SET
                    domain = EXCLUDED.domain,
                    task_type = EXCLUDED.task_type,
                    input_data = EXCLUDED.input_data,
                    solution = EXCLUDED.solution,
                    confidence = EXCLUDED.confidence,
                    execution_time = EXCLUDED.execution_time,
                    timestamp = EXCLUDED.timestamp,
                    updated_at = CURRENT_TIMESTAMP,
                    environment = EXCLUDED.environment,
                    deployment_id = EXCLUDED.deployment_id
                '''
            else:  # SQLite
                query = '''
                    INSERT OR REPLACE INTO tasks
                    (task_id, domain, task_type, input_data, solution, confidence, execution_time, timestamp, environment, deployment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''

            params = (
                task.task_id,
                task.domain.value,
                task.task_type.value,
                json.dumps(safe_input_data),
                json.dumps(safe_solution_data),
                solution.confidence,
                solution.execution_time,
                task.timestamp,
                environment,
                deployment_id
            )

            cursor.execute(query, params)
            if self.db_type == "sqlite":
                self.conn.commit()

            logger.debug(f"💾 Task {task.task_id} saved (env: {environment})")
        except Exception as e:
            logger.error(f"Failed to save task to SQL database: {e}")
            raise

    def _save_task_json(self, task, solution):
        """Save task to JSON file using .env configuration"""
        max_tasks = int(os.getenv('JSON_MAX_TASKS', '10000'))
        environment = os.getenv('ENVIRONMENT', 'development')
        deployment_id = os.getenv('DEPLOYMENT_ID', 'universal-ai')

        with self.lock:
            try:
                with open(self.tasks_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {'metadata': {}, 'tasks': []}

            task_record = {
                'task_id': task.task_id,
                'domain': task.domain.value,
                'task_type': task.task_type.value,
                'input_data': task.input_data,
                'solution': solution.solution,
                'confidence': solution.confidence,
                'execution_time': solution.execution_time,
                'timestamp': task.timestamp,
                'environment': environment,
                'deployment_id': deployment_id,
                'created_at': time.time()
            }

            data['tasks'] = [t for t in data['tasks'] if t['task_id'] != task.task_id]
            data['tasks'].append(task_record)

            if len(data['tasks']) > max_tasks:
                data['tasks'] = data['tasks'][-max_tasks:]

            data['metadata'].update({
                'last_updated': time.time(),
                'total_tasks': len(data['tasks']),
                'environment': environment,
                'deployment_id': deployment_id
            })

            with open(self.tasks_file, 'w') as f:
                json.dump(data, f, indent=2)

    def get_task_count(self) -> int:
        """Get total number of tasks"""
        try:
            if self.db_type in ["postgresql", "sqlite"]:
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tasks")
                return cursor.fetchone()[0]
            elif self.db_type == "json":
                try:
                    with open(self.tasks_file, 'r') as f:
                        data = json.load(f)
                    return len(data.get('tasks', []))
                except (FileNotFoundError, json.JSONDecodeError):
                    return 0
        except Exception as e:
            logger.error(f"Failed to get task count: {e}")
            return 0

    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        """Get recent tasks"""
        try:
            if self.db_type in ["postgresql", "sqlite"]:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT task_id, domain, task_type, confidence, execution_time, timestamp, created_at
                    FROM tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
                columns = ['task_id', 'domain', 'task_type', 'confidence', 'execution_time', 'timestamp', 'created_at']
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            elif self.db_type == "json":
                try:
                    with open(self.tasks_file, 'r') as f:
                        data = json.load(f)
                    tasks = data.get('tasks', [])
                    return sorted(tasks, key=lambda x: x.get('created_at', 0), reverse=True)[:limit]
                except (FileNotFoundError, json.JSONDecodeError):
                    return []
        except Exception as e:
            logger.error(f"Failed to get recent tasks: {e}")
            return []

    def get_domain_statistics(self) -> Dict[str, Dict]:
        """Get statistics by domain"""
        try:
            if self.db_type in ["postgresql", "sqlite"]:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT domain,
                           COUNT(*) as total_tasks,
                           AVG(confidence) as avg_confidence,
                           AVG(execution_time) as avg_execution_time,
                           SUM(CASE WHEN confidence > 0.5 THEN 1 ELSE 0 END) as successful_tasks
                    FROM tasks
                    GROUP BY domain
                ''')
                stats = {}
                for row in cursor.fetchall():
                    domain, total, avg_conf, avg_time, successful = row
                    stats[domain] = {
                        'total_tasks': total,
                        'avg_confidence': round(float(avg_conf or 0), 3),
                        'avg_execution_time': round(float(avg_time or 0), 3),
                        'successful_tasks': successful,
                        'success_rate': round(successful / total * 100, 1) if total > 0 else 0
                    }
                return stats
            elif self.db_type == "json":
                try:
                    with open(self.tasks_file, 'r') as f:
                        data = json.load(f)
                    tasks = data.get('tasks', [])
                    stats = {}
                    for task in tasks:
                        domain = task.get('domain', 'unknown')
                        if domain not in stats:
                            stats[domain] = {
                                'total_tasks': 0,
                                'total_confidence': 0,
                                'total_execution_time': 0,
                                'successful_tasks': 0
                            }
                        stats[domain]['total_tasks'] += 1
                        confidence = task.get('confidence', 0)
                        stats[domain]['total_confidence'] += confidence
                        stats[domain]['total_execution_time'] += task.get('execution_time', 0)
                        if confidence > 0.5:
                            stats[domain]['successful_tasks'] += 1

                    for domain, data in stats.items():
                        total = data['total_tasks']
                        if total > 0:
                            data['avg_confidence'] = round(data['total_confidence'] / total, 3)
                            data['avg_execution_time'] = round(data['total_execution_time'] / total, 3)
                            data['success_rate'] = round(data['successful_tasks'] / total * 100, 1)
                        else:
                            data['avg_confidence'] = 0
                            data['avg_execution_time'] = 0
                            data['success_rate'] = 0

                        del data['total_confidence']
                        del data['total_execution_time']
                    return stats
                except (FileNotFoundError, json.JSONDecodeError):
                    return {}
        except Exception as e:
            logger.error(f"Failed to get domain statistics: {e}")
            return {}

    def get_safe_connection_string(self) -> str:
        """Get PostgreSQL connection string with masked password for logging"""
        if self.db_type == "postgresql":
            host = os.getenv('POSTGRES_HOST')
            port = os.getenv('POSTGRES_PORT', '5432')
            database = os.getenv('POSTGRES_DB')
            user = os.getenv('POSTGRES_USER')
            return f"postgresql://{user}:***@{host}:{port}/{database}"
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.connection_string}.db"
        else:
            return f"json://{self.connection_string}_json"

    def log_connection_info(self):
        """Log connection information for debugging"""
        logger.info(f"📊 Database Configuration:")
        logger.info(f"   Type: {self.db_type}")
        logger.info(f"   Connection String: {self.get_safe_connection_string()}")
        if self.db_type == "postgresql":
            logger.info(f"   Host: {os.getenv('POSTGRES_HOST')}")
            logger.info(f"   Port: {os.getenv('POSTGRES_PORT')}")
            logger.info(f"   Database: {os.getenv('POSTGRES_DB')}")
            logger.info(f"   User: {os.getenv('POSTGRES_USER')}")

    def cleanup_old_tasks(self, days_old: int = 7) -> int:
        """Clean up old tasks from database"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 3600)
            if self.db_type in ["postgresql", "sqlite"]:
                cursor = self.conn.cursor()

                if self.db_type == "postgresql":
                    cursor.execute("DELETE FROM tasks WHERE timestamp < %s", (cutoff_time,))
                    deleted_count = cursor.rowcount
                else:  # SQLite
                    cursor.execute("DELETE FROM tasks WHERE timestamp < ?", (cutoff_time,))
                    deleted_count = cursor.rowcount
                    self.conn.commit()

                cursor.close()
                return deleted_count

            elif self.db_type == "json":
                with self.lock:
                    try:
                        with open(self.tasks_file, 'r') as f:
                            data = json.load(f)
                        original_count = len(data.get('tasks', []))
                        data['tasks'] = [
                            task for task in data.get('tasks', [])
                            if task.get('timestamp', 0) >= cutoff_time
                        ]
                        with open(self.tasks_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        return original_count - len(data['tasks'])
                    except (FileNotFoundError, json.JSONDecodeError):
                        return 0
            else:
                logger.warning(f"Unknown database type for cleanup: {self.db_type}")
                return 0

        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}")
            return 0

    def export_tasks_to_csv(self, filepath: str) -> bool:
        """Export tasks to CSV file"""
        try:
            import csv

            if self.db_type in ["postgresql", "sqlite"]:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT task_id, domain, task_type, confidence, execution_time,
                           timestamp, created_at, environment, deployment_id
                    FROM tasks
                    ORDER BY created_at DESC
                ''')

                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['task_id', 'domain', 'task_type', 'confidence',
                                   'execution_time', 'timestamp', 'created_at',
                                   'environment', 'deployment_id'])
                    writer.writerows(cursor.fetchall())

            elif self.db_type == "json":
                with open(self.tasks_file, 'r') as f:
                    data = json.load(f)

                tasks = data.get('tasks', [])
                with open(filepath, 'w', newline='') as csvfile:
                    if tasks:
                        fieldnames = tasks[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(tasks)

            return True
        except Exception as e:
            logger.error(f"Failed to export tasks to CSV: {e}")
            return False

class UniversalNeuralArchitecture(nn.Module if PYTORCH_AVAILABLE else object):
    """Universal Neural Network Architecture that adapts to any domain"""
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = None, output_dim: int = 256):
        if PYTORCH_AVAILABLE:
            super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Cache for dynamic projections to prevent memory leaks
        self._projection_cache = {}

        if PYTORCH_AVAILABLE:
            self._build_pytorch_model()
        else:
            # Fallback implementation without PyTorch
            self.weights = {}
            self.biases = {}
            self._initialize_fallback_weights()

    def _build_pytorch_model(self):
        """Build PyTorch model components with proper error handling"""
        try:
            # Multi-head attention for pattern recognition
            self.attention = nn.MultiheadAttention(
                self.input_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )

            # Transformer encoder for sequence processing
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
                activation='relu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

            # Adaptive layers that can grow/shrink based on domain
            self.adaptive_layers = nn.ModuleList()
            prev_dim = self.input_dim
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.adaptive_layers.append(nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ))
                prev_dim = hidden_dim

            # Domain-specific heads (will be added dynamically)
            self.domain_heads = nn.ModuleDict()

            # Universal output layer
            self.universal_output = nn.Linear(prev_dim, self.output_dim)

            # Meta-learning components
            self.meta_learner = nn.LSTM(
                self.input_dim,
                256,
                batch_first=True,
                dropout=0.1,
                num_layers=2
            )
            self.meta_output = nn.Linear(256, 64)

            # Input projection layer for dimension mismatches
            self.input_projection = nn.Linear(1, self.input_dim)  # Will be replaced as needed

            logger.info("✅ PyTorch neural architecture built successfully")
        except Exception as e:
            logger.error(f"❌ Failed to build PyTorch model: {e}")
            # Fall back to simple model
            self._build_simple_fallback_model()

    def _build_simple_fallback_model(self):
        """Build a simple fallback model if complex model fails"""
        try:
            self.adaptive_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.output_dim)
                )
            ])
            self.domain_heads = nn.ModuleDict()
            self.universal_output = nn.Identity()
            logger.warning("⚠️ Using simple fallback PyTorch model")
        except Exception as e:
            logger.error(f"❌ Even simple model failed: {e}")

    def _initialize_fallback_weights(self):
        """Initialize weights for fallback implementation"""
        import random
        random.seed(42)  # For reproducibility

        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Xavier initialization
            limit = (6.0 / (prev_dim + hidden_dim)) ** 0.5
            self.weights[f'layer_{i}'] = [
                [random.uniform(-limit, limit) for _ in range(prev_dim)]
                for _ in range(hidden_dim)
            ]
            self.biases[f'layer_{i}'] = [0.0 for _ in range(hidden_dim)]
            prev_dim = hidden_dim

        # Output layer
        limit = (6.0 / (prev_dim + self.output_dim)) ** 0.5
        self.weights['output'] = [
            [random.uniform(-limit, limit) for _ in range(prev_dim)]
            for _ in range(self.output_dim)
        ]
        self.biases['output'] = [0.0 for _ in range(self.output_dim)]

    def add_domain_head(self, domain: DomainType, output_size: int):
        """Dynamically add a domain-specific head"""
        if PYTORCH_AVAILABLE and hasattr(self, 'domain_heads'):
            try:
                self.domain_heads[domain.value] = nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_size)
                )
                logger.info(f"✅ Added domain head for {domain.value} with output size {output_size}")
            except Exception as e:
                logger.error(f"❌ Failed to add domain head for {domain.value}: {e}")

    def forward(self, x, domain: Optional[DomainType] = None, use_attention: bool = True):
        if not PYTORCH_AVAILABLE:
            return self._forward_fallback(x)

        try:
            # Prepare input tensor safely
            x = self._prepare_input_tensor(x)

            # Apply input projection if dimensions don't match
            if x.size(-1) != self.input_dim:
                x = self._apply_input_projection(x)

            # Ensure proper dimensions for transformer
            original_shape = x.shape
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension

            # Apply attention mechanism
            if use_attention and hasattr(self, 'attention'):
                try:
                    attended_x, attention_weights = self.attention(x, x, x)
                    x = attended_x + x  # Residual connection
                except Exception as e:
                    logger.warning(f"Attention mechanism failed: {e}")

            # Apply transformer for sequence understanding
            if hasattr(self, 'transformer'):
                try:
                    x = self.transformer(x)
                    x = x.mean(dim=1)  # Global average pooling
                except Exception as e:
                    logger.warning(f"Transformer failed: {e}")
                    # Fallback: just squeeze the sequence dimension
                    x = x.squeeze(1) if x.dim() == 3 else x

            # Apply adaptive layers
            if hasattr(self, 'adaptive_layers'):
                for i, layer in enumerate(self.adaptive_layers):
                    try:
                        x = layer(x)
                    except Exception as e:
                        logger.warning(f"Adaptive layer {i} failed: {e}")
                        break

            # Apply domain-specific head if available
            if domain and hasattr(self, 'domain_heads') and domain.value in self.domain_heads:
                try:
                    domain_output = self.domain_heads[domain.value](x)
                    universal_output = self.universal_output(x)
                    return torch.cat([universal_output, domain_output], dim=-1)
                except Exception as e:
                    logger.warning(f"Domain head failed for {domain.value}: {e}")

            # Return universal output
            if hasattr(self, 'universal_output'):
                return self.universal_output(x)
            else:
                return x

        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Return default output tensor
            return torch.zeros(self.output_dim, dtype=torch.float32)

    def _prepare_input_tensor(self, x):
        """Safely prepare input tensor with comprehensive type handling"""
        try:
            # If already a tensor, ensure it's float
            if isinstance(x, torch.Tensor):
                return x.float()

            # Handle different input types
            if isinstance(x, (list, tuple)):
                flat_data = self._flatten_input(x)
                tensor = torch.tensor(flat_data, dtype=torch.float32)
                return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
            elif isinstance(x, dict):
                flat_data = self._dict_to_numeric(x)
                tensor = torch.tensor(flat_data, dtype=torch.float32)
                return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
            elif isinstance(x, str):
                numeric_repr = self._string_to_numeric(x)
                return torch.tensor(numeric_repr, dtype=torch.float32).unsqueeze(0)
            elif isinstance(x, (int, float)):
                # Single number - create a vector
                return torch.tensor([float(x)] * min(512, self.input_dim), dtype=torch.float32).unsqueeze(0)
            elif hasattr(x, '__array__'):  # numpy array
                return torch.from_numpy(x).float()
            else:
                # Try to convert to string and then to numeric
                str_repr = str(x)
                numeric_repr = self._string_to_numeric(str_repr)
                return torch.tensor(numeric_repr, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            logger.warning(f"Input tensor preparation failed: {e}")
            # Return default tensor
            return torch.zeros(1, min(512, self.input_dim), dtype=torch.float32)

    def _apply_input_projection(self, x):
        """Apply input projection to match expected dimensions"""
        try:
            input_size = x.size(-1)

            # Use cached projection or create new one
            if input_size not in self._projection_cache:
                self._projection_cache[input_size] = nn.Linear(input_size, self.input_dim)
                logger.debug(f"Created projection layer: {input_size} -> {self.input_dim}")

            projection = self._projection_cache[input_size]
            return projection(x)
        except Exception as e:
            logger.warning(f"Input projection failed: {e}")
            # Fallback: pad or truncate
            if x.size(-1) < self.input_dim:
                # Pad with zeros
                batch_size = x.size(0)
                seq_len = x.size(1) if x.dim() > 2 else 1
                padding_size = self.input_dim - x.size(-1)

                if x.dim() == 2:
                    padding = torch.zeros(batch_size, padding_size, dtype=x.dtype, device=x.device)
                    return torch.cat([x, padding], dim=-1)
                elif x.dim() == 3:
                    padding = torch.zeros(batch_size, seq_len, padding_size, dtype=x.dtype, device=x.device)
                    return torch.cat([x, padding], dim=-1)
                else:
                    return x
            else:
                # Truncate
                return x[..., :self.input_dim]

    def _flatten_input(self, data, max_length=512):
        """Flatten nested input data to numeric array"""
        result = []
        def flatten_recursive(item, depth=0):
            if len(result) >= max_length or depth > 10:  # Prevent infinite recursion
                return
            if isinstance(item, (int, float)):
                if not (np.isnan(item) or np.isinf(item)):  # Skip NaN and inf
                    result.append(float(item))
                else:
                    result.append(0.0)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    flatten_recursive(sub_item, depth + 1)
            elif isinstance(item, dict):
                for value in item.values():
                    flatten_recursive(value, depth + 1)
            elif isinstance(item, str):
                # Convert string to hash-based numeric (more stable)
                hash_val = abs(hash(item)) % 10000 / 10000.0
                result.append(hash_val)
            elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                # Handle other iterable types
                try:
                    for sub_item in item:
                        flatten_recursive(sub_item, depth + 1)
                except:
                    # If iteration fails, convert to string
                    hash_val = abs(hash(str(item))) % 10000 / 10000.0
                    result.append(hash_val)
            else:
                # Convert other types to hash-based numeric
                hash_val = abs(hash(str(item))) % 10000 / 10000.0
                result.append(hash_val)

        flatten_recursive(data)

        # Ensure we have some data
        if not result:
            result = [0.5]  # Default value

        # Pad or truncate to desired length
        target_length = min(max_length, self.input_dim)
        if len(result) < target_length:
            result.extend([0.0] * (target_length - len(result)))
        else:
            result = result[:target_length]

        return result

    def _dict_to_numeric(self, data_dict, max_length=512):
        """Convert dictionary to numeric array with better handling"""
        result = []
        # Sort keys for consistency
        sorted_items = sorted(data_dict.items(), key=lambda x: str(x[0]))

        for key, value in sorted_items:
            if len(result) >= max_length:
                break

            # Add key representation
            key_hash = abs(hash(str(key))) % 1000 / 1000.0
            result.append(key_hash)

            # Add value representation
            if isinstance(value, (int, float)):
                if not (np.isnan(value) or np.isinf(value)):
                    result.append(float(value))
                else:
                    result.append(0.0)
            elif isinstance(value, (list, tuple)):
                # Take mean of numeric values or hash of non-numeric
                numeric_vals = [v for v in value if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))]
                if numeric_vals:
                    result.append(sum(numeric_vals) / len(numeric_vals))
                else:
                    result.append(abs(hash(str(value))) % 1000 / 1000.0)
            elif isinstance(value, str):
                result.append(abs(hash(value)) % 1000 / 1000.0)
            else:
                result.append(abs(hash(str(value))) % 1000 / 1000.0)

        # Ensure minimum length and pad/truncate
        target_length = min(max_length, self.input_dim)
        if len(result) < target_length:
            result.extend([0.0] * (target_length - len(result)))
        else:
            result = result[:target_length]

        return result

    def _string_to_numeric(self, text, max_length=512):
        """Convert string to numeric representation"""
        if not text:
            return [0.0] * min(max_length, self.input_dim)

        # Method 1: Character-based encoding
        char_values = []
        for char in text[:max_length]:
            # Normalize character values to [0, 1]
            char_val = ord(char) / 1114111.0  # Max Unicode code point
            char_values.append(char_val)

        # Method 2: Add word-level features if space allows
        words = text.lower().split()
        if len(char_values) < max_length and words:
            # Add word count feature
            word_count_feature = min(len(words) / 100.0, 1.0)
            char_values.append(word_count_feature)

            # Add average word length feature
            avg_word_len = sum(len(word) for word in words) / len(words)
            avg_word_len_feature = min(avg_word_len / 20.0, 1.0)
            char_values.append(avg_word_len_feature)

        # Pad or truncate to target length
        target_length = min(max_length, self.input_dim)
        if len(char_values) < target_length:
            char_values.extend([0.0] * (target_length - len(char_values)))
        else:
            char_values = char_values[:target_length]

        return char_values

    def _forward_fallback(self, x):
        """Improved fallback forward pass without PyTorch"""
        def relu(val):
            return max(0, val)

        def matrix_multiply(matrix, vector):
            if not matrix or not vector:
                return []
            try:
                return [
                    sum(matrix[i][j] * vector[j] for j in range(min(len(vector), len(matrix[i]))))
                    for i in range(len(matrix))
                ]
            except (IndexError, TypeError):
                return [0.0] * len(matrix)

        try:
            # Convert input to list if needed
            if isinstance(x, (list, tuple)):
                current = list(x)
            elif hasattr(x, 'tolist'):
                current = x.tolist()
            elif isinstance(x, (int, float)):
                current = [float(x)]
            elif isinstance(x, str):
                current = self._string_to_numeric(x, self.input_dim)
            elif isinstance(x, dict):
                current = self._dict_to_numeric(x, self.input_dim)
            else:
                current = [0.5] * self.input_dim

            # Ensure input dimension matches
            if len(current) != self.input_dim:
                if len(current) < self.input_dim:
                    current.extend([0.0] * (self.input_dim - len(current)))
                else:
                    current = current[:self.input_dim]

            # Forward pass through layers
            for i in range(len(self.hidden_dims)):
                layer_key = f'layer_{i}'
                if layer_key in self.weights and layer_key in self.biases:
                    current = matrix_multiply(self.weights[layer_key], current)
                    current = [
                        current[j] + self.biases[layer_key][j]
                        for j in range(min(len(current), len(self.biases[layer_key])))
                    ]
                    current = [relu(val) for val in current]

            # Output layer
            if 'output' in self.weights and 'output' in self.biases:
                output = matrix_multiply(self.weights['output'], current)
                output = [
                    output[j] + self.biases['output'][j]
                    for j in range(min(len(output), len(self.biases['output'])))
                ]
                return output
            else:
                return current[:self.output_dim] if len(current) >= self.output_dim else current

        except Exception as e:
            logger.error(f"Fallback forward pass failed: {e}")
            return [0.0] * self.output_dim

    def get_model_info(self):
        """Get information about the model architecture"""
        info = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'pytorch_available': PYTORCH_AVAILABLE,
            'domain_heads': list(self.domain_heads.keys()) if hasattr(self, 'domain_heads') else [],
            'projection_cache_size': len(self._projection_cache)
        }

        if PYTORCH_AVAILABLE and hasattr(self, 'adaptive_layers'):
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            info['total_parameters'] = total_params

        return info

    def reset_projection_cache(self):
        """Reset the projection cache to free memory"""
        self._projection_cache.clear()
        logger.info("🧹 Projection cache cleared")

class DomainAdapter:
    """Adapts the universal neural network to specific domains"""
    def __init__(self):
        self.domain_patterns = {}
        self.domain_preprocessors = {}
        self.domain_postprocessors = {}
        self.adaptation_history = {}

    def analyze_domain(self, task: UniversalTask) -> Dict[str, Any]:
        """Analyze the domain characteristics of a task"""
        domain_info = {
            'domain': task.domain,
            'task_type': task.task_type,
            'data_characteristics': self._analyze_data_characteristics(task.input_data),
            'complexity_score': self._calculate_complexity(task.input_data),
            'recommended_architecture': self._recommend_architecture(task),
            'preprocessing_steps': self._recommend_preprocessing(task),
            'postprocessing_steps': self._recommend_postprocessing(task)
        }
        return domain_info

    def _analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze characteristics of input data"""
        characteristics = {
            'data_type': type(data).__name__,
            'is_structured': False,
            'is_sequential': False,
            'is_textual': False,
            'is_numerical': False,
            'is_categorical': False,
            'dimensionality': 'unknown',
            'size_estimate': 0
        }

        try:
            if isinstance(data, str):
                characteristics.update({
                    'is_textual': True,
                    'size_estimate': len(data),
                    'dimensionality': '1D'
                })
            elif isinstance(data, (list, tuple)):
                characteristics.update({
                    'is_sequential': True,
                    'size_estimate': len(data),
                    'dimensionality': '1D'
                })
                if data and isinstance(data[0], (int, float)):
                    characteristics['is_numerical'] = True
                elif data and isinstance(data[0], str):
                    characteristics['is_textual'] = True
            elif isinstance(data, dict):
                characteristics.update({
                    'is_structured': True,
                    'size_estimate': len(data),
                    'dimensionality': 'structured'
                })
            elif hasattr(data, 'shape'):  # numpy array or similar
                characteristics.update({
                    'is_numerical': True,
                    'size_estimate': data.size if hasattr(data, 'size') else len(str(data)),
                    'dimensionality': f"{len(data.shape)}D" if hasattr(data, 'shape') else 'unknown'
                })
        except Exception as e:
            logger.warning(f"Error analyzing data characteristics: {e}")

        return characteristics

    def _calculate_complexity(self, data: Any) -> float:
        """Calculate complexity score of the data"""
        try:
            if isinstance(data, str):
                return min(len(data) / 1000.0, 10.0)
            elif isinstance(data, (list, tuple)):
                return min(len(data) / 100.0, 10.0)
            elif isinstance(data, dict):
                return min(len(str(data)) / 500.0, 10.0)
            elif hasattr(data, 'size'):
                return min(data.size / 1000.0, 10.0)
            else:
                return 1.0
        except:
            return 1.0

    def _recommend_architecture(self, task: UniversalTask) -> Dict[str, Any]:
        """Recommend neural architecture based on task"""
        recommendations = {
            'suggested_layers': [512, 256, 128],
            'activation_function': 'relu',
            'use_attention': False,
            'use_transformer': False,
            'use_cnn': False,
            'use_rnn': False
        }

        # Domain-specific recommendations
        if task.domain in [DomainType.NATURAL_LANGUAGE, DomainType.CONTENT_GENERATION]:
            recommendations.update({
                'use_transformer': True,
                'use_attention': True,
                'suggested_layers': [768, 512, 256]
            })
        elif task.domain in [DomainType.COMPUTER_VISION, DomainType.CONTENT_GENERATION]:
            recommendations.update({
                'use_cnn': True,
                'suggested_layers': [1024, 512, 256, 128]
            })
        elif task.task_type in [TaskType.TIME_SERIES_FORECASTING] or task.domain in [DomainType.AUDIO_PROCESSING]:
            recommendations.update({
                'use_rnn': True,
                'use_attention': True,
                'suggested_layers': [512, 256, 128]
            })

        return recommendations

    def _recommend_preprocessing(self, task: UniversalTask) -> List[str]:
        """Recommend preprocessing steps"""
        steps = ['normalize']
        data_chars = self._analyze_data_characteristics(task.input_data)

        if data_chars['is_textual']:
            steps.extend(['tokenize', 'encode', 'pad_sequences'])
        elif data_chars['is_numerical']:
            steps.extend(['scale', 'handle_missing'])
        elif data_chars['is_categorical']:
            steps.extend(['encode_categorical', 'one_hot'])

        return steps

    def _recommend_postprocessing(self, task: UniversalTask) -> List[str]:
        """Recommend postprocessing steps"""
        steps = []

        if task.task_type == TaskType.CLASSIFICATION:
            steps.extend(['softmax', 'argmax'])
        elif task.task_type == TaskType.REGRESSION:
            steps.extend(['denormalize'])
        elif task.task_type in [TaskType.TEXT_GENERATION, TaskType.CODE_GENERATION]:
            steps.extend(['decode', 'format_output'])

        # Add format_output for precheck validation tasks
        if task.domain == DomainType.PRECHECK_VALIDATION:
            steps.append('format_output')
        elif task.domain in [DomainType.INFRASTRUCTURE, DomainType.FINANCE, DomainType.HEALTHCARE]:
            steps.append('format_output')

        return steps

    def adapt_for_domain(self, model: UniversalNeuralArchitecture, task: UniversalTask) -> UniversalNeuralArchitecture:
        """Adapt the model for a specific domain"""
        domain_info = self.analyze_domain(task)

        # Add domain-specific head if needed
        if task.domain not in [head.replace('_head', '') for head in model.domain_heads.keys() if PYTORCH_AVAILABLE]:
            output_size = self._determine_output_size(task)
            model.add_domain_head(task.domain, output_size)

        # Store adaptation history
        self.adaptation_history[task.task_id] = {
            'domain': task.domain,
            'adaptations_made': domain_info,
            'timestamp': time.time()
        }

        return model

    def _determine_output_size(self, task: UniversalTask) -> int:
        """Determine appropriate output size for domain"""
        if task.task_type == TaskType.CLASSIFICATION:
            return 10  # Default number of classes
        elif task.task_type == TaskType.REGRESSION:
            return 1
        elif task.task_type in [TaskType.TEXT_GENERATION, TaskType.CODE_GENERATION]:
            return 50000  # Vocabulary size
        else:
            return 256  # Default embedding size

class UniversalLearningEngine:
    """Universal learning engine that can learn from any domain"""
    def __init__(self):
        self.learning_strategies = {}
        self.meta_knowledge = {}
        self.transfer_learning_cache = {}
        self.continual_learning_buffer = []
        self.learning_history = []

    def learn_from_task(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """Learn from a completed task"""
        learning_result = {
            'patterns_discovered': [],
            'knowledge_updated': False,
            'transfer_opportunities': [],
            'confidence_improvement': 0.0
        }

        try:
            # Extract patterns from the task-solution pair
            patterns = self._extract_patterns(task, solution)
            learning_result['patterns_discovered'] = patterns

            # Update meta-knowledge
            self._update_meta_knowledge(task, solution, patterns)
            learning_result['knowledge_updated'] = True

            # Identify transfer learning opportunities
            transfer_ops = self._identify_transfer_opportunities(task)
            learning_result['transfer_opportunities'] = transfer_ops

            # Calculate confidence improvement
            confidence_improvement = self._calculate_confidence_improvement(task, solution)
            learning_result['confidence_improvement'] = confidence_improvement

            # Add to continual learning buffer
            self.continual_learning_buffer.append({
                'task': task,
                'solution': solution,
                'timestamp': time.time()
            })

            # Maintain buffer size
            if len(self.continual_learning_buffer) > 10000:
                self.continual_learning_buffer = self.continual_learning_buffer[-5000:]

            # Record learning history
            self.learning_history.append({
                'task_id': task.task_id,
                'domain': task.domain.value,
                'task_type': task.task_type.value,
                'learning_result': learning_result,
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"Learning from task failed: {e}")

        return learning_result

    def _extract_patterns(self, task: UniversalTask, solution: UniversalSolution) -> List[str]:
        """Extract patterns from task-solution pairs"""
        patterns = []

        try:
            # Domain-specific pattern extraction
            if task.domain == DomainType.INFRASTRUCTURE:
                patterns.extend(self._extract_infrastructure_patterns(task, solution))
            elif task.domain == DomainType.FINANCE:
                patterns.extend(self._extract_finance_patterns(task, solution))
            elif task.domain == DomainType.HEALTHCARE:
                patterns.extend(self._extract_healthcare_patterns(task, solution))

            # General patterns
            if solution.confidence > 0.9:
                patterns.append(f"high_confidence_{task.task_type.value}")
            if solution.execution_time < 1.0:
                patterns.append(f"fast_execution_{task.domain.value}")

            # Pattern based on data characteristics
            data_size = len(str(task.input_data))
            if data_size > 10000:
                patterns.append("large_data_processing")
            elif data_size < 100:
                patterns.append("small_data_processing")

        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")

        return patterns

    def _extract_infrastructure_patterns(self, task: UniversalTask, solution: UniversalSolution) -> List[str]:
        """Extract infrastructure-specific patterns"""
        patterns = []
        if 'cpu' in str(task.input_data).lower():
            patterns.append('cpu_optimization')
        if 'memory' in str(task.input_data).lower():
            patterns.append('memory_management')
        if 'network' in str(task.input_data).lower():
            patterns.append('network_optimization')
        return patterns

    def _extract_finance_patterns(self, task: UniversalTask, solution: UniversalSolution) -> List[str]:
        """Extract finance-specific patterns"""
        patterns = []
        if 'risk' in str(task.input_data).lower():
            patterns.append('risk_assessment')
        if 'price' in str(task.input_data).lower():
            patterns.append('price_prediction')
        if 'portfolio' in str(task.input_data).lower():
            patterns.append('portfolio_optimization')
        return patterns

    def _extract_healthcare_patterns(self, task: UniversalTask, solution: UniversalSolution) -> List[str]:
        """Extract healthcare-specific patterns"""
        patterns = []
        if 'diagnosis' in str(task.input_data).lower():
            patterns.append('medical_diagnosis')
        if 'treatment' in str(task.input_data).lower():
            patterns.append('treatment_recommendation')
        if 'patient' in str(task.input_data).lower():
            patterns.append('patient_monitoring')
        return patterns

    def _update_meta_knowledge(self, task: UniversalTask, solution: UniversalSolution, patterns: List[str]):
        """Update meta-knowledge base"""
        domain_key = task.domain.value
        if domain_key not in self.meta_knowledge:
            self.meta_knowledge[domain_key] = {
                'successful_patterns': {},
                'common_solutions': {},
                'performance_metrics': [],
                'learning_count': 0
            }

        # Update patterns
        for pattern in patterns:
            if pattern not in self.meta_knowledge[domain_key]['successful_patterns']:
                self.meta_knowledge[domain_key]['successful_patterns'][pattern] = 0
            self.meta_knowledge[domain_key]['successful_patterns'][pattern] += 1

        # Update performance metrics
        self.meta_knowledge[domain_key]['performance_metrics'].append({
            'confidence': solution.confidence,
            'execution_time': solution.execution_time,
            'timestamp': time.time()
        })

        # Keep only recent metrics
        if len(self.meta_knowledge[domain_key]['performance_metrics']) > 1000:
            self.meta_knowledge[domain_key]['performance_metrics'] = \
                self.meta_knowledge[domain_key]['performance_metrics'][-500:]

        self.meta_knowledge[domain_key]['learning_count'] += 1

    def _identify_transfer_opportunities(self, task: UniversalTask) -> List[str]:
        """Identify opportunities for transfer learning"""
        opportunities = []
        current_domain = task.domain.value

        # Look for similar domains with successful patterns
        for domain, knowledge in self.meta_knowledge.items():
            if domain != current_domain and knowledge['learning_count'] > 10:
                # Calculate similarity (simplified)
                similarity_score = self._calculate_domain_similarity(current_domain, domain)
                if similarity_score > 0.5:
                    opportunities.append(f"transfer_from_{domain}")

        return opportunities

    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between domains"""
        similar_domains = {
            'infrastructure': ['devops', 'cloud_computing', 'network_management'],
            'finance': ['trading', 'risk_management'],
            'healthcare': ['medical_diagnosis', 'drug_discovery'],
            'natural_language': ['content_generation', 'text_generation'],
            'computer_vision': ['image_generation', 'video_analysis']
        }

        for base_domain, related in similar_domains.items():
            if domain1 == base_domain and domain2 in related:
                return 0.8
            elif domain2 == base_domain and domain1 in related:
                return 0.8
            elif domain1 in related and domain2 in related:
                return 0.6

        return 0.1  # Default low similarity

    def _calculate_confidence_improvement(self, task: UniversalTask, solution: UniversalSolution) -> float:
        """Calculate how much confidence improved from learning"""
        domain_key = task.domain.value
        if domain_key not in self.meta_knowledge:
            return 0.0

        recent_metrics = self.meta_knowledge[domain_key]['performance_metrics'][-10:]
        if len(recent_metrics) < 2:
            return 0.0

        recent_confidence = sum(m['confidence'] for m in recent_metrics) / len(recent_metrics)
        current_confidence = solution.confidence

        return max(0.0, current_confidence - recent_confidence)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        insights = {
            'total_tasks_learned': len(self.learning_history),
            'domains_encountered': len(self.meta_knowledge),
            'top_patterns': {},
            'learning_trends': {},
            'transfer_opportunities': 0
        }

        # Calculate top patterns across all domains
        all_patterns = {}
        for domain_knowledge in self.meta_knowledge.values():
            for pattern, count in domain_knowledge['successful_patterns'].items():
                if pattern not in all_patterns:
                    all_patterns[pattern] = 0
                all_patterns[pattern] += count

        insights['top_patterns'] = dict(sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:10])

        # Calculate learning trends
        for domain, knowledge in self.meta_knowledge.items():
            if knowledge['performance_metrics']:
                recent_metrics = knowledge['performance_metrics'][-20:]
                avg_confidence = sum(m['confidence'] for m in recent_metrics) / len(recent_metrics)
                avg_execution_time = sum(m['execution_time'] for m in recent_metrics) / len(recent_metrics)

                insights['learning_trends'][domain] = {
                    'average_confidence': avg_confidence,
                    'average_execution_time': avg_execution_time,
                    'learning_count': knowledge['learning_count']
                }

        return insights

class ExplainableAIEngine:
    """Comprehensive Explainable AI engine for the Universal Neural System"""
    def __init__(self):
        self.explainers = {}
        self.explanation_cache = {}
        self.visualization_cache = {}
        # Initialize different types of explainers
        self.text_explainer = None
        self.tabular_explainer = None
        self.shap_explainer = None

        if EXPLAINABILITY_AVAILABLE:
            self._initialize_explainers()

    def _initialize_explainers(self):
        """Initialize various explainers"""
        try:
            # Text explainer for NLP tasks
            self.text_explainer = LimeTextExplainer(
                class_names=['negative', 'neutral', 'positive'],
                feature_selection='auto',
                split_expression=r'\W+',
                bow=True,
            )
            logger.info("✅ Text explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize text explainer: {e}")

    def generate_explanation(self, task: UniversalTask, solution: UniversalSolution,
                           model: UniversalNeuralArchitecture, model_output: Any = None) -> Dict[str, Any]:
        """Generate comprehensive explanation for a solution"""
        explanation = {
            'method': 'comprehensive_xai',
            'timestamp': time.time(),
            'task_context': {
                'domain': task.domain.value,
                'task_type': task.task_type.value,
                'data_type': type(task.input_data).__name__
            },
            'decision_factors': [],
            'confidence_breakdown': {},
            'feature_importance': {},
            'attention_analysis': {},
            'counterfactuals': [],
            'uncertainty_metrics': {},
            'human_readable_summary': "",
            'technical_details': {},
            'visualizations': {}
        }

        try:
            # 1. Generate decision factors
            explanation['decision_factors'] = self._extract_decision_factors(task, solution, model_output)

            # 2. Analyze confidence breakdown
            explanation['confidence_breakdown'] = self._analyze_confidence_breakdown(task, solution, model_output)

            # 3. Feature importance analysis
            explanation['feature_importance'] = self._calculate_feature_importance(task, solution, model)

            # 4. Attention mechanism analysis (if available)
            if PYTORCH_AVAILABLE and hasattr(model, 'attention'):
                explanation['attention_analysis'] = self._analyze_attention_weights(task, model)

            # 5. Generate counterfactuals
            explanation['counterfactuals'] = self._generate_counterfactuals(task, solution, model)

            # 6. Uncertainty analysis
            explanation['uncertainty_metrics'] = self._analyze_uncertainty(task, solution, model)

            # 7. Domain-specific explanations
            explanation.update(self._generate_domain_specific_explanation(task, solution))

            # 8. Human-readable summary
            explanation['human_readable_summary'] = self._generate_human_readable_summary(task, solution, explanation)

            # 9. Technical details for experts
            explanation['technical_details'] = self._generate_technical_details(task, solution, model, model_output)

            # 10. Generate visualizations
            if VISUALIZATION_AVAILABLE:
                explanation['visualizations'] = self._generate_visualizations(task, solution, explanation)

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            explanation['error'] = str(e)
            explanation['human_readable_summary'] = f"Unable to generate detailed explanation due to: {e}"

        return explanation

    def _extract_decision_factors(self, task: UniversalTask, solution: UniversalSolution, model_output: Any) -> List[Dict]:
        """Extract key factors that influenced the decision"""
        factors = []

        try:
            # Input data characteristics
            if isinstance(task.input_data, dict):
                for key, value in task.input_data.items():
                    if isinstance(value, (int, float)):
                        impact_score = abs(value) / (abs(value) + 1)  # Normalize impact
                        factors.append({
                            'factor': f"input_{key}",
                            'value': value,
                            'impact_score': impact_score,
                            'description': f"Input feature '{key}' with value {value}"
                        })

            # Domain expertise influence
            factors.append({
                'factor': 'domain_expertise',
                'value': task.domain.value,
                'impact_score': 0.8,  # High impact from domain adaptation
                'description': f"Applied specialized knowledge for {task.domain.value} domain"
            })

            # Task type influence
            factors.append({
                'factor': 'task_methodology',
                'value': task.task_type.value,
                'impact_score': 0.7,
                'description': f"Used {task.task_type.value} methodology for problem solving"
            })

            # Model confidence influence
            factors.append({
                'factor': 'model_confidence',
                'value': solution.confidence,
                'impact_score': solution.confidence,
                'description': f"Model confidence level: {solution.confidence:.3f}"
            })

            # Sort by impact score
            factors.sort(key=lambda x: x['impact_score'], reverse=True)

        except Exception as e:
            logger.warning(f"Decision factor extraction failed: {e}")

        return factors[:10]  # Return top 10 factors

    def _analyze_confidence_breakdown(self, task: UniversalTask, solution: UniversalSolution, model_output: Any) -> Dict[str, float]:
        """Break down confidence score into components"""
        breakdown = {
            'base_model_confidence': 0.0,
            'domain_adaptation_bonus': 0.0,
            'data_quality_factor': 0.0,
            'historical_performance_factor': 0.0,
            'uncertainty_penalty': 0.0
        }

        try:
            # Base model confidence from output
            if isinstance(model_output, dict) and 'probabilities' in model_output:
                probs = model_output['probabilities']
                if probs and isinstance(probs, list):
                    breakdown['base_model_confidence'] = max(probs) if probs else 0.5
            else:
                breakdown['base_model_confidence'] = 0.6  # Default

            # Domain adaptation bonus
            if solution.domain_adapted:
                breakdown['domain_adaptation_bonus'] = 0.15

            # Data quality assessment
            data_quality = self._assess_data_quality(task.input_data)
            breakdown['data_quality_factor'] = data_quality * 0.1

            # Historical performance (simplified)
            breakdown['historical_performance_factor'] = 0.05

            # Uncertainty penalty
            uncertainty = self._calculate_prediction_uncertainty(model_output)
            breakdown['uncertainty_penalty'] = -uncertainty * 0.1

        except Exception as e:
            logger.warning(f"Confidence breakdown analysis failed: {e}")

        return breakdown

    def _calculate_feature_importance(self, task: UniversalTask, solution: UniversalSolution, model: UniversalNeuralArchitecture) -> Dict[str, float]:
        """Calculate feature importance using various methods"""
        importance = {}

        try:
            if isinstance(task.input_data, dict):
                # For structured data, calculate importance based on variance and correlation
                for key, value in task.input_data.items():
                    if isinstance(value, (int, float)):
                        # Simple importance based on magnitude and domain knowledge
                        base_importance = abs(value) / (abs(value) + 1)
                        # Domain-specific importance weighting
                        domain_weight = self._get_domain_feature_weight(task.domain, key)
                        importance[key] = base_importance * domain_weight
                    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                        # For time series or array data
                        variance = np.var(value) if len(value) > 1 else 0
                        mean_val = np.mean(value)
                        importance[key] = min(variance + abs(mean_val) * 0.1, 1.0)
            elif isinstance(task.input_data, str):
                # For text data, use simple keyword importance
                words = task.input_data.lower().split()
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

                # Calculate TF-like importance
                total_words = len(words)
                for word, count in word_counts.items():
                    if len(word) > 2:  # Skip short words
                        importance[f"word_{word}"] = count / total_words

            # Normalize importance scores
            if importance:
                max_importance = max(importance.values())
                if max_importance > 0:
                    importance = {k: v/max_importance for k, v in importance.items()}

        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        return importance

    def _get_domain_feature_weight(self, domain: DomainType, feature_name: str) -> float:
        """Get domain-specific feature weights"""
        domain_weights = {
            DomainType.INFRASTRUCTURE: {
                'cpu': 1.0, 'memory': 0.9, 'network': 0.8, 'disk': 0.7,
                'cpu_usage': 1.0, 'memory_usage': 0.9, 'network_latency': 0.8
            },
            DomainType.FINANCE: {
                'price': 1.0, 'volume': 0.8, 'risk': 0.9, 'return': 0.95,
                'volatility': 0.85, 'liquidity': 0.7
            },
            DomainType.HEALTHCARE: {
                'temperature': 0.9, 'heart_rate': 0.85, 'blood_pressure': 0.9,
                'symptoms': 1.0, 'age': 0.7, 'vital_signs': 0.95
            }
        }

        domain_dict = domain_weights.get(domain, {})
        # Check for exact match or partial match
        for key, weight in domain_dict.items():
            if key.lower() in feature_name.lower():
                return weight
        return 0.5  # Default weight

    def _analyze_attention_weights(self, task: UniversalTask, model: UniversalNeuralArchitecture) -> Dict[str, Any]:
        """Analyze attention mechanism weights"""
        attention_analysis = {
            'attention_available': False,
            'attention_patterns': [],
            'focus_areas': [],
            'attention_distribution': {}
        }

        try:
            if PYTORCH_AVAILABLE and hasattr(model, 'attention'):
                attention_analysis['attention_available'] = True
                # This would require storing attention weights during forward pass
                # For now, we'll simulate attention analysis
                if isinstance(task.input_data, str):
                    words = task.input_data.split()
                    # Simulate attention weights (in real implementation, get from model)
                    simulated_weights = [0.1 + 0.8 * (len(word) / 10) for word in words]
                    total_weight = sum(simulated_weights)
                    normalized_weights = [w/total_weight for w in simulated_weights]

                    attention_analysis['attention_patterns'] = [
                        {'token': word, 'weight': weight}
                        for word, weight in zip(words, normalized_weights)
                    ]

                    # Identify focus areas (top 3 attended tokens)
                    sorted_attention = sorted(
                        zip(words, normalized_weights),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    attention_analysis['focus_areas'] = [
                        {'token': token, 'weight': weight}
                        for token, weight in sorted_attention[:3]
                    ]

        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")

        return attention_analysis

    def _generate_counterfactuals(self, task: UniversalTask, solution: UniversalSolution, model: UniversalNeuralArchitecture) -> List[Dict]:
        """Generate counterfactual explanations"""
        counterfactuals = []

        try:
            # Generate simple counterfactuals by modifying input features
            if isinstance(task.input_data, dict):
                for key, value in task.input_data.items():
                    if isinstance(value, (int, float)):
                        # Generate counterfactuals by changing this feature
                        for multiplier in [0.5, 0.8, 1.2, 1.5]:
                            if multiplier != 1.0:
                                modified_data = task.input_data.copy()
                                modified_data[key] = value * multiplier

                                counterfactual = {
                                    'modified_feature': key,
                                    'original_value': value,
                                    'modified_value': value * multiplier,
                                    'change_description': f"If {key} was {value * multiplier:.2f} instead of {value:.2f}",
                                    'predicted_impact': self._estimate_impact(key, value, value * multiplier, task.domain)
                                }
                                counterfactuals.append(counterfactual)

            # Limit to most impactful counterfactuals
            counterfactuals.sort(key=lambda x: abs(x['predicted_impact']), reverse=True)

        except Exception as e:
            logger.warning(f"Counterfactual generation failed: {e}")

        return counterfactuals[:5]  # Return top 5 counterfactuals

    def _estimate_impact(self, feature: str, original_value: float, modified_value: float, domain: DomainType) -> float:
        """Estimate the impact of changing a feature value"""
        change_ratio = abs(modified_value - original_value) / (abs(original_value) + 1e-6)
        domain_sensitivity = self._get_domain_feature_weight(domain, feature)
        return change_ratio * domain_sensitivity

    def _analyze_uncertainty(self, task: UniversalTask, solution: UniversalSolution, model: UniversalNeuralArchitecture) -> Dict[str, float]:
        """Analyze prediction uncertainty"""
        uncertainty_metrics = {
            'epistemic_uncertainty': 0.0,  # Model uncertainty
            'aleatoric_uncertainty': 0.0,  # Data uncertainty
            'total_uncertainty': 0.0,
            'confidence_interval_lower': 0.0,
            'confidence_interval_upper': 0.0
        }

        try:
            # Simplified uncertainty estimation
            base_uncertainty = 1.0 - solution.confidence

            # Epistemic uncertainty (model knowledge)
            uncertainty_metrics['epistemic_uncertainty'] = base_uncertainty * 0.6

            # Aleatoric uncertainty (data noise)
            data_noise = self._estimate_data_noise(task.input_data)
            uncertainty_metrics['aleatoric_uncertainty'] = data_noise * 0.4

            # Total uncertainty
            uncertainty_metrics['total_uncertainty'] = (
                uncertainty_metrics['epistemic_uncertainty'] +
                uncertainty_metrics['aleatoric_uncertainty']
            )

            # Confidence intervals (simplified)
            if isinstance(solution.solution, (int, float)):
                margin = uncertainty_metrics['total_uncertainty'] * abs(solution.solution)
                uncertainty_metrics['confidence_interval_lower'] = solution.solution - margin
                uncertainty_metrics['confidence_interval_upper'] = solution.solution + margin

        except Exception as e:
            logger.warning(f"Uncertainty analysis failed: {e}")

        return uncertainty_metrics

    def _estimate_data_noise(self, input_data: Any) -> float:
        """Estimate noise level in input data"""
        try:
            if isinstance(input_data, list) and all(isinstance(x, (int, float)) for x in input_data):
                if len(input_data) > 1:
                    return min(np.std(input_data) / (np.mean(np.abs(input_data)) + 1e-6), 1.0)
            elif isinstance(input_data, dict):
                numeric_values = [v for v in input_data.values() if isinstance(v, (int, float))]
                if numeric_values:
                    return min(np.std(numeric_values) / (np.mean(np.abs(numeric_values)) + 1e-6), 1.0)
        except:
            pass
        return 0.1  # Default low noise

    def _generate_domain_specific_explanation(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """Generate domain-specific explanations"""
        domain_explanation = {}

        try:
            if task.domain == DomainType.INFRASTRUCTURE:
                domain_explanation.update(self._explain_infrastructure_decision(task, solution))
            elif task.domain == DomainType.FINANCE:
                domain_explanation.update(self._explain_finance_decision(task, solution))
            elif task.domain == DomainType.HEALTHCARE:
                domain_explanation.update(self._explain_healthcare_decision(task, solution))
            elif task.domain == DomainType.NATURAL_LANGUAGE:
                domain_explanation.update(self._explain_nlp_decision(task, solution))
        except Exception as e:
            logger.warning(f"Domain-specific explanation failed: {e}")

        return domain_explanation

    def _explain_infrastructure_decision(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """Infrastructure-specific explanation"""
        explanation = {
            'domain_context': 'Infrastructure Management',
            'key_metrics_analyzed': [],
            'performance_indicators': {},
            'recommended_actions': [],
            'risk_assessment': {}
        }

        if isinstance(task.input_data, dict):
            # Analyze infrastructure metrics
            if 'cpu_usage' in task.input_data:
                cpu_data = task.input_data['cpu_usage']
                if isinstance(cpu_data, list):
                    avg_cpu = np.mean(cpu_data)
                    explanation['key_metrics_analyzed'].append(f"CPU Usage: {avg_cpu:.1%}")
                    explanation['performance_indicators']['cpu_status'] = (
                        'critical' if avg_cpu > 0.9 else
                        'warning' if avg_cpu > 0.7 else 'normal'
                    )

            if 'memory_usage' in task.input_data:
                memory_data = task.input_data['memory_usage']
                if isinstance(memory_data, list):
                    avg_memory = np.mean(memory_data)
                    explanation['key_metrics_analyzed'].append(f"Memory Usage: {avg_memory:.1%}")
                    explanation['performance_indicators']['memory_status'] = (
                        'critical' if avg_memory > 0.85 else
                        'warning' if avg_memory > 0.7 else 'normal'
                    )

        # Generate recommendations based on solution
        if isinstance(solution.solution, dict) and 'recommended_action' in solution.solution:
            action = solution.solution['recommended_action']
            explanation['recommended_actions'].append({
                'action': action,
                'rationale': f"Based on current system metrics, {action} is recommended",
                'priority': 'high' if solution.confidence > 0.8 else 'medium'
            })

        return explanation

    def _explain_finance_decision(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """Finance-specific explanation"""
        explanation = {
            'domain_context': 'Financial Analysis',
            'market_factors': [],
            'risk_metrics': {},
            'investment_rationale': [],
            'regulatory_considerations': []
        }

        # Analyze financial data patterns
        if isinstance(task.input_data, list) and all(isinstance(x, (int, float)) for x in task.input_data):
            price_data = task.input_data
            if len(price_data) > 1:
                returns = [(price_data[i] - price_data[i-1]) / price_data[i-1] for i in range(1, len(price_data))]
                volatility = np.std(returns) if returns else 0
                trend = (price_data[-1] - price_data[0]) / price_data[0] if price_data[0] != 0 else 0

                explanation['market_factors'].append(f"Price trend: {trend:.2%}")
                explanation['market_factors'].append(f"Volatility: {volatility:.2%}")
                explanation['risk_metrics']['volatility'] = volatility
                explanation['risk_metrics']['trend_strength'] = abs(trend)

        return explanation

    def _explain_healthcare_decision(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """Healthcare-specific explanation"""
        explanation = {
            'domain_context': 'Healthcare Analysis',
            'clinical_indicators': [],
            'risk_factors': [],
            'medical_rationale': [],
            'confidence_factors': {}
        }

        if isinstance(task.input_data, dict):
            # Analyze vital signs
            if 'vital_signs' in task.input_data:
                vitals = task.input_data['vital_signs']
                if isinstance(vitals, dict):
                    for vital, value in vitals.items():
                        if isinstance(value, (int, float)):
                            normal_range = self._get_normal_range(vital)
                            status = self._assess_vital_status(vital, value, normal_range)
                            explanation['clinical_indicators'].append({
                                'indicator': vital,
                                'value': value,
                                'status': status,
                                'normal_range': normal_range
                            })

            # Analyze symptoms
            if 'symptoms' in task.input_data:
                symptoms = task.input_data['symptoms']
                if isinstance(symptoms, list):
                    explanation['clinical_indicators'].append({
                        'indicator': 'symptoms',
                        'value': ', '.join(symptoms),
                        'count': len(symptoms)
                    })

        return explanation

    def _get_normal_range(self, vital: str) -> Dict[str, float]:
        """Get normal ranges for vital signs"""
        normal_ranges = {
            'temperature': {'min': 97.0, 'max': 99.5},
            'heart_rate': {'min': 60, 'max': 100},
            'systolic_bp': {'min': 90, 'max': 140},
            'diastolic_bp': {'min': 60, 'max': 90}
        }
        return normal_ranges.get(vital, {'min': 0, 'max': 100})

    def _assess_vital_status(self, vital: str, value: float, normal_range: Dict[str, float]) -> str:
        """Assess if vital sign is normal, low, or high"""
        if value < normal_range['min']:
            return 'low'
        elif value > normal_range['max']:
            return 'high'
        else:
            return 'normal'

    def _explain_nlp_decision(self, task: UniversalTask, solution: UniversalSolution) -> Dict[str, Any]:
        """NLP-specific explanation"""
        explanation = {
            'domain_context': 'Natural Language Processing',
            'text_analysis': {},
            'linguistic_features': [],
            'sentiment_indicators': [],
            'key_phrases': []
        }

        if isinstance(task.input_data, str):
            text = task.input_data

            # Basic text analysis
            explanation['text_analysis'] = {
                'word_count': len(text.split()),
                'character_count': len(text),
                'sentence_count': text.count('.') + text.count('!') + text.count('?')
            }

            # Identify key phrases (simplified)
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get most frequent meaningful words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            explanation['key_phrases'] = [word for word, freq in sorted_words[:5]]

            # Simple sentiment indicators
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'fantastic', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing']

            pos_count = sum(1 for word in positive_words if word in text.lower())
            neg_count = sum(1 for word in negative_words if word in text.lower())

            explanation['sentiment_indicators'] = {
                'positive_signals': pos_count,
                'negative_signals': neg_count,
                'sentiment_balance': pos_count - neg_count
            }

        return explanation

    def _generate_human_readable_summary(self, task: UniversalTask, solution: UniversalSolution, explanation: Dict[str, Any]) -> str:
        """Generate human-readable explanation summary"""
        try:
            summary_parts = []

            # Basic decision summary
            summary_parts.append(f"For this {task.domain.value} {task.task_type.value} task, ")

            # Confidence statement
            confidence_level = (
                "high confidence" if solution.confidence > 0.8 else
                "moderate confidence" if solution.confidence > 0.6 else
                "low confidence"
            )
            summary_parts.append(f"the system reached a decision with {confidence_level} (score: {solution.confidence:.2f}). ")

            # Key factors
            decision_factors = explanation.get('decision_factors', [])
            if decision_factors:
                top_factor = decision_factors[0]
                summary_parts.append(f"The most influential factor was {top_factor['description']}. ")

            # Domain-specific insights
            if task.domain == DomainType.INFRASTRUCTURE:
                summary_parts.append("The analysis considered system performance metrics and resource utilization patterns. ")
            elif task.domain == DomainType.FINANCE:
                summary_parts.append("The decision incorporated market trends, risk factors, and financial indicators. ")
            elif task.domain == DomainType.HEALTHCARE:
                summary_parts.append("The assessment evaluated clinical indicators and medical risk factors. ")

            # Uncertainty mention
            uncertainty = explanation.get('uncertainty_metrics', {}).get('total_uncertainty', 0)
            if uncertainty > 0.3:
                summary_parts.append(f"Note: There is some uncertainty in this prediction (uncertainty score: {uncertainty:.2f}). ")

            # Recommendation
            if solution.confidence > 0.7:
                summary_parts.append("This recommendation can be acted upon with confidence.")
            else:
                summary_parts.append("Consider gathering additional data or seeking expert review before acting on this recommendation.")

            return ''.join(summary_parts)

        except Exception as e:
            logger.warning(f"Human-readable summary generation failed: {e}")
            return f"The system processed a {task.domain.value} task with {solution.confidence:.2f} confidence."

    def _generate_technical_details(self, task: UniversalTask, solution: UniversalSolution,
                                  model: UniversalNeuralArchitecture, model_output: Any) -> Dict[str, Any]:
        """Generate technical details for expert users"""
        technical_details = {
            'model_architecture': {
                'type': 'UniversalNeuralArchitecture',
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims,
                'output_dim': model.output_dim,
                'domain_adapted': solution.domain_adapted
            },
            'processing_pipeline': [],
            'mathematical_details': {},
            'performance_metrics': {
                'execution_time': solution.execution_time,
                'confidence_score': solution.confidence,
                'model_version': solution.model_used
            },
            'data_preprocessing': {},
            'postprocessing_steps': []
        }

        try:
            # Add processing pipeline details
            technical_details['processing_pipeline'] = [
                'Input data validation and preprocessing',
                'Domain-specific feature extraction',
                'Neural network forward pass',
                'Attention mechanism application (if applicable)',
                'Domain-specific postprocessing',
                'Confidence calculation and calibration'
            ]

            # Mathematical details
            if isinstance(model_output, dict) and 'probabilities' in model_output:
                probs = model_output['probabilities']
                if probs:
                    technical_details['mathematical_details']['entropy'] = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                    technical_details['mathematical_details']['max_probability'] = max(probs)
                    technical_details['mathematical_details']['probability_distribution'] = probs

            # Data preprocessing details
            technical_details['data_preprocessing'] = {
                'input_type': type(task.input_data).__name__,
                'preprocessing_applied': ['normalization', 'feature_extraction'],
                'input_shape': self._get_input_shape(task.input_data)
            }

        except Exception as e:
            logger.warning(f"Technical details generation failed: {e}")

        return technical_details

    def _get_input_shape(self, input_data: Any) -> str:
        """Get string representation of input data shape"""
        try:
            if isinstance(input_data, list):
                return f"List[{len(input_data)}]"
            elif isinstance(input_data, dict):
                return f"Dict[{len(input_data)} keys]"
            elif isinstance(input_data, str):
                return f"String[{len(input_data)} chars]"
            elif hasattr(input_data, 'shape'):
                return str(input_data.shape)
            else:
                return str(type(input_data).__name__)
        except:
            return "Unknown"

    def _generate_visualizations(self, task: UniversalTask, solution: UniversalSolution, explanation: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization data/configs"""
        visualizations = {}

        try:
            # Feature importance visualization
            feature_importance = explanation.get('feature_importance', {})
            if feature_importance:
                visualizations['feature_importance'] = {
                    'type': 'bar_chart',
                    'data': feature_importance,
                    'title': 'Feature Importance Analysis',
                    'description': 'Relative importance of input features in the decision'
                }

            # Confidence breakdown visualization
            confidence_breakdown = explanation.get('confidence_breakdown', {})
            if confidence_breakdown:
                visualizations['confidence_breakdown'] = {
                    'type': 'stacked_bar',
                    'data': confidence_breakdown,
                    'title': 'Confidence Score Breakdown',
                    'description': 'Components contributing to the overall confidence score'
                }

            # Attention weights visualization (if available)
            attention_analysis = explanation.get('attention_analysis', {})
            if attention_analysis.get('attention_available') and attention_analysis.get('attention_patterns'):
                visualizations['attention_weights'] = {
                    'type': 'heatmap',
                    'data': attention_analysis['attention_patterns'],
                    'title': 'Attention Weights Visualization',
                    'description': 'Model attention focus across input elements'
                }

            # Uncertainty visualization
            uncertainty_metrics = explanation.get('uncertainty_metrics', {})
            if uncertainty_metrics:
                visualizations['uncertainty_analysis'] = {
                    'type': 'gauge_chart',
                    'data': uncertainty_metrics,
                    'title': 'Prediction Uncertainty Analysis',
                    'description': 'Breakdown of prediction uncertainty components'
                }

        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        return visualizations

    def _assess_data_quality(self, input_data: Any) -> float:
        """Assess quality of input data"""
        try:
            if isinstance(input_data, dict):
                # Check for missing values, data types, etc.
                total_fields = len(input_data)
                valid_fields = sum(1 for v in input_data.values() if v is not None and v != "")
                return valid_fields / total_fields if total_fields > 0 else 0.5
            elif isinstance(input_data, list):
                if not input_data:
                    return 0.1
                # Check for consistency and completeness
                non_null_count = sum(1 for x in input_data if x is not None)
                return non_null_count / len(input_data)
            elif isinstance(input_data, str):
                # Basic text quality assessment
                if len(input_data.strip()) == 0:
                    return 0.1
                word_count = len(input_data.split())
                return min(word_count / 10, 1.0)  # Normalize by expected word count
            else:
                return 0.7  # Default for other types
        except Exception:
            return 0.5  # Default quality score

    def _calculate_prediction_uncertainty(self, model_output: Any) -> float:
        """Calculate prediction uncertainty from model output"""
        try:
            if isinstance(model_output, dict) and 'probabilities' in model_output:
                probs = model_output['probabilities']
                if probs and isinstance(probs, list):
                    # Calculate entropy as uncertainty measure
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                    max_entropy = np.log(len(probs))  # Maximum possible entropy
                    return entropy / max_entropy if max_entropy > 0 else 0.5
            return 0.3  # Default uncertainty
        except Exception:
            return 0.5  # Default uncertainty on error

class WorldClassUniversalNeuralSystem:
    """World-class universal neural network system that can handle any domain"""

    def __init__(self):
        self.start_time = time.time()
        self.system_id = f"universal_ai_{int(self.start_time)}"

        logger.info("🔧 Initializing core components...")

        # Core components
        logger.debug("🔧 Creating neural architecture...")
        self.neural_architecture = UniversalNeuralArchitecture()
        logger.debug("✅ Neural architecture created")

        logger.debug("🔧 Creating domain adapter...")
        self.domain_adapter = DomainAdapter()
        logger.debug("✅ Domain adapter created")

        logger.debug("🔧 Creating learning engine...")
        self.learning_engine = UniversalLearningEngine()
        logger.debug("✅ Learning engine created")

        # Task management
        logger.debug("🔧 Initializing task management...")
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else []
        logger.debug("✅ Task management initialized")

        # Performance tracking
        logger.debug("🔧 Initializing performance tracking...")
        self.performance_metrics = {
            'total_tasks_processed': 0,
            'successful_resolutions': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0,
            'domains_mastered': set(),
            'learning_iterations': 0
        }
        logger.debug("✅ Performance tracking initialized")

        # Knowledge base
        logger.debug("🔧 Initializing knowledge base...")
        self.universal_knowledge_base = {}
        self.domain_expertise_levels = {}

        # Model management
        self.model_versions = {}
        self.best_models_per_domain = {}

        # Initialize domain expertise tracking
        for domain in DomainType:
            self.domain_expertise_levels[domain.value] = 0.0
        logger.debug("✅ Knowledge base initialized")

        logger.info("🌟 World-Class Universal Neural System initialized")
        logger.info(f"🧠 System ID: {self.system_id}")
        logger.info(f"🔧 PyTorch Available: {PYTORCH_AVAILABLE}")
        logger.info(f"🔧 TensorFlow Available: {TENSORFLOW_AVAILABLE}")
        logger.info(f"🔧 Transformers Available: {TRANSFORMERS_AVAILABLE}")
        logger.info(f"🔧 Scikit-learn Available: {SKLEARN_AVAILABLE}")

        # Add database integration
        logger.debug("🔧 Initializing database...")
        try:
            self.database = UniversalDatabase()
            logger.info("✅ Database integration initialized")

            # Test database connection
            if self.database.test_connection():
                logger.info(f"✅ Database connection verified: {self.database.db_type}")
                task_count = self.database.get_task_count()
                logger.info(f"📊 Existing tasks in database: {task_count}")
            else:
                logger.warning("⚠️ Database connection test failed")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.database = None

        logger.debug("🔧 Initializing explainable AI...")
        self.explainable_ai = ExplainableAIEngine()
        logger.info("🔍 Explainable AI engine initialized")

        logger.debug("🔧 Integrating precheck system...")
        try:
            self.precheck_manager = precheck_manager
            self.precheck_processor = precheck_manager.get_processor()

            # Integrate the precheck engine with this universal system
            if (self.precheck_processor and
                hasattr(self.precheck_processor, 'precheck_engine') and
                hasattr(self.precheck_processor.precheck_engine, 'integrate_with_universal_system')):

                integration_success = self.precheck_processor.precheck_engine.integrate_with_universal_system(self)
                if integration_success:
                    logger.info("✅ Precheck system fully integrated with Universal Neural System")
                else:
                    logger.warning("⚠️ Precheck integration partially failed")

            logger.info("✅ Precheck system integrated")
        except Exception as e:
            logger.error(f"❌ Precheck integration failed: {e}")
            self.precheck_manager = None
            self.precheck_processor = None

        logger.info("✅ All components initialized successfully")

        # Initialize Wiki Knowledge Base - ENHANCED WITH NEURAL INTEGRATION
        logger.debug("🔧 Initializing Wiki Knowledge Base with Neural Integration...")
        try:
            # Pass the universal system to enable neural fallback
            self.wiki_kb = WikiKnowledgeBase(universal_system=self)

            if hasattr(self.wiki_kb, 'initialized') and self.wiki_kb.initialized:
                logger.info("✅ Wiki Knowledge Base integrated with Universal Neural System")
                logger.info(f"📚 Wiki pages loaded: {len(self.wiki_kb.pages)}")
                logger.info("🧠 Neural fallback enabled for enhanced AI responses")

                # Add to performance metrics
                if 'wiki_questions_processed' not in self.performance_metrics:
                    self.performance_metrics['wiki_questions_processed'] = 0

                # Log some sample pages
                if self.wiki_kb.pages:
                    sample_pages = list(self.wiki_kb.pages.items())[:3]
                    for page_id, page_data in sample_pages:
                        logger.info(f"  📄 {page_data['title']} ({page_data['content_length']} chars)")
            else:
                logger.warning("⚠️ Wiki Knowledge Base not properly initialized")
                self.wiki_kb = None

        except ImportError as e:
            logger.error(f"❌ Failed to import WikiKnowledgeBase: {e}")
            self.wiki_kb = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Wiki Knowledge Base: {e}")
            logger.info(f"   Error details: {str(e)}")
            self.wiki_kb = None

    # Add this method to handle wiki questions
    async def process_wiki_question(self, question: str, max_tokens: int = 1000, include_sources: bool = True) -> Dict:
        """Process a wiki-based question"""
        if not self.wiki_kb or not self.wiki_kb.initialized:
            return {
                "success": False,
                "error": "Wiki Knowledge Base not available",
                "answer": "Wiki Knowledge Base is not initialized. Please check if wiki extraction files are available.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Update performance metrics
            start_time = time.time()
            result = await self.wiki_kb.answer_question(question, max_tokens, include_sources)
            # Trace wiki question processing
            processing_time = time.time() - start_time
            if 'wiki_questions_processed' not in self.performance_metrics:
                self.performance_metrics['wiki_questions_processed'] = 0
            self.performance_metrics['wiki_questions_processed'] += 1
            return {
                "success": True,
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence_score": result["confidence_score"],
                "response_time": result["response_time"],
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing wiki question: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"Error processing question: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def process_precheck_task_sync(self, task: UniversalTask) -> UniversalSolution:
        """Improved synchronous precheck task processing"""
        start_time = time.time()

        try:
            logger.info(f"🔍 Processing precheck task {task.task_id}")

            # Use centralized precheck manager
            processor = precheck_manager.get_processor()
            if processor:
                try:
                    result = processor.process_task_sync(task)
                    execution_time = time.time() - start_time

                    return UniversalSolution(
                        task_id=task.task_id,
                        solution=result,
                        confidence=result.get('confidence', 0.8),
                        reasoning=result.get('explanation', 'Precheck analysis completed'),
                        execution_time=execution_time,
                        model_used="centralized_precheck_processor",
                        domain_adapted=True
                    )
                except Exception as e:
                    logger.error(f"Precheck processor failed: {e}")
                    return self._fallback_precheck_processing(task, start_time)
            else:
                return self._fallback_precheck_processing(task, start_time)

        except Exception as e:
            logger.error(f"Precheck task processing failed: {e}")
            execution_time = time.time() - start_time

            return UniversalSolution(
                task_id=task.task_id,
                solution={"decision": "ERROR", "error": str(e)},
                confidence=0.0,
                reasoning=f"Precheck processing failed: {str(e)}",
                execution_time=execution_time,
                model_used="error_handler"
            )

    def _fallback_precheck_processing(self, task: UniversalTask, start_time: float) -> UniversalSolution:
        """Enhanced fallback using existing precheck engine"""
        try:
            # Use the global precheck_manager to get the engine
            processor = precheck_manager.get_processor()
            if processor and hasattr(processor, 'precheck_engine'):
                engine = processor.precheck_engine

                # Convert task to PrecheckFailure format
                failure = self._task_to_precheck_failure(task)

                # Use your existing AI analysis method
                result = engine.ai_analyze_failure(failure)

                execution_time = time.time() - start_time
                return UniversalSolution(
                    task_id=task.task_id,
                    solution=result,
                    confidence=result.get('confidence', 0.7),
                    reasoning=result.get('reasoning', 'Precheck engine analysis'),
                    execution_time=execution_time,
                    model_used="existing_precheck_engine"
                )
            else:
                return self._basic_precheck_fallback(task, start_time)

        except Exception as e:
            logger.error(f"Fallback precheck processing failed: {e}")
            execution_time = time.time() - start_time
            return UniversalSolution(
                task_id=task.task_id,
                solution={"decision": "REQUIRE_MANUAL_REVIEW", "error": str(e)},
                confidence=0.3,
                reasoning=f"Precheck fallback failed: {str(e)}",
                execution_time=execution_time,
                model_used="error_fallback"
            )

    def _task_to_precheck_failure(self, task: UniversalTask):
        """Convert UniversalTask to PrecheckFailure format"""
        from core.precheck_engine import PrecheckFailure, IngredientType

        input_data = task.input_data
        return PrecheckFailure(
            failure_id=task.task_id,
            precheck_name=input_data.get('precheck_name', 'unknown'),
            ingredient_type=IngredientType.INTEL,  # Default
            ingredient_name=input_data.get('ingredient_name', 'unknown'),
            failure_details=input_data,
            milestone=input_data.get('milestone', 'unknown'),
            timestamp=task.timestamp,
            error_message=input_data.get('error_message', ''),
            metadata=task.metadata
        )

    def cleanup_resources(self):
        """Clean up system resources periodically"""
        try:
            # Clear old completed tasks
            current_time = time.time()
            old_tasks = [
                task_id for task_id, data in self.completed_tasks.items()
                if current_time - data['task'].timestamp > 3600  # 1 hour
            ]
            for task_id in old_tasks:
                del self.completed_tasks[task_id]

            # Force garbage collection
            gc.collect()

            # Clear neural network caches if available
            if hasattr(self.neural_architecture, '_projection_cache'):
                if len(self.neural_architecture._projection_cache) > 100:
                    # Keep only most recent projections
                    recent_keys = list(self.neural_architecture._projection_cache.keys())[-50:]
                    self.neural_architecture._projection_cache = {
                        k: self.neural_architecture._projection_cache[k] for k in recent_keys
                    }

            # Clear learning engine buffer if it gets too large
            if hasattr(self.learning_engine, 'continual_learning_buffer'):
                if len(self.learning_engine.continual_learning_buffer) > 1000:
                    # Keep only most recent learning examples
                    self.learning_engine.continual_learning_buffer = \
                        self.learning_engine.continual_learning_buffer[-500:]

            # Database cleanup if available
            if self.database and hasattr(self.database, 'cleanup_old_tasks'):
                try:
                    deleted_count = self.database.cleanup_old_tasks(days_old=7)  # Clean tasks older than 7 days
                    if deleted_count > 0:
                        logger.debug(f"🧹 Cleaned up {deleted_count} old database tasks")
                except Exception as e:
                    logger.warning(f"Database cleanup failed: {e}")

            logger.debug("🧹 System resources cleaned up successfully")

        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

    def process_universal_task(self, task: UniversalTask) -> UniversalSolution:
        """Process any universal task and provide a solution"""
        start_time = time.time()
        try:
            logger.info(f"🎯 Processing task {task.task_id} - Domain: {task.domain.value}, Type: {task.task_type.value}")

            # Add to active tasks
            self.active_tasks[task.task_id] = task

            # Analyze and adapt for domain
            domain_analysis = self.domain_adapter.analyze_domain(task)
            adapted_model = self.domain_adapter.adapt_for_domain(self.neural_architecture, task)

            # Preprocess input data
            processed_input = self._preprocess_input(task, domain_analysis)

            # Generate solution using adapted model
            raw_solution = self._generate_solution(adapted_model, processed_input, task)

            # Postprocess solution
            final_solution = self._postprocess_solution(raw_solution, task, domain_analysis)

            # Calculate confidence and reasoning
            confidence = self._calculate_confidence(task, final_solution, domain_analysis)
            reasoning = self._generate_reasoning(task, final_solution, domain_analysis)

            execution_time = time.time() - start_time

            # Create solution object
            solution = UniversalSolution(
                task_id=task.task_id,
                solution=final_solution,
                confidence=confidence,
                reasoning=reasoning,
                execution_time=execution_time,
                model_used=f"universal_neural_v1_{task.domain.value}",
                domain_adapted=True,
                learned_patterns=domain_analysis.get('patterns', [])
            )

            try:
                explanation = self.explainable_ai.generate_explanation(
                    task, solution, adapted_model, raw_solution
                )
                solution.explanation = explanation
                solution.feature_importance = explanation.get('feature_importance', {})
                solution.decision_path = [factor['description'] for factor in explanation.get('decision_factors', [])]
                solution.counterfactuals = explanation.get('counterfactuals', [])
                solution.uncertainty_analysis = explanation.get('uncertainty_metrics', {})

                attention_analysis = explanation.get('attention_analysis', {})
                if attention_analysis.get('attention_available'):
                    solution.attention_weights = [
                        pattern['weight'] for pattern in attention_analysis.get('attention_patterns', [])
                    ]

                logger.debug(f"🔍 Explanation generated for task {task.task_id}")
            except Exception as e:
                logger.warning(f"Explanation generation failed for task {task.task_id}: {e}")
                solution.explanation = {'error': str(e), 'human_readable_summary': 'Explanation generation failed'}

            # Learn from this task-solution pair
            learning_result = self.learning_engine.learn_from_task(task, solution)

            # Update performance metrics
            self._update_performance_metrics(task, solution, learning_result)

            # Move from active to completed
            del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = {
                'task': task,
                'solution': solution,
                'learning_result': learning_result
            }

            # Save to database
            if self.database:
                try:
                    self.database.save_task_result(task, solution)
                    logger.debug(f"💾 Task {task.task_id} saved to database")
                except Exception as e:
                    logger.warning(f"Failed to save task to database: {e}")

            logger.info(f"✅ Task {task.task_id} completed - Confidence: {confidence:.3f}, Time: {execution_time:.3f}s")
            return solution

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Task {task.task_id} failed: {e}")

            # Create error solution
            error_solution = UniversalSolution(
                task_id=task.task_id,
                solution=f"Error: {str(e)}",
                confidence=0.0,
                reasoning=f"Task failed due to error: {str(e)}",
                execution_time=execution_time,
                model_used="error_handler",
                domain_adapted=False
            )

            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            return error_solution

    def get_database_status(self):
        """Get database status"""
        try:
            status = {
                'status': 'connected' if self.database.test_connection() else 'disconnected',
                'total_tasks': self.database.get_task_count(),
                'recent_tasks': self.database.get_recent_tasks(5),
                'domain_statistics': self.database.get_domain_statistics(),
                'database_info': self.database.get_database_info()
            }
            return status
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _preprocess_input(self, task: UniversalTask, domain_analysis: Dict[str, Any]) -> Any:
        """Preprocess input data based on domain analysis"""
        try:
            data = task.input_data
            preprocessing_steps = domain_analysis.get('preprocessing_steps', [])

            for step in preprocessing_steps:
                if step == 'normalize' and isinstance(data, (list, tuple)):
                    # Normalize numerical data
                    if all(isinstance(x, (int, float)) for x in data):
                        max_val = max(data) if data else 1
                        min_val = min(data) if data else 0
                        range_val = max_val - min_val if max_val != min_val else 1
                        data = [(x - min_val) / range_val for x in data]
                elif step == 'tokenize' and isinstance(data, str):
                    # Simple tokenization
                    data = data.lower().split()
                elif step == 'encode' and isinstance(data, list) and all(isinstance(x, str) for x in data):
                    # Simple encoding of text tokens
                    vocab = {word: i for i, word in enumerate(set(data))}
                    data = [vocab.get(word, 0) for word in data]
                elif step == 'pad_sequences' and isinstance(data, list):
                    # Pad sequences to fixed length
                    target_length = 512
                    if len(data) < target_length:
                        data.extend([0] * (target_length - len(data)))
                    else:
                        data = data[:target_length]

            return data
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return task.input_data

    def _generate_solution(self, model: UniversalNeuralArchitecture, processed_input: Any, task: UniversalTask) -> Any:
        """Generate solution using the neural model"""
        try:
            if PYTORCH_AVAILABLE:
                model.eval()
                with torch.no_grad():
                    # Convert input to tensor
                    if isinstance(processed_input, list):
                        if all(isinstance(x, (int, float)) for x in processed_input):
                            input_tensor = torch.tensor(processed_input, dtype=torch.float32)
                        else:
                            # Handle mixed types
                            numeric_input = []
                            for x in processed_input:
                                if isinstance(x, (int, float)):
                                    numeric_input.append(float(x))
                                elif isinstance(x, dict):
                                    # Handle dictionary inputs - extract numeric values
                                    if 'value' in x:
                                        numeric_input.append(float(x['value']))
                                    elif 'score' in x:
                                        numeric_input.append(float(x['score']))
                                    else:
                                        # Use hash of dict as numeric representation
                                        numeric_input.append(hash(str(x)) % 1000 / 1000.0)
                                else:
                                    numeric_input.append(hash(str(x)) % 1000 / 1000.0)
                            input_tensor = torch.tensor(numeric_input, dtype=torch.float32)
                    elif isinstance(processed_input, dict):
                        # Handle dictionary input - flatten to numeric values
                        numeric_values = []
                        for key, value in processed_input.items():
                            if isinstance(value, (int, float)):
                                numeric_values.append(float(value))
                            elif isinstance(value, list):
                                for item in value:
                                    if isinstance(item, (int, float)):
                                        numeric_values.append(float(item))
                                    else:
                                        numeric_values.append(hash(str(item)) % 1000 / 1000.0)
                            else:
                                numeric_values.append(hash(str(value)) % 1000 / 1000.0)

                        # Ensure we have at least some values
                        if not numeric_values:
                            numeric_values = [0.5] * 10

                        # Pad or truncate to reasonable size
                        if len(numeric_values) < 512:
                            numeric_values.extend([0.0] * (512 - len(numeric_values)))
                        else:
                            numeric_values = numeric_values[:512]

                        input_tensor = torch.tensor(numeric_values, dtype=torch.float32)
                    else:
                        # Convert other types to tensor
                        if isinstance(processed_input, str):
                            # Convert string to numerical representation
                            numeric_repr = [ord(c) / 255.0 for c in processed_input[:512]]
                            if len(numeric_repr) < 512:
                                numeric_repr.extend([0.0] * (512 - len(numeric_repr)))
                            input_tensor = torch.tensor(numeric_repr, dtype=torch.float32)
                        else:
                            try:
                                input_tensor = torch.tensor([float(processed_input)], dtype=torch.float32)
                            except (ValueError, TypeError):
                                # Fallback for non-convertible types
                                input_tensor = torch.tensor([0.5], dtype=torch.float32)

                    # Ensure input has correct dimensions
                    if input_tensor.dim() == 1:
                        input_tensor = input_tensor.unsqueeze(0)

                    # Forward pass
                    output = model(input_tensor, domain=task.domain)

                    # Convert output based on task type
                    if task.task_type == TaskType.CLASSIFICATION:
                        probabilities = torch.softmax(output, dim=-1)
                        predicted_class = torch.argmax(probabilities, dim=-1)
                        return {
                            'predicted_class': int(predicted_class.item()),
                            'probabilities': [float(p) for p in probabilities.squeeze().tolist()]
                        }
                    elif task.task_type == TaskType.REGRESSION:
                        output_squeezed = output.squeeze()
                        if output_squeezed.numel() == 1:
                            return float(output.squeeze().item())
                        else:
                            return float(output_squeezed.mean().item())
            else:
                # Fallback implementation
                output = model.forward(processed_input, domain=task.domain)
                if task.task_type == TaskType.CLASSIFICATION:
                    # Simple softmax and argmax
                    import math
                    exp_output = [math.exp(x) for x in output]
                    sum_exp = sum(exp_output)
                    probabilities = [x / sum_exp for x in exp_output]
                    predicted_class = probabilities.index(max(probabilities))
                    return {
                        'predicted_class': predicted_class,
                        'probabilities': probabilities
                    }
                elif task.task_type == TaskType.REGRESSION:
                    return float(output[0]) if output else 0.0
                else:
                    return output
        except Exception as e:
            logger.error(f"Solution generation failed: {e}")
            return self._generate_fallback_solution(task)

    def _generate_fallback_solution(self, task: UniversalTask) -> Any:
        """Generate a fallback solution when the main model fails"""
        if task.task_type == TaskType.CLASSIFICATION:
            return {'predicted_class': 0, 'probabilities': [1.0]}
        elif task.task_type == TaskType.REGRESSION:
            return 0.0
        elif task.task_type in [TaskType.TEXT_GENERATION, TaskType.CODE_GENERATION]:
            return f"Generated response for {task.domain.value} task"
        elif task.task_type == TaskType.RECOMMENDATION:
            return ["recommendation_1", "recommendation_2", "recommendation_3"]
        else:
            return f"Solution for {task.task_type.value} in {task.domain.value}"

    def _postprocess_solution(self, raw_solution: Any, task: UniversalTask, domain_analysis: Dict[str, Any]) -> Any:
        """Postprocess the raw solution"""
        try:
            solution = raw_solution
            postprocessing_steps = domain_analysis.get('postprocessing_steps', [])

            for step in postprocessing_steps:
                if step == 'denormalize' and isinstance(solution, (int, float)):
                    # Denormalize if we have the original range
                    solution = solution * 100  # Example denormalization
                elif step == 'decode' and isinstance(solution, list):
                    # Decode numerical tokens back to text (simplified)
                    if all(isinstance(x, (int, float)) for x in solution):
                        solution = ' '.join([f"token_{int(x)}" for x in solution[:10]])
                elif step == 'format_output':
                    # Format output based on domain
                    if task.domain == DomainType.PRECHECK_VALIDATION:
                        solution = self._format_precheck_output(solution, task)
                    elif task.domain == DomainType.INFRASTRUCTURE:
                        solution = self._format_infrastructure_output(solution)
                    elif task.domain == DomainType.FINANCE:
                        solution = self._format_finance_output(solution)
                    elif task.domain == DomainType.HEALTHCARE:
                        solution = self._format_healthcare_output(solution)

            return solution
        except Exception as e:
            logger.warning(f"Postprocessing failed: {e}")
            return raw_solution

    def _format_infrastructure_output(self, solution: Any) -> Dict[str, Any]:
        """Format infrastructure-specific output"""
        if isinstance(solution, dict) and 'predicted_class' in solution:
            class_names = ['scale_up', 'scale_down', 'optimize', 'maintain', 'alert']
            predicted_action = class_names[solution['predicted_class'] % len(class_names)]
            return {
                'recommended_action': predicted_action,
                'confidence_scores': solution.get('probabilities', []),
                'infrastructure_impact': 'medium',
                'estimated_cost': 50.0
            }
        return {'recommended_action': 'monitor', 'confidence_scores': [1.0]}

    def _format_finance_output(self, solution: Any) -> Dict[str, Any]:
        """Format finance-specific output"""
        if isinstance(solution, (int, float)):
            return {
                'predicted_value': float(solution),
                'risk_level': 'medium',
                'confidence_interval': [solution * 0.9, solution * 1.1],
                'recommendation': 'hold' if abs(solution) < 0.1 else 'buy' if solution > 0 else 'sell'
            }
        return {'predicted_value': 0.0, 'risk_level': 'unknown', 'recommendation': 'hold'}

    def _format_healthcare_output(self, solution: Any) -> Dict[str, Any]:
        """Format healthcare-specific output"""
        if isinstance(solution, dict) and 'predicted_class' in solution:
            conditions = ['normal', 'mild_concern', 'moderate_concern', 'severe_concern', 'critical']
            predicted_condition = conditions[solution['predicted_class'] % len(conditions)]
            return {
                'assessment': predicted_condition,
                'confidence_scores': solution.get('probabilities', []),
                'recommended_actions': ['monitor', 'consult_specialist', 'immediate_attention'][solution['predicted_class'] % 3],
                'urgency_level': solution['predicted_class'] % 5
            }
        return {'assessment': 'normal', 'confidence_scores': [1.0], 'recommended_actions': 'monitor'}

    def _format_precheck_output(self, solution: Any, task: UniversalTask) -> Dict[str, Any]:
        """Format precheck validation specific output"""
        # Use modular processor if available
        if MODULAR_SYSTEM_AVAILABLE and precheck_processor:
            try:
                logger.debug(f"Using modular precheck processor for task {task.task_id}")
                return precheck_processor.format_precheck_output(solution, task)
            except Exception as e:
                logger.warning(f"Modular precheck processor failed: {e}, falling back to legacy")
                # Fall through to legacy implementation

        # Legacy implementation (original code preserved as fallback)
        logger.debug(f"Using legacy precheck processing for task {task.task_id}")

        # Extract input data for context
        input_data = task.input_data if hasattr(task, 'input_data') else {}

        # Determine decision based on neural network output
        if isinstance(solution, dict) and 'predicted_class' in solution:
            class_idx = solution['predicted_class']
            probabilities = solution.get('probabilities', [])
        elif isinstance(solution, list) and len(solution) > 0:
            # Convert raw neural output to decision
            probabilities = self._softmax(solution[:5])  # Take first 5 outputs
            class_idx = probabilities.index(max(probabilities))
        else:
            class_idx = 0
            probabilities = [1.0]

        # Define precheck decisions
        precheck_decisions = [
            "APPROVE",           # 0 - Green light
            "APPROVE_WITH_CONDITIONS",  # 1 - Yellow light with conditions
            "REQUIRE_MANUAL_REVIEW",    # 2 - Orange light - human review needed
            "REJECT",            # 3 - Red light - block deployment
            "ESCALATE"           # 4 - Purple light - escalate to senior team
        ]

        decision = precheck_decisions[class_idx % len(precheck_decisions)]
        confidence = max(probabilities) if probabilities else 0.5

        # Basic analysis for legacy mode
        test_coverage = input_data.get('test_coverage', 0)
        failed_tests = input_data.get('failed_tests', 0)
        security_issues = input_data.get('security_issues', 0)

        # Simple risk assessment
        risk_score = (failed_tests * 0.3) + (security_issues * 0.5) + ((1 - test_coverage) * 0.2)

        if risk_score <= 0.2:
            risk_level = "LOW"
        elif risk_score <= 0.5:
            risk_level = "MEDIUM"
        elif risk_score <= 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "decision": decision,
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 3),
            "explanation": f"Legacy precheck analysis: {decision} based on test coverage {test_coverage:.1%}, {failed_tests} failed tests, {security_issues} security issues",
            "metadata": {
                "legacy_mode": True,
                "analysis_timestamp": time.time(),
                "model_confidence": confidence,
                "decision_rationale": f"Legacy decision '{decision}' with {confidence:.1%} confidence"
            }
        }

    def _softmax(self, x):
        """Simple softmax implementation"""
        import math
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]

    def _calculate_confidence(self, task: UniversalTask, solution: Any, domain_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the solution"""
        try:
            base_confidence = 0.6

            # Adjust based on domain expertise
            domain_expertise = self.domain_expertise_levels.get(task.domain.value, 0.0)
            expertise_bonus = domain_expertise * 0.3

            # Adjust based on data quality
            complexity_score = domain_analysis.get('complexity_score', 1.0)
            complexity_penalty = min(complexity_score * 0.1, 0.3)

            # Adjust based on solution consistency
            consistency_bonus = 0.0
            if isinstance(solution, dict) and 'probabilities' in solution:
                probs = solution['probabilities']
                if probs and isinstance(probs, list):
                    # Ensure all probabilities are numbers
                    numeric_probs = []
                    for prob in probs:
                        if isinstance(prob, (int, float)):
                            numeric_probs.append(float(prob))
                        elif isinstance(prob, dict):
                            # If prob is a dict, try to extract a numeric value
                            if 'score' in prob:
                                numeric_probs.append(float(prob['score']))
                            elif 'confidence' in prob:
                                numeric_probs.append(float(prob['confidence']))
                            else:
                                numeric_probs.append(0.5)  # Default
                        else:
                            numeric_probs.append(0.5)  # Default for non-numeric

                    if numeric_probs:
                        max_prob = max(numeric_probs)
                        consistency_bonus = max_prob * 0.2

            # Adjust based on historical performance
            domain_key = task.domain.value
            if domain_key in self.learning_engine.meta_knowledge:
                recent_metrics = self.learning_engine.meta_knowledge[domain_key]['performance_metrics'][-10:]
                if recent_metrics:
                    avg_historical_confidence = sum(m['confidence'] for m in recent_metrics) / len(recent_metrics)
                    historical_bonus = avg_historical_confidence * 0.1
                else:
                    historical_bonus = 0.0
            else:
                historical_bonus = 0.0

            final_confidence = base_confidence + expertise_bonus - complexity_penalty + consistency_bonus + historical_bonus
            return max(0.0, min(1.0, final_confidence))
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _generate_reasoning(self, task: UniversalTask, solution: Any, domain_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for the solution"""
        try:
            reasoning_parts = []

            # Domain-specific reasoning
            reasoning_parts.append(f"Applied {task.domain.value} domain expertise")

            # Task-specific reasoning
            reasoning_parts.append(f"Executed {task.task_type.value} algorithm")

            # Data-specific reasoning
            data_chars = domain_analysis.get('data_characteristics', {})
            if data_chars.get('is_textual'):
                reasoning_parts.append("Processed textual data using NLP techniques")
            elif data_chars.get('is_numerical'):
                reasoning_parts.append("Analyzed numerical patterns and trends")
            elif data_chars.get('is_sequential'):
                reasoning_parts.append("Applied sequential analysis methods")

            # Solution-specific reasoning
            if isinstance(solution, dict) and 'predicted_class' in solution:
                reasoning_parts.append(f"Classification result based on pattern recognition")
            elif isinstance(solution, (int, float)):
                reasoning_parts.append(f"Numerical prediction using regression analysis")

            # Confidence reasoning
            confidence = self._calculate_confidence(task, solution, domain_analysis)
            if confidence > 0.8:
                reasoning_parts.append("High confidence due to strong pattern matches")
            elif confidence > 0.6:
                reasoning_parts.append("Moderate confidence with some uncertainty")
            else:
                reasoning_parts.append("Lower confidence due to limited data or complexity")

            return ". ".join(reasoning_parts) + "."
        except Exception as e:
            logger.warning(f"Reasoning generation failed: {e}")
            return f"Solution generated using {task.domain.value} domain knowledge and {task.task_type.value} methodology."

    def _update_performance_metrics(self, task: UniversalTask, solution: UniversalSolution, learning_result: Dict[str, Any]):
        """Update system performance metrics"""
        try:
            self.performance_metrics['total_tasks_processed'] += 1

            # Update successful resolutions
            if solution.confidence > 0.4:
                self.performance_metrics['successful_resolutions'] += 1

            # Update average confidence
            total_tasks = self.performance_metrics['total_tasks_processed']
            current_avg = self.performance_metrics['average_confidence']
            self.performance_metrics['average_confidence'] = (
                (current_avg * (total_tasks - 1) + solution.confidence) / total_tasks
            )

            # Update average execution time
            current_avg_time = self.performance_metrics['average_execution_time']
            self.performance_metrics['average_execution_time'] = (
                (current_avg_time * (total_tasks - 1) + solution.execution_time) / total_tasks
            )

            # Update domain expertise
            domain_key = task.domain.value
            current_expertise = self.domain_expertise_levels[domain_key]
            expertise_gain = solution.confidence * 0.01  # Small incremental learning
            self.domain_expertise_levels[domain_key] = min(1.0, current_expertise + expertise_gain)

            # Track mastered domains (expertise > 0.7)
            if self.domain_expertise_levels[domain_key] > 0.7:
                self.performance_metrics['domains_mastered'].add(domain_key)

            # Update learning iterations
            if learning_result.get('knowledge_updated', False):
                self.performance_metrics['learning_iterations'] += 1
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including wiki knowledge base"""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Database status
        db_status = "unknown"
        db_tasks = 0
        try:
            if hasattr(self, 'database') and self.database and hasattr(self.database, 'db_connection') and self.database.db_connection:
                cursor = self.database.db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM universal_ai.tasks")
                db_tasks = cursor.fetchone()[0]
                db_status = "connected"
                cursor.close()
        except Exception as e:
            db_status = f"error: {str(e)}"
            logger.warning(f"Database status check failed: {e}")

        # Wiki Knowledge Base status
        wiki_status = {
            "available": hasattr(self, 'wiki_kb') and self.wiki_kb is not None,
            "initialized": False,
            "total_pages": 0,
            "extraction_timestamp": None,
            "azure_openai_configured": False,
            "last_error": None
        }

        if hasattr(self, 'wiki_kb') and self.wiki_kb:
            try:
                wiki_status.update({
                    "initialized": self.wiki_kb.initialized,
                    "total_pages": len(self.wiki_kb.pages),
                    "extraction_timestamp": self.wiki_kb.wiki_data.get("extracted_at") if self.wiki_kb.wiki_data else None,
                    "azure_openai_configured": bool(self.wiki_kb.api_key and self.wiki_kb.endpoint_url),
                    "deployment_name": getattr(self.wiki_kb, 'deployment_name', 'unknown'),
                    "max_tokens_default": getattr(self.wiki_kb, 'max_tokens_default', 1000)
                })

                # Get sample page titles if available
                if self.wiki_kb.pages:
                    sample_pages = list(self.wiki_kb.pages.values())[:3]
                    wiki_status["sample_pages"] = [
                        {
                            "title": page["title"],
                            "content_length": page["content_length"]
                        }
                        for page in sample_pages
                    ]
            except Exception as e:
                wiki_status["last_error"] = str(e)
                logger.warning(f"Wiki status check failed: {e}")

        # Environment and configuration info
        environment_info = {
            "environment": os.getenv('ENVIRONMENT', 'development'),
            "deployment_id": os.getenv('DEPLOYMENT_ID', 'universal-ai'),
            "host": os.getenv('AI_ENGINE_HOST', 'localhost'),
            "port": get_env_int('AI_ENGINE_PORT', 8000),
            "default_host": get_default_host()
        }

        # Service endpoints
        service_endpoints = {}
        key_services = [
            'AI_ENGINE_MAIN', 'POSTGRESQL', 'REDIS', 'PROMETHEUS',
            'GRAFANA', 'ENHANCED_DASHBOARD', 'ALERT_PROCESSOR'
        ]
        for service in key_services:
            if service in PORTS:
                service_endpoints[service.lower()] = PORTS[service]

        # Security and monitoring status
        security_monitoring = {
            "encryption_enabled": is_encryption_available(),
            "ssl_enabled": get_env_bool('SSL_ENABLED', False),
            "prometheus_enabled": get_env_bool('PROMETHEUS_ENABLED', True),
            "grafana_enabled": get_env_bool('GRAFANA_ENABLED', True),
            "github_integration": get_env_bool('GITHUB_INTEGRATION_ENABLED', False)
        }

        status = {
            'system_id': self.system_id,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'performance_metrics': self.performance_metrics.copy(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'domain_expertise': self.domain_expertise_levels.copy(),
            'learning_insights': self.learning_engine.get_learning_insights(),
            'system_health': self._assess_system_health(),
            'capabilities': {
                'pytorch_available': PYTORCH_AVAILABLE,
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            },

            # NEW: Wiki Knowledge Base integration
            'wiki_knowledge_base': wiki_status,

            # NEW: Database information
            'database': {
                'status': db_status,
                'type': os.getenv('DB_TYPE', 'postgresql'),
                'host': os.getenv('POSTGRES_HOST'),
                'port': get_env_int('POSTGRES_PORT', 5432),
                'database': os.getenv('POSTGRES_DB'),
                'total_tasks': db_tasks
            },

            # NEW: Environment and deployment info
            'environment_info': environment_info,

            # NEW: Service endpoints
            'service_endpoints': service_endpoints,

            # NEW: Security and monitoring
            'security_monitoring': security_monitoring,

            'timestamp': current_time
        }

        # Convert set to list for JSON serialization
        status['performance_metrics']['domains_mastered'] = list(status['performance_metrics']['domains_mastered'])

        # Add overall health assessment
        status['overall_status'] = self._determine_overall_status(status)

        return status

    def _determine_overall_status(self, status: Dict) -> str:
        """Determine overall system status based on components"""
        try:
            issues = []

            # Check database
            if status['database']['status'] != 'connected':
                issues.append('database_disconnected')

            # Check AI capabilities
            if not any([
                status['capabilities']['pytorch_available'],
                status['capabilities']['tensorflow_available']
            ]):
                issues.append('no_ml_framework')

            # Check system health score
            health_score = status['system_health'].get('overall_score', 0)
            if health_score < 0.5:
                issues.append('low_health_score')

            # Check wiki knowledge base if expected
            if get_env_bool('WIKI_QA_ENABLED', True):
                if not status['wiki_knowledge_base']['available']:
                    issues.append('wiki_kb_unavailable')

            # Determine status
            if not issues:
                return 'healthy'
            elif len(issues) <= 2:
                return 'degraded'
            else:
                return 'unhealthy'

        except Exception as e:
            logger.error(f"Status determination error: {e}")
            return 'unknown'

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        health = {
            'overall_score': 0.0,
            'components': {
                'neural_architecture': 'healthy',
                'domain_adapter': 'healthy',
                'learning_engine': 'healthy',
                'memory_usage': 'normal',
                'processing_speed': 'normal'
            },
            'recommendations': []
        }

        try:
            # Calculate overall health score
            success_rate = (
                self.performance_metrics['successful_resolutions'] /
                max(1, self.performance_metrics['total_tasks_processed'])
            )
            avg_confidence = self.performance_metrics['average_confidence']
            avg_execution_time = self.performance_metrics['average_execution_time']
            domains_mastered_count = len(self.performance_metrics['domains_mastered'])

            # Health score calculation
            health_score = (
                success_rate * 0.4 +
                avg_confidence * 0.3 +
                min(1.0, domains_mastered_count / 10) * 0.2 +
                max(0.0, 1.0 - avg_execution_time / 10.0) * 0.1
            )

            health['overall_score'] = health_score

            # Generate recommendations
            if success_rate < 0.7:
                health['recommendations'].append("Consider retraining models for better success rate")
            if avg_confidence < 0.6:
                health['recommendations'].append("Improve confidence through more diverse training data")
            if avg_execution_time > 5.0:
                health['recommendations'].append("Optimize processing pipeline for better performance")
            if domains_mastered_count < 5:
                health['recommendations'].append("Expand training to cover more domains")

            # Component health assessment
            if len(self.active_tasks) > 100:
                health['components']['memory_usage'] = 'high'
                health['recommendations'].append("Consider task queue management")

            if avg_execution_time > 10.0:
                health['components']['processing_speed'] = 'slow'
                health['recommendations'].append("Optimize neural architecture")

        except Exception as e:
            logger.warning(f"Health assessment failed: {e}")
            health['components']['assessment'] = 'error'

        return health

    def save_system_state(self, filepath: str) -> bool:
        """Save the complete system state"""
        try:
            state = {
                'system_id': self.system_id,
                'start_time': self.start_time,
                'performance_metrics': self.performance_metrics.copy(),
                'domain_expertise_levels': self.domain_expertise_levels.copy(),
                'universal_knowledge_base': self.universal_knowledge_base.copy(),
                'learning_engine_meta_knowledge': self.learning_engine.meta_knowledge.copy(),
                'learning_history': self.learning_engine.learning_history.copy(),
                'adaptation_history': self.domain_adapter.adaptation_history.copy(),
                'completed_tasks_summary': {
                    task_id: {
                        'domain': data['task'].domain.value,
                        'task_type': data['task'].task_type.value,
                        'confidence': data['solution'].confidence,
                        'execution_time': data['solution'].execution_time
                    }
                    for task_id, data in self.completed_tasks.items()
                },
                'timestamp': time.time()
            }

            # Convert set to list for JSON serialization
            state['performance_metrics']['domains_mastered'] = list(state['performance_metrics']['domains_mastered'])

            # Save to file
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"💾 System state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            return False

    def load_system_state(self, filepath: str) -> bool:
        """Load system state from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"State file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore state
            self.system_id = state.get('system_id', self.system_id)
            self.start_time = state.get('start_time', self.start_time)

            # Restore performance metrics
            loaded_metrics = state.get('performance_metrics', {})
            for key, value in loaded_metrics.items():
                if key == 'domains_mastered':
                    self.performance_metrics[key] = set(value)
                else:
                    self.performance_metrics[key] = value

            # Restore domain expertise
            self.domain_expertise_levels.update(state.get('domain_expertise_levels', {}))

            # Restore knowledge bases
            self.universal_knowledge_base.update(state.get('universal_knowledge_base', {}))
            self.learning_engine.meta_knowledge.update(state.get('learning_engine_meta_knowledge', {}))
            self.learning_engine.learning_history.extend(state.get('learning_history', []))
            self.domain_adapter.adaptation_history.update(state.get('adaptation_history', {}))

            logger.info(f"📂 System state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            return False

# API Server for the Universal System
class UniversalNeuralAPI:
    """Universal Neural Network API with endpoint conflict resolution"""

    def __init__(self, universal_system):
        self.system = universal_system
        self.app = None

        if FLASK_AVAILABLE:
            from flask import Flask, request, jsonify
            self.app = Flask(__name__)
            # Clear any existing endpoints to prevent conflicts
            self.app.url_map._rules.clear()
            self.app.view_functions.clear()
            self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        if not self.app:
            return

        from flask import request, jsonify

        @self.app.route('/api/process_task', methods=['POST'])
        def process_task():
            try:
                data = request.json
                # Create universal task
                task = UniversalTask(
                    task_id=data.get('task_id', f"task_{int(time.time())}"),
                    domain=DomainType(data.get('domain', 'generic')),
                    task_type=TaskType(data.get('task_type', 'classification')),
                    input_data=data.get('input_data'),
                    expected_output=data.get('expected_output'),
                    metadata=data.get('metadata', {}),
                    priority=data.get('priority', 1),
                    user_id=data.get('user_id', 'api_user')
                )

                # Process task (synchronous for API)
                solution = self.system.process_universal_task(task)

                return jsonify({
                    'success': True,
                    'task_id': task.task_id,
                    'solution': solution.solution,
                    'confidence': solution.confidence,
                    'reasoning': solution.reasoning,
                    'execution_time': solution.execution_time,
                    'model_used': solution.model_used
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/system_status', methods=['GET'])
        def system_status():
            try:
                status = self.system.get_system_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/domains', methods=['GET'])
        def get_domains():
            return jsonify({
                'domains': [domain.value for domain in DomainType],
                'task_types': [task_type.value for task_type in TaskType]
            })

        @self.app.route('/api/wiki/ask', methods=['POST'])
        def wiki_ask():
            try:
                data = request.json
                question = data.get('question', '')

                if len(question.strip()) < 3:
                    return jsonify({
                        "success": False,
                        "error": "Question too short"
                    }), 400

                # Process question synchronously for API
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.system.process_wiki_question(question)
                    )
                finally:
                    loop.close()

                return jsonify(result)

            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify(health_check())

        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            try:
                status = self.system.get_system_status()
                metrics_lines = [
                    "# HELP universal_tasks_total Total tasks processed",
                    "# TYPE universal_tasks_total counter",
                    f"universal_tasks_total {status['performance_metrics']['total_tasks_processed']}",
                    "",
                    "# HELP universal_confidence_avg Average confidence",
                    "# TYPE universal_confidence_avg gauge",
                    f"universal_confidence_avg {status['performance_metrics']['average_confidence']}",
                    "",
                    "# HELP universal_health_score System health score",
                    "# TYPE universal_health_score gauge",
                    f"universal_health_score {status['system_health']['overall_score']}"
                ]
                return '\n'.join(metrics_lines), 200, {'Content-Type': 'text/plain'}
            except Exception as e:
                return f"# Error: {e}\n", 500, {'Content-Type': 'text/plain'}

# Configuration functions
def get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer"""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_env_bool(key: str, default: bool) -> bool:
    """Get environment variable as boolean"""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_default_host() -> str:
    """Get default host"""
    return os.getenv('DEFAULT_HOST', 'localhost')


# Main function
def main():
    """Main function"""
    try:
        logger.info("🚀 Starting Universal Neural System...")

        # Setup directories
        setup_directories()

        # Initialize system
        universal_system = WorldClassUniversalNeuralSystem()

        # Setup API server
        api_server = None
        if FLASK_AVAILABLE:
            api_server = UniversalNeuralAPI(universal_system)

            def start_api():
                try:
                    # Get API server configuration with fallbacks
                    if PORTS_AVAILABLE:
                        # Try to get AI_ENGINE_API service
                        api_service = port_registry.get_service('AI_ENGINE_API')
                        if api_service:
                            host, port = api_service.host, api_service.port
                        else:
                            # Fallback: try to use AI_ENGINE_MAIN with different port
                            main_service = port_registry.get_service('AI_ENGINE_MAIN')
                            if main_service:
                                host = main_service.host
                                port = 8090  # Default API port
                            else:
                                host, port = get_default_host(), 8090
                    else:
                        # Use fallback configuration
                        host = get_default_host()
                        port = PORTS.get('AI_ENGINE_API', 8090)

                    logger.info(f"🌐 Starting API server on {host}:{port}")
                    api_server.app.run(host=host, port=port, debug=False, threaded=True)

                except KeyError as e:
                    logger.error(f"API server configuration error: {e}")
                    logger.info("🔄 Trying fallback API configuration...")
                    try:
                        fallback_host = get_default_host()
                        fallback_port = 8090
                        logger.info(f"🌐 Starting API server on fallback {fallback_host}:{fallback_port}")
                        api_server.app.run(host=fallback_host, port=fallback_port, debug=False, threaded=True)
                    except Exception as fallback_error:
                        logger.error(f"Fallback API server also failed: {fallback_error}")

                except Exception as e:
                    logger.error(f"API server failed: {e}")

            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()

        # Display system info
        logger.info("=" * 80)
        logger.info("🌟 UNIVERSAL NEURAL SYSTEM READY")
        logger.info(f"🧠 System ID: {universal_system.system_id}")

        # Display API endpoint with proper error handling
        if api_server:
            try:
                if PORTS_AVAILABLE:
                    api_service = port_registry.get_service('AI_ENGINE_API')
                    if api_service:
                        logger.info(f"🌐 API: {api_service.url}")
                    else:
                        main_service = port_registry.get_service('AI_ENGINE_MAIN')
                        if main_service:
                            logger.info(f"🌐 API: http://{main_service.host}:8090")
                        else:
                            logger.info(f"🌐 API: http://{get_default_host()}:8090")
                else:
                    api_port = PORTS.get('AI_ENGINE_API', 8090)
                    logger.info(f"🌐 API: http://{get_default_host()}:{api_port}")
            except Exception as e:
                logger.warning(f"Could not determine API URL: {e}")
                logger.info(f"🌐 API: http://{get_default_host()}:8090 (fallback)")

        # Display service registry information
        if PORTS_AVAILABLE:
            logger.info("🔗 Service Registry:")
            logger.info(f"   📁 Config: {port_registry.registry_file}")
            logger.info(f"   🌍 Environment: {port_registry.get_environment()}")
            logger.info(f"   📊 Services: {len(port_registry.services)}")

            # Show key services
            key_services = ['AI_ENGINE_MAIN', 'PROMETHEUS', 'GRAFANA', 'ENHANCED_DASHBOARD']
            for service_name in key_services:
                service = port_registry.get_service(service_name)
                if service:
                    logger.info(f"   🎯 {service_name}: {service.url}")
        else:
            logger.info("🔗 Using Fallback Configuration")
            for service_name, port in PORTS.items():
                if service_name in ['AI_ENGINE_MAIN', 'AI_ENGINE_API', 'PROMETHEUS', 'GRAFANA']:
                    logger.info(f"   🎯 {service_name}: http://localhost:{port}")

        logger.info("=" * 80)

        # Main loop
        task_counter = 0
        while True:
            try:
                time.sleep(60)  # Process every minute
                task_counter += 1

                # Create sample task
                sample_task = UniversalTask(
                    task_id=f"sample_{task_counter}",
                    domain=DomainType.INFRASTRUCTURE,
                    task_type=TaskType.MONITORING,
                    input_data={'cpu_usage': 0.75, 'memory_usage': 0.60},
                    metadata={'source': 'continuous_learning'}
                )

                # Process task
                solution = universal_system.process_universal_task(sample_task)

                # Log progress every 10 tasks
                if task_counter % 10 == 0:
                    status = universal_system.get_system_status()
                    logger.info(f"📊 Progress: {status['performance_metrics']['total_tasks_processed']} tasks processed")

                    # Periodic cleanup
                    if task_counter % 100 == 0:
                        universal_system.cleanup_resources()
                        logger.info("🧹 Performed periodic cleanup")

            except KeyboardInterrupt:
                logger.info("🛑 Shutting down...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(30)

    except Exception as e:
        logger.error(f"System startup failed: {e}")
        import traceback
        traceback.print_exc()


# Signal handlers
def signal_handler(signum, frame):
    logger.info("🛑 Received shutdown signal")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Entry point
if __name__ == "__main__":
    main()
