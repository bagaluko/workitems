"""
Configuration and Settings for Secure Enterprise AI System
Enhanced with Environment-Based Configuration (.env)
"""
import os
import sys
import logging
import codecs
import secrets
import base64
import socket
import json
from collections import deque
from typing import Dict, List, Optional, Tuple

# Load environment variables first
try:
    from dotenv import load_dotenv
    # Load .env file from the aiengine directory
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    env_loaded = load_dotenv(env_path)

    if env_loaded:
        print(f"✅ Environment variables loaded from {env_path}")
    else:
        print(f"⚠️ .env file not found at {env_path}")
        # Try alternative locations
        alternative_paths = [
            '/aiengine/src/aiengine/.env',
            os.path.join(os.getcwd(), '.env'),
            '.env'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                load_dotenv(alt_path)
                print(f"✅ Environment variables loaded from {alt_path}")
                break
        else:
            print("⚠️ No .env file found, using system environment variables only")

except ImportError:
    print("⚠️ python-dotenv not installed")
    print("   Install with: pip install python-dotenv")
    print("   Using system environment variables only")

# Fix Unicode encoding for Windows console
try:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
except:
    pass  # Skip if already configured

# Create directories first
os.makedirs("logs", exist_ok=True)
os.makedirs("audit", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("config", exist_ok=True)

# ===== ENVIRONMENT-BASED DEFAULTS =====
def get_default_host():
    """Get the default host from environment variables"""
    return os.getenv('DEFAULT_HOST', os.getenv('AI_ENGINE_HOST', 'localhost'))

def get_default_port():
    """Get the default port from environment variables"""
    return get_env_int('DEFAULT_PORT', 8000)

# Setup logging FIRST before any other imports
def setup_logging():
    """Setup secure logging configuration"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', 'detailed')
    log_file_path = os.getenv('LOG_FILE_PATH', 'logs/enterprise_ai_secure.log')

    if log_format == 'detailed':
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    else:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path, encoding='utf-8')
        ]
    )

    # Setup audit logger
    audit_logger = logging.getLogger('audit')
    audit_handler = logging.FileHandler('audit/audit.log', encoding='utf-8')
    audit_handler.setFormatter(logging.Formatter('%(asctime)s - AUDIT - %(message)s'))
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    return audit_logger

# Setup logging immediately
setup_logging()
logger = logging.getLogger(__name__)

# ===== UTILITY FUNCTIONS =====
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on', 'enabled')

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float value from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid float value for {key}, using default: {default}")
        return default

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    """Get list value from environment variable"""
    if default is None:
        default = []
    value = os.getenv(key, '')
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]

def parse_port_config(port_config):
    """Parse port configuration from distributed registry format (HOST:PORT or just PORT)"""
    try:
        if isinstance(port_config, str):
            if ':' in port_config:
                host_port = port_config.split(':')
                if len(host_port) >= 2:
                    return int(host_port[-1])
                else:
                    return int(port_config)
            else:
                return int(port_config)
        elif isinstance(port_config, int):
            return port_config
        else:
            logger.warning(f"Unknown port config format: {port_config}, using default {get_default_port()}")
            return get_default_port()
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse port config '{port_config}': {e}, using default {get_default_port()}")
        return get_default_port()

def parse_host_from_config(port_config):
    """Extract host from distributed registry format (HOST:PORT)"""
    try:
        if isinstance(port_config, str) and ':' in port_config:
            host_port = port_config.split(':')
            if len(host_port) >= 2:
                return host_port[0]
        return get_default_host()
    except Exception as e:
        logger.warning(f"Failed to parse host from config '{port_config}': {e}, using {get_default_host()}")
        return get_default_host()

def convert_deque_to_list(obj):
    """Recursively convert deque objects to lists for JSON serialization"""
    if isinstance(obj, deque):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_deque_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_deque_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_deque_to_list(item) for item in obj)
    return obj

class DequeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles deque objects"""
    def default(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        return super().default(obj)

# ===== DEPENDENCY CHECKS =====
# Check for cryptography dependencies
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
    logger.info("✅ Cryptography dependencies loaded successfully")
except ImportError as e:
    CRYPTOGRAPHY_AVAILABLE = False
    logger.warning("⚠️ Cryptography dependencies not installed")

# Check for Flask availability
try:
    from flask import Flask, Response, jsonify, render_template_string, request
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from marshmallow import Schema, fields, validate, ValidationError
    FLASK_AVAILABLE = True
    logger.info("✅ Flask and security dependencies loaded successfully")
except ImportError as e:
    FLASK_AVAILABLE = False
    logger.warning("⚠️ Flask and security dependencies not installed")

# ===== ENVIRONMENT-BASED PORT REGISTRY =====
def load_port_registry_from_env():
    """Load port registry from environment variables"""
    ports = {}
    default_host = get_default_host()

    # Core AI Services - from .env
    ai_engine_host = os.getenv('AI_ENGINE_HOST', default_host)
    ai_engine_port = os.getenv('AI_ENGINE_PORT', str(get_default_port()))
    ports['AI_ENGINE_MAIN'] = f"{ai_engine_host}:{ai_engine_port}"
    ports['AI_ENGINE_API'] = f"{ai_engine_host}:{get_env_int('AI_ENGINE_PORT', get_default_port())}"

    # GitHub Integration - from .env
    if get_env_bool('GITHUB_INTEGRATION_ENABLED', False):
        github_host = os.getenv('GITHUB_WEBHOOK_HOST', ai_engine_host)
        github_port = get_env_int('GITHUB_WEBHOOK_PORT', 8080)
        ports['AI_DIRECT_WEBHOOK'] = f"{github_host}:{github_port}"
        ports['GITHUB_WEBHOOK'] = f"{github_host}:{github_port}"

    # Database Services - from .env
    postgres_host = os.getenv('POSTGRES_HOST', default_host)
    postgres_port = get_env_int('POSTGRES_PORT', 5432)
    ports['POSTGRESQL'] = f"{postgres_host}:{postgres_port}"

    # Redis configuration with proper fallback
    redis_host = os.getenv('REDIS_HOST', default_host)
    redis_port = get_env_int('REDIS_PORT', 6379)
    redis_url = os.getenv('REDIS_URL', f'redis://{redis_host}:{redis_port}')

    if redis_url.startswith('redis://'):
        redis_parts = redis_url.replace('redis://', '').split(':')
        if len(redis_parts) >= 2:
            ports['REDIS'] = f"{redis_parts[0]}:{redis_parts[1].split('/')[0]}"
        else:
            ports['REDIS'] = f"{redis_host}:{redis_port}"
    else:
        ports['REDIS'] = f"{redis_host}:{redis_port}"

    # Monitoring Services - from .env
    if get_env_bool('PROMETHEUS_ENABLED', True):
        prometheus_host = os.getenv('PROMETHEUS_HOST', default_host)
        prometheus_port = get_env_int('PROMETHEUS_PORT', 9090)
        ports['PROMETHEUS'] = f"{prometheus_host}:{prometheus_port}"

        # Prometheus metrics endpoint
        metrics_host = os.getenv('PROMETHEUS_METRICS_HOST', prometheus_host)
        metrics_port = get_env_int('PROMETHEUS_METRICS_PORT', 32287)
        ports['PROMETHEUS_METRICS'] = f"{metrics_host}:{metrics_port}"

    if get_env_bool('GRAFANA_ENABLED', True):
        grafana_host = os.getenv('GRAFANA_HOST', default_host)
        grafana_port = get_env_int('GRAFANA_PORT', 3000)
        grafana_url = os.getenv('GRAFANA_URL', f'http://{grafana_host}:{grafana_port}')

        if grafana_url.startswith('http://') or grafana_url.startswith('https://'):
            # Extract host and port from URL
            url_parts = grafana_url.replace('http://', '').replace('https://', '').split(':')
            if len(url_parts) >= 2:
                grafana_host = url_parts[0]
                grafana_port = url_parts[1].split('/')[0]
                ports['GRAFANA'] = f"{grafana_host}:{grafana_port}"
            else:
                ports['GRAFANA'] = f"{grafana_host}:{grafana_port}"
        else:
            ports['GRAFANA'] = grafana_url

    # Service mappings with environment-based defaults
    service_mappings = {
        'ALERT_PROCESSOR': ('ALERT_PROCESSOR_HOST', 'ALERT_PROCESSOR_PORT', default_host, 8052),
        'ENHANCED_DASHBOARD': ('ENHANCED_DASHBOARD_HOST', 'ENHANCED_DASHBOARD_PORT', default_host, 5000),
        'BASIC_DASHBOARD': ('BASIC_DASHBOARD_HOST', 'BASIC_DASHBOARD_PORT', default_host, 8082),
        'DOMAIN_ADAPTER': ('DOMAIN_ADAPTER_HOST', 'DOMAIN_ADAPTER_PORT', ai_engine_host, 8083),
        'LEARNING_ENGINE': ('LEARNING_ENGINE_HOST', 'LEARNING_ENGINE_PORT', ai_engine_host, 8877),
        'BACKUP_SERVICES': ('BACKUP_SERVICES_HOST', 'BACKUP_SERVICES_PORT', ai_engine_host, 8765),
        'AI_KAFKA_WEBHOOK': ('KAFKA_WEBHOOK_HOST', 'KAFKA_WEBHOOK_PORT', os.getenv('KAFKA_HOST', default_host), 8081),
        'AI_WEBSOCKET_CLIENT': ('WEBSOCKET_CLIENT_HOST', 'WEBSOCKET_CLIENT_PORT', ai_engine_host, 8091),
        'AI_FRONTEND_REALTIME': ('FRONTEND_REALTIME_HOST', 'FRONTEND_REALTIME_PORT', ai_engine_host, 8123),
        'NGINX_PORTAL': ('NGINX_HOST', 'NGINX_PORT', default_host, 80),
        'ALERTMANAGER': ('ALERTMANAGER_HOST', 'ALERTMANAGER_PORT', default_host, 9093),
        'NODE_EXPORTER': ('NODE_EXPORTER_HOST', 'NODE_EXPORTER_PORT', default_host, 9100),
        'REDIS_EXPORTER': ('REDIS_EXPORTER_HOST', 'REDIS_EXPORTER_PORT', default_host, 9121),
        'KAFKA': ('KAFKA_HOST', 'KAFKA_PORT', default_host, 9092),
        'ZOOKEEPER': ('ZOOKEEPER_HOST', 'ZOOKEEPER_PORT', default_host, 2181),
        'WIKI_QA_API': ('WIKI_QA_API_HOST', 'WIKI_QA_API_PORT', default_host, 8002),
    }

    for service_name, (host_env, port_env, default_host_val, default_port) in service_mappings.items():
        host = os.getenv(host_env, default_host_val)
        port = get_env_int(port_env, default_port)
        ports[service_name] = f"{host}:{port}"

    # MongoDB with environment-based defaults
    mongodb_host = os.getenv('MONGODB_HOST', default_host)
    mongodb_port = get_env_int('MONGODB_PORT', 27017)
    ports['MONGODB'] = f"{mongodb_host}:{mongodb_port}"

    # Log loaded configuration
    logger.info(f"✅ Loaded {len(ports)} services from environment configuration")

    # Log key services for verification
    key_services = ['AI_ENGINE_MAIN', 'POSTGRESQL', 'PROMETHEUS', 'GRAFANA']
    for service in key_services:
        if service in ports:
            logger.info(f"   {service}: {ports[service]}")

    return ports

# Load ports from environment
PORTS_RAW = load_port_registry_from_env()

# ===== PORT MANAGEMENT FALLBACK =====
class EnvironmentPortManager:
    """Environment-based port manager using .env configuration"""
    def __init__(self):
        self.ports = PORTS_RAW
        self.default_host = get_default_host()
        self.default_port = get_default_port()
        self.reload_from_env()

    def reload_from_env(self):
        """Reload configuration from environment variables"""
        self.ports = load_port_registry_from_env()
        self.default_host = get_default_host()
        self.default_port = get_default_port()

    def get_port(self, service_name):
        """Get port for service with fallback logic"""
        # Try exact match first
        if service_name in self.ports:
            return parse_port_config(self.ports[service_name])

        # Try uppercase version
        upper_name = service_name.upper()
        if upper_name in self.ports:
            return parse_port_config(self.ports[upper_name])

        # Try lowercase version
        lower_name = service_name.lower()
        if lower_name in self.ports:
            return parse_port_config(self.ports[lower_name])

        # Environment variable fallback
        env_port = get_env_int(f'{service_name.upper()}_PORT', 0)
        if env_port > 0:
            return env_port

        # Default fallback
        logger.warning(f"Port not found for {service_name}, using default {self.default_port}")
        return self.default_port

    def get_host(self, service_name):
        """Get host for service"""
        if service_name in self.ports:
            return parse_host_from_config(self.ports[service_name])

        # Environment variable fallback
        env_host = os.getenv(f'{service_name.upper()}_HOST')
        if env_host:
            return env_host

        return self.default_host

    def get_main_ai_port(self):
        return self.get_port('AI_ENGINE_MAIN')

    def get_ai_service_ports(self):
        return {
            'AI_ENGINE_MAIN': self.get_port('AI_ENGINE_MAIN'),
            'ENHANCED_DASHBOARD': self.get_port('ENHANCED_DASHBOARD'),
            'BASIC_DASHBOARD': self.get_port('BASIC_DASHBOARD'),
            'ALERT_PROCESSOR': self.get_port('ALERT_PROCESSOR'),
            'AI_DIRECT_WEBHOOK': self.get_port('AI_DIRECT_WEBHOOK'),
            'AI_KAFKA_WEBHOOK': self.get_port('AI_KAFKA_WEBHOOK'),
            'DOMAIN_ADAPTER': self.get_port('DOMAIN_ADAPTER'),
            'LEARNING_ENGINE': self.get_port('LEARNING_ENGINE'),
            'BACKUP_SERVICES': self.get_port('BACKUP_SERVICES'),
        }

# Initialize environment-based port manager
port_manager = EnvironmentPortManager()
PORT_MANAGEMENT_AVAILABLE = True
logger.info("✅ Environment-based port management system loaded")

# ===== SECURITY CONFIGURATION =====
class SecurityConfig:
    """Environment-based security configuration management"""
    def __init__(self):
        self.default_host = get_default_host()
        self.default_port = get_default_port()
        self.load_config()
        if CRYPTOGRAPHY_AVAILABLE:
            self.setup_encryption()
        else:
            self.fernet = None
            logger.warning("⚠️ Encryption not available - cryptography package not installed")
        self.setup_port_security()
        self.validate_configuration()

    def load_config(self):
        """Load configuration from environment variables"""
        # PORT CONFIGURATION - AI Engine Main Service
        self.DEFAULT_AI_ENGINE_PORT = get_env_int('AI_ENGINE_PORT', self.default_port)
        self.AI_ENGINE_PORT = self.DEFAULT_AI_ENGINE_PORT
        self.AI_ENGINE_HOST = os.getenv('AI_ENGINE_HOST', self.default_host)

        # SECURITY CONFIGURATION
        self.SECRET_KEY = os.getenv('SECRET_KEY', self._generate_secret_key())
        self.JWT_SECRET = os.getenv('JWT_SECRET_KEY', self._generate_secret_key())
        self.ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', self._generate_encryption_key())

        # Database security
        self.DATABASE_PASSWORD = os.getenv('POSTGRES_PASSWORD', self._generate_secret_key())
        postgres_host = os.getenv('POSTGRES_HOST', self.default_host)
        postgres_port = get_env_int('POSTGRES_PORT', 5432)
        postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        postgres_db = os.getenv('POSTGRES_DB', 'universal_ai_prod')

        self.DATABASE_URL = os.getenv('DATABASE_URL',
            f'postgresql://{postgres_user}:{self.DATABASE_PASSWORD}@{postgres_host}:{postgres_port}/{postgres_db}')

        # Security settings from .env
        self.MAX_REQUEST_SIZE = get_env_int('MAX_REQUEST_SIZE', 1048576)  # 1MB
        self.RATE_LIMIT = os.getenv('API_RATE_LIMIT', '100 per minute')
        self.SESSION_TIMEOUT = get_env_int('SESSION_TIMEOUT', 3600)  # 1 hour
        self.MAX_LOGIN_ATTEMPTS = get_env_int('MAX_LOGIN_ATTEMPTS', 5)

        # AI Safety settings from .env
        self.MAX_COST_THRESHOLD = get_env_float('MAX_COST_THRESHOLD', 500.0)
        self.MIN_CONFIDENCE_THRESHOLD = get_env_float('MIN_CONFIDENCE_THRESHOLD', 0.7)
        self.REQUIRE_HUMAN_APPROVAL_COST = get_env_float('REQUIRE_HUMAN_APPROVAL_COST', 1000.0)

        # Audit settings from .env
        self.AUDIT_LOG_RETENTION_DAYS = get_env_int('AUDIT_LOG_RETENTION_DAYS', 90)
        self.ENABLE_DETAILED_LOGGING = get_env_bool('ENABLE_DETAILED_LOGGING', True)

        # Network and Host security settings from .env
        default_allowed_hosts = [self.default_host, '127.0.0.1']
        if self.default_host != 'localhost':
            default_allowed_hosts.append('localhost')

        self.ALLOWED_HOSTS = get_env_list('API_CORS_ORIGINS', default_allowed_hosts)
        self.ENABLE_PORT_VALIDATION = get_env_bool('ENABLE_PORT_VALIDATION', True)
        self.BIND_ADDRESS = os.getenv('AI_ENGINE_HOST', '0.0.0.0')

        # External service URLs from .env
        redis_host = os.getenv('REDIS_HOST', self.default_host)
        redis_port = get_env_int('REDIS_PORT', 6379)
        self.REDIS_URL = os.getenv('REDIS_URL', f'redis://{redis_host}:{redis_port}/0')

        mongodb_host = os.getenv('MONGODB_HOST', self.default_host)
        mongodb_port = get_env_int('MONGODB_PORT', 27017)
        self.MONGODB_URL = os.getenv('MONGODB_URL', f'mongodb://{mongodb_host}:{mongodb_port}/ai_engine')

        # Monitoring and alerting from .env
        prometheus_host = os.getenv('PROMETHEUS_HOST', self.default_host)
        prometheus_port = get_env_int('PROMETHEUS_PORT', 9090)
        self.PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', f'http://{prometheus_host}:{prometheus_port}')

        grafana_host = os.getenv('GRAFANA_HOST', self.default_host)
        grafana_port = get_env_int('GRAFANA_PORT', 3000)
        self.GRAFANA_URL = os.getenv('GRAFANA_URL', f'http://{grafana_host}:{grafana_port}')

        self.ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')

        # SSL/TLS Configuration from .env
        self.SSL_ENABLED = get_env_bool('SSL_ENABLED', False)
        self.SSL_VERIFY_MODE = os.getenv('SSL_VERIFY_MODE', 'none')
        self.SSL_CERT_PATH = os.getenv('SSL_CERT_PATH', '')
        self.SSL_KEY_PATH = os.getenv('SSL_KEY_PATH', '')

        logger.info(f"✅ Configuration loaded from environment - AI Engine: {self.AI_ENGINE_HOST}:{self.AI_ENGINE_PORT}")

    def setup_port_security(self):
        """Setup port-based security measures"""
        self.port_manager = port_manager

        # Validate main AI Engine port
        if self.ENABLE_PORT_VALIDATION:
            try:
                if self._is_port_available(self.AI_ENGINE_PORT):
                    logger.info(f"✅ Port {self.AI_ENGINE_PORT} is available for AI Engine")
                else:
                    logger.warning(f"⚠️ Port {self.AI_ENGINE_PORT} is in use - AI Engine may conflict")

                # Validate other critical ports
                self._validate_critical_ports()
            except Exception as e:
                logger.warning(f"Port validation failed: {e}")

    def _validate_critical_ports(self):
        """Validate critical service ports"""
        critical_services = ['POSTGRESQL', 'REDIS', 'PROMETHEUS', 'GRAFANA']
        for service in critical_services:
            try:
                port = self.port_manager.get_port(service)
                host = self.port_manager.get_host(service)
                if port and not self._is_port_available(port, host):
                    logger.info(f"🔧 {service} running on {host}:{port}")
            except Exception as e:
                logger.warning(f"Failed to validate {service}: {e}")

    def _is_port_available(self, port: int, host: str = None) -> bool:
        """Check if port is available"""
        if host is None:
            host = self.default_host
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception:
            return False

    def _generate_secret_key(self) -> str:
        """Generate a secure random key"""
        return secrets.token_urlsafe(32)

    def _generate_encryption_key(self) -> str:
        """Generate encryption key"""
        if CRYPTOGRAPHY_AVAILABLE:
            key = Fernet.generate_key()
            return base64.urlsafe_b64encode(key).decode()
        else:
            return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def setup_encryption(self):
        """Setup encryption utilities"""
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("⚠️ Cryptography not available - encryption disabled")
            self.fernet = None
            return

        try:
            key = base64.urlsafe_b64decode(self.ENCRYPTION_KEY.encode())
            self.fernet = Fernet(key)
            logger.info("✅ Encryption system initialized")
        except Exception:
            logger.warning("🔧 Generating new encryption key")
            self.ENCRYPTION_KEY = self._generate_encryption_key()
            key = base64.urlsafe_b64decode(self.ENCRYPTION_KEY.encode())
            self.fernet = Fernet(key)

    def validate_configuration(self):
        """Validate the complete configuration"""
        validation_errors = []

        # Validate port configuration
        if not (1024 <= self.AI_ENGINE_PORT <= 65535):
            validation_errors.append(f"Invalid AI Engine port: {self.AI_ENGINE_PORT}")

        # Validate security settings
        if self.MAX_COST_THRESHOLD <= 0:
            validation_errors.append("MAX_COST_THRESHOLD must be positive")

        if not (0.0 <= self.MIN_CONFIDENCE_THRESHOLD <= 1.0):
            validation_errors.append("MIN_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")

        # Validate database configuration
        if not os.getenv('POSTGRES_HOST'):
            validation_errors.append("POSTGRES_HOST not configured in .env")

        if not os.getenv('POSTGRES_DB'):
            validation_errors.append("POSTGRES_DB not configured in .env")

        # Log warnings for missing dependencies but don't fail
        if not FLASK_AVAILABLE:
            logger.warning("⚠️ Flask dependencies not available - web interface disabled")

        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("⚠️ Cryptography not available - encryption disabled")

        if validation_errors:
            for error in validation_errors:
                logger.error(f"❌ Configuration Error: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")

        logger.info("✅ Configuration validation passed")

    def get_port(self, service_name: str) -> int:
        """Get port for service with security validation"""
        return self.port_manager.get_port(service_name)

    def get_host(self, service_name: str) -> str:
        """Get host for service"""
        return self.port_manager.get_host(service_name)

    def get_service_url(self, service_name: str, secure: bool = None) -> str:
        """Get full URL for a service"""
        host = self.get_host(service_name)
        port = self.get_port(service_name)

        # Auto-detect protocol based on SSL configuration or port
        if secure is None:
            secure = self.SSL_ENABLED or port == 443

        protocol = 'https' if secure else 'http'
        return f"{protocol}://{host}:{port}"

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.fernet:
            logger.warning("Encryption not available - returning plain text")
            return data
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.fernet:
            logger.warning("Decryption not available - returning as-is")
            return encrypted_data
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

# ===== GLOBAL INSTANCES AND CONFIGURATION =====
# Create global security config instance
security_config = SecurityConfig()

# Create PORTS dictionary for easy access
PORTS = PORTS_RAW.copy()

# Add all AI services
AI_SERVICE_PORTS = port_manager.get_ai_service_ports()
PORTS.update(AI_SERVICE_PORTS)

# ===== UTILITY FUNCTIONS FOR DISTRIBUTED SERVICES =====
def get_service_endpoint(service_name):
    """Get full endpoint (host:port) for distributed services"""
    try:
        if service_name.upper() in PORTS:
            return PORTS[service_name.upper()]
        else:
            host = port_manager.get_host(service_name)
            port = port_manager.get_port(service_name)
            return f"{host}:{port}"
    except Exception as e:
        logger.warning(f"Failed to get endpoint for {service_name}: {e}")
        default_host = get_default_host()
        default_port = get_default_port()
        return f"{default_host}:{default_port}"

def get_service_url(service_name, protocol="http"):
    """Get full URL for distributed services"""
    endpoint = get_service_endpoint(service_name)
    return f"{protocol}://{endpoint}"

# ===== EXPORTED FUNCTIONS =====
def get_port(service_name: str) -> int:
    """Get port for a service"""
    return security_config.get_port(service_name)

def get_host(service_name: str) -> str:
    """Get host for a service"""
    return security_config.get_host(service_name)

def is_flask_available() -> bool:
    """Check if Flask is available"""
    return FLASK_AVAILABLE

def is_encryption_available() -> bool:
    """Check if encryption is available"""
    return CRYPTOGRAPHY_AVAILABLE

def validate_system_requirements():
    """Validate system requirements and return status"""
    status = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'info': {
            'ai_engine_host': security_config.AI_ENGINE_HOST,
            'ai_engine_port': security_config.AI_ENGINE_PORT,
            'default_host': get_default_host(),
            'default_port': get_default_port(),
            'total_services': len(PORTS),
            'security_enabled': True,
            'encryption_enabled': CRYPTOGRAPHY_AVAILABLE,
            'flask_available': FLASK_AVAILABLE,
            'port_management_available': PORT_MANAGEMENT_AVAILABLE,
            'cryptography_available': CRYPTOGRAPHY_AVAILABLE,
            'ssl_enabled': security_config.SSL_ENABLED,
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'deployment_id': os.getenv('DEPLOYMENT_ID', 'universal-ai')
        }
    }

    if not FLASK_AVAILABLE:
        status['warnings'].append("Flask not available - web interface disabled")

    if not CRYPTOGRAPHY_AVAILABLE:
        status['warnings'].append("Cryptography not available - encryption disabled")

    # Validate critical environment variables
    critical_env_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in critical_env_vars if not os.getenv(var)]

    if missing_vars:
        status['warnings'].append(f"Missing critical environment variables: {missing_vars}")

    return status

def get_system_status():
    """Get comprehensive system status"""
    return {
        'configuration': 'loaded_from_environment',
        'ai_engine_host': security_config.AI_ENGINE_HOST,
        'ai_engine_port': security_config.AI_ENGINE_PORT,
        'default_host': get_default_host(),
        'default_port': get_default_port(),
        'services_configured': len(PORTS),
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'deployment_id': os.getenv('DEPLOYMENT_ID', 'universal-ai'),
        'dependencies': {
            'flask': FLASK_AVAILABLE,
            'cryptography': CRYPTOGRAPHY_AVAILABLE,
            'port_management': PORT_MANAGEMENT_AVAILABLE
        },
        'security': {
            'encryption': CRYPTOGRAPHY_AVAILABLE,
            'ssl_enabled': security_config.SSL_ENABLED,
            'rate_limiting': FLASK_AVAILABLE,
            'audit_logging': True
        },
        'database': {
            'type': os.getenv('DB_TYPE', 'postgresql'),
            'host': os.getenv('POSTGRES_HOST'),
            'port': get_env_int('POSTGRES_PORT', 5432),
            'database': os.getenv('POSTGRES_DB'),
            'ssl_mode': os.getenv('DB_SSL_MODE', 'prefer')
        },
        'monitoring': {
            'prometheus_enabled': get_env_bool('PROMETHEUS_ENABLED', True),
            'grafana_enabled': get_env_bool('GRAFANA_ENABLED', True),
            'prometheus_url': security_config.PROMETHEUS_URL,
            'grafana_url': security_config.GRAFANA_URL
        }
    }

def reload_configuration():
    """Reload configuration from environment variables"""
    global port_manager, security_config, PORTS_RAW, PORTS

    logger.info("🔄 Reloading configuration from environment variables...")

    # Reload port manager
    port_manager.reload_from_env()

    # Reload security config
    security_config.load_config()

    # Reload ports
    PORTS_RAW = load_port_registry_from_env()
    PORTS = PORTS_RAW.copy()
    AI_SERVICE_PORTS = port_manager.get_ai_service_ports()
    PORTS.update(AI_SERVICE_PORTS)

    logger.info("✅ Configuration reloaded successfully")

# ===== STARTUP MESSAGES =====
print("🔒 Starting Secure Modern Enterprise AI System - Environment-Based Configuration")
print(f"🎯 AI Engine: {security_config.AI_ENGINE_HOST}:{security_config.AI_ENGINE_PORT}")
print(f"🏠 Default Host: {get_default_host()}")
print(f"📋 Services Configured: {len(PORTS)} (from .env)")
print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
print(f"🆔 Deployment ID: {os.getenv('DEPLOYMENT_ID', 'universal-ai')}")

security_features = []
if CRYPTOGRAPHY_AVAILABLE:
    security_features.append("Encryption ✅")
else:
    security_features.append("Encryption ❌")

if FLASK_AVAILABLE:
    security_features.append("Rate Limiting ✅")
else:
    security_features.append("Rate Limiting ❌")

if security_config.SSL_ENABLED:
    security_features.append("SSL/TLS ✅")
else:
    security_features.append("SSL/TLS ❌")

security_features.append("Audit Logging ✅")
print(f"🔧 Security Features: {', '.join(security_features)}")

# Display key service endpoints
key_services = ['AI_ENGINE_MAIN', 'POSTGRESQL', 'PROMETHEUS', 'GRAFANA']
for service in key_services:
    if service in PORTS:
        print(f"🎯 {service}: {PORTS[service]}")

logger.info("✅ Environment-based settings configuration completed successfully")

# ===== FEATURE FLAGS =====
ADVANCED_MONITORING = get_env_bool('ADVANCED_MONITORING', True)
MODEL_GOVERNANCE = get_env_bool('MODEL_GOVERNANCE', True)
EVENT_STREAMING = get_env_bool('EVENT_STREAMING', True)
GITHUB_INTEGRATION_ENABLED = get_env_bool('GITHUB_INTEGRATION_ENABLED', False)

# ===== EXPORTS =====
__all__ = [
    'parse_port_config', 'parse_host_from_config', 'get_service_endpoint', 'get_service_url',
    'security_config', 'PORTS', 'get_port', 'get_host', 'is_flask_available', 'is_encryption_available',
    'validate_system_requirements', 'get_system_status', 'convert_deque_to_list', 'DequeEncoder',
    'setup_logging', 'logger', 'port_manager', 'reload_configuration', 'get_default_host', 'get_default_port',
    'ADVANCED_MONITORING', 'MODEL_GOVERNANCE', 'EVENT_STREAMING', 'GITHUB_INTEGRATION_ENABLED',
    'get_env_bool', 'get_env_int', 'get_env_float', 'get_env_list'
]
