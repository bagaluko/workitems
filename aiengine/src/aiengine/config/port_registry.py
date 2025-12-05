# config/port_registry.py
import os
import json
import yaml
import logging
from typing import Dict, Union, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration with all details"""
    name: str
    host: str
    port: int
    protocol: str = "http"
    path: str = ""
    description: str = ""
    environment: str = "production"

    @property
    def url(self) -> str:
        """Get full service URL"""
        base_url = f"{self.protocol}://{self.host}:{self.port}"
        return f"{base_url}{self.path}" if self.path else base_url

    @property
    def address(self) -> str:
        """Get host:port address"""
        return f"{self.host}:{self.port}"

class PortRegistry:
    def __init__(self, registry_file="/etc/port-registry.conf"):
        self.registry_file = registry_file
        self.services = {}
        self.global_config = {}
        self.load_registry()

    def load_registry(self):
        """Load port configuration from registry file"""
        try:
            if not os.path.exists(self.registry_file):
                logger.warning(f"Port registry file not found: {self.registry_file}")
                self._load_fallback_config()
                return

            # Force YAML loading since our .conf file contains YAML
            logger.info("🔍 Loading as YAML format")
            self._load_yaml_config()

            logger.info(f"✅ Loaded {len(self.services)} service configurations from {self.registry_file}")

        except Exception as e:
            logger.error(f"Failed to load port registry: {e}")
            self._load_fallback_config()

    def _load_yaml_config(self):
        """Load YAML configuration file"""
        with open(self.registry_file, 'r') as f:
            config = yaml.safe_load(f)

        # Load global configuration
        self.global_config = config.get('global', {})

        # Load services
        services_config = config.get('services', {})
        for service_name, service_data in services_config.items():
            if isinstance(service_data, dict):
                self.services[service_name] = ServiceConfig(
                    name=service_name,
                    host=service_data.get('host', self.global_config.get('default_host', 'localhost')),
                    port=service_data.get('port', 8000),
                    protocol=service_data.get('protocol', self.global_config.get('default_protocol', 'http')),
                    path=service_data.get('path', ''),
                    description=service_data.get('description', ''),
                    environment=service_data.get('environment', self.global_config.get('environment', 'production'))
                )
            else:
                # Simple format: service_name: "host:port"
                host, port = self._parse_address(str(service_data))
                self.services[service_name] = ServiceConfig(
                    name=service_name,
                    host=host,
                    port=port
                )

    def _load_json_config(self):
        """Load JSON configuration file"""
        with open(self.registry_file, 'r') as f:
            config = json.load(f)

        self.global_config = config.get('global', {})
        services_config = config.get('services', {})

        for service_name, service_data in services_config.items():
            if isinstance(service_data, dict):
                self.services[service_name] = ServiceConfig(
                    name=service_name,
                    host=service_data.get('host', 'localhost'),
                    port=service_data.get('port', 8000),
                    protocol=service_data.get('protocol', 'http'),
                    path=service_data.get('path', ''),
                    description=service_data.get('description', ''),
                    environment=service_data.get('environment', 'production')
                )

    def _load_simple_config(self):
        """Load simple key=value configuration file"""
        with open(self.registry_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Handle global configuration
                if line.startswith('[global]'):
                    continue
                elif '=' in line and not line.startswith('['):
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Check if it's a global config
                        if key.startswith('GLOBAL_'):
                            config_key = key.replace('GLOBAL_', '').lower()
                            self.global_config[config_key] = value
                        else:
                            # Parse service configuration
                            host, port = self._parse_address(value)
                            self.services[key] = ServiceConfig(
                                name=key,
                                host=host,
                                port=port
                            )

                    except Exception as e:
                        logger.warning(f"Invalid config at line {line_num}: {line} - {e}")

    def _parse_address(self, address: str) -> Tuple[str, int]:
        """Parse address string to host and port"""
        try:
            if ':' in address:
                host, port_str = address.rsplit(':', 1)
                return host, int(port_str)
            else:
                return self.global_config.get('default_host', 'localhost'), int(address)
        except (ValueError, IndexError):
            logger.warning(f"Invalid address format: {address}")
            return 'localhost', 8000

    def _load_fallback_config(self):
        """Load fallback configuration if registry file is unavailable"""
        logger.info("Loading fallback service configuration")

        # Fallback global config
        self.global_config = {
            'environment': 'development',
            'default_host': 'localhost',
            'default_protocol': 'http'
        }

        # Fallback services
        fallback_services = {
            "AI_ENGINE_MAIN": {"host": "10.223.169.25", "port": 8000},
            "AI_DIRECT_WEBHOOK": {"host": "10.223.210.70", "port": 8080},
            "AI_KAFKA_WEBHOOK": {"host": "10.223.210.70", "port": 8081},
            "AI_WEBSOCKET_CLIENT": {"host": "10.223.210.70", "port": 8091},
            "ALERT_PROCESSOR": {"host": "10.223.210.70", "port": 8052},
            "ENHANCED_DASHBOARD": {"host": "10.223.210.70", "port": 5000},
            "AI_FRONTEND_REALTIME": {"host": "10.223.210.70", "port": 8123},
            "NGINX_PORTAL": {"host": "10.223.210.70", "port": 80},
            "PROMETHEUS": {"host": "10.223.251.30", "port": 9090},
            "ALERTMANAGER": {"host": "10.223.251.30", "port": 9093},
            "NODE_EXPORTER": {"host": "10.223.251.30", "port": 9100},
            "PROMETHEUS_METRICS": {"host": "10.223.251.30", "port": 32287},
            "GRAFANA": {"host": "10.106.208.44", "port": 32000},
            "KAFKA": {"host": "10.106.208.44", "port": 30092},
            "MONGODB": {"host": "localhost", "port": 27017},
            "REDIS": {"host": "localhost", "port": 6379},
            "ZOOKEEPER": {"host": "10.106.208.44", "port": 2181},
            "BASIC_DASHBOARD": {"host": "localhost", "port": 8082},
            "DOMAIN_ADAPTER": {"host": "localhost", "port": 8083},
            "LEARNING_ENGINE": {"host": "localhost", "port": 8877},
            "BACKUP_SERVICES": {"host": "localhost", "port": 8765},
            "AVAILABLE_PORT": {"host": "localhost", "port": 8899}
        }

        for service_name, config in fallback_services.items():
            self.services[service_name] = ServiceConfig(
                name=service_name,
                host=config["host"],
                port=config["port"]
            )

    # Service access methods
    def get_service(self, service_name: str) -> Optional[ServiceConfig]:
        """Get complete service configuration"""
        return self.services.get(service_name)

    def get_port(self, service_name: str) -> int:
        """Get port number for a service"""
        service = self.services.get(service_name)
        if service:
            return service.port
        logger.warning(f"Service {service_name} not found in registry")
        return 8000

    def get_host(self, service_name: str) -> str:
        """Get host for a service"""
        service = self.services.get(service_name)
        if service:
            return service.host
        logger.warning(f"Service {service_name} not found in registry")
        return 'localhost'

    def get_host_port(self, service_name: str) -> Tuple[str, int]:
        """Get host and port for a service"""
        service = self.services.get(service_name)
        if service:
            return service.host, service.port
        logger.warning(f"Service {service_name} not found in registry")
        return 'localhost', 8000

    def get_url(self, service_name: str, path: str = "") -> str:
        """Get full URL for a service"""
        service = self.services.get(service_name)
        if service:
            base_url = service.url
            return f"{base_url}{path}" if path else base_url
        logger.warning(f"Service {service_name} not found in registry")
        return f"http://localhost:8000{path}"

    def get_address(self, service_name: str) -> str:
        """Get host:port address for a service"""
        service = self.services.get(service_name)
        if service:
            return service.address
        return "localhost:8000"

    def list_services(self) -> Dict[str, ServiceConfig]:
        """Get all services"""
        return self.services.copy()

    def get_services_by_host(self, host: str) -> Dict[str, ServiceConfig]:
        """Get all services running on a specific host"""
        return {name: service for name, service in self.services.items() if service.host == host}

    def get_environment(self) -> str:
        """Get current environment"""
        return self.global_config.get('environment', 'production')

    def reload(self):
        """Reload configuration from file"""
        self.services.clear()
        self.global_config.clear()
        self.load_registry()

# Global port registry instance
port_registry = PortRegistry()
