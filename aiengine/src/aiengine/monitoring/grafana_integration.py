# core/integrations/grafana_integration.py
"""
Grafana Integration for Universal Neural System
Handles remote connections, IP restrictions, and authentication scenarios
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available for Grafana integration")

class GrafanaIntegration:
    """
    Enhanced Grafana integration with comprehensive error handling,
    remote connection support, and IP restriction detection
    """

    def __init__(self, url: str, api_key: str = None, username: str = None, password: str = None):
        """
        Initialize Grafana integration

        Args:
            url: Grafana server URL
            api_key: Optional API key for authentication
            username: Optional username for basic auth
            password: Optional password for basic auth
        """
        self.url = url.rstrip('/')

        # Get credentials with environment variable fallbacks
        self.api_key = api_key or os.getenv('GRAFANA_API_KEY', '')

        # Multiple fallback options for username
        self.username = (
            username or
            os.getenv('GRAFANA_USERNAME') or
            os.getenv('GRAFANA_USER') or
            os.getenv('GRAFANA_ADMIN_USER') or
            'admin'
        )

        # Multiple fallback options for password
        self.password = (
            password or
            os.getenv('GRAFANA_PASSWORD') or
            os.getenv('GRAFANA_PASS') or
            os.getenv('GRAFANA_ADMIN_PASSWORD') or
            'admin'
        )

        # Connection settings
        self.timeout = int(os.getenv('GRAFANA_TIMEOUT', '30'))
        self.connect_timeout = int(os.getenv('GRAFANA_CONNECT_TIMEOUT', '15'))
        self.read_timeout = int(os.getenv('GRAFANA_READ_TIMEOUT', '30'))
        self.retry_attempts = int(os.getenv('GRAFANA_RETRY_ATTEMPTS', '3'))
        self.skip_tls_verify = os.getenv('GRAFANA_SKIP_TLS_VERIFY', 'false').lower() == 'true'

        # State tracking
        self.session = None
        self.connection_tested = False
        self.connection_status = "unknown"
        self.last_error = None
        self.last_test_time = None
        self.organization_info = None
        self.user_info = None

        # Initialize session if requests is available
        if REQUESTS_AVAILABLE:
            self._setup_session()
        else:
            logger.warning("requests library not available - Grafana integration disabled")

        # Log initialization (without sensitive data)
        logger.info(f"🎯 Grafana integration initialized")
        logger.info(f"   URL: {self.url}")
        logger.info(f"   Username: {self.username}")
        logger.info(f"   Password: {'configured' if self.password and self.password != 'admin' else 'default/not configured'}")
        logger.info(f"   API Key: {'configured' if self.api_key else 'not configured'}")
        logger.info(f"   Timeout: {self.timeout}s (connect: {self.connect_timeout}s, read: {self.read_timeout}s)")

    def _setup_session(self):
        """Setup requests session optimized for remote connections"""
        self.session = requests.Session()

        # Configure SSL/TLS verification
        if self.skip_tls_verify:
            self.session.verify = False
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.debug("TLS verification disabled for remote connection")

        # Setup retry strategy for remote connections
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            connect=self.retry_attempts,
            read=self.retry_attempts,
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set common headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Universal-Neural-System/1.0',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        })

        logger.debug(f"Session configured for remote Grafana:")
        logger.debug(f"  Connect timeout: {self.connect_timeout}s")
        logger.debug(f"  Read timeout: {self.read_timeout}s")
        logger.debug(f"  Retry attempts: {self.retry_attempts}")

    def _get_auth_method(self) -> str:
        """Determine which authentication method will be used"""
        if self.api_key:
            return f"API Key ({self.api_key[:10]}...)"
        elif self.username and self.password and self.password != 'admin':
            return f"Basic Auth ({self.username})"
        elif self.username and self.password:
            return f"Basic Auth with default password ({self.username})"
        else:
            return "No authentication"

    def test_connection(self) -> bool:
        """
        Enhanced connection test with comprehensive error handling

        Returns:
            bool: True if connection successful, False otherwise
        """
        if not REQUESTS_AVAILABLE or not self.session:
            logger.warning("Cannot test Grafana connection - requests not available")
            self.connection_status = "requests_unavailable"
            self.connection_tested = True
            return False

        self.last_test_time = time.time()
        logger.info(f"🔍 Testing Grafana connection to {self.url}")
        logger.info(f"   Authentication method: {self._get_auth_method()}")

        # Test 1: Basic connectivity and service detection
        try:
            response = self.session.get(
                f"{self.url}/api/health",
                timeout=(self.connect_timeout, self.read_timeout)
            )

            logger.info(f"   Health endpoint response: {response.status_code}")

            if response.status_code == 403:
                return self._handle_403_response()
            elif response.status_code == 200:
                logger.info("   ✅ Health endpoint accessible")
                return self._test_authentication()
            elif response.status_code == 401:
                logger.info("   🔐 Health endpoint requires authentication")
                return self._test_authentication()
            else:
                logger.info(f"   ⚠️ Unexpected health response: {response.status_code}")
                return self._test_authentication()

        except requests.exceptions.ConnectionError as e:
            logger.error(f"   ❌ Connection failed: {str(e)[:100]}")
            self.connection_status = "connection_failed"
            self.last_error = f"Connection error: {str(e)[:100]}"
            self.connection_tested = True
            return False

        except requests.exceptions.Timeout as e:
            logger.warning(f"   ⏰ Connection timeout after {self.timeout}s: {e}")
            self.connection_status = "timeout"
            self.last_error = f"Timeout after {self.timeout}s"
            self.connection_tested = True
            return False

        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {str(e)[:100]}")
            self.connection_status = "error"
            self.last_error = str(e)[:100]
            self.connection_tested = True
            return False

    def _handle_403_response(self) -> bool:
        """Handle 403 Forbidden responses - typically IP restrictions"""
        logger.info("   🚫 Received 403 Forbidden")

        # Test if this is a blanket IP restriction
        try:
            login_response = self.session.get(
                f"{self.url}/login",
                timeout=(self.connect_timeout, self.read_timeout)
            )

            if login_response.status_code == 403:
                # Even login page is forbidden - this is IP restriction
                logger.info("   🚫 Login page also forbidden - IP restriction detected")
                logger.info(f"   ✅ Grafana service confirmed running at {self.url}")
                logger.info("   💡 Your IP address needs to be whitelisted")

                self.connection_status = "ip_restricted"
                self.last_error = "IP address not in whitelist"
                self.connection_tested = True

                # Try to get more info from the response
                if hasattr(login_response, 'headers'):
                    server = login_response.headers.get('Server', 'Unknown')
                    logger.debug(f"   Server: {server}")

                return False
            else:
                # Login page accessible, try authentication
                logger.info("   🔐 Login page accessible, testing authentication")
                return self._test_authentication()

        except Exception as e:
            logger.debug(f"   Error testing login page: {e}")
            self.connection_status = "ip_restricted"
            self.last_error = "Likely IP restriction"
            self.connection_tested = True
            return False

    def _test_authentication(self) -> bool:
        """Test different authentication methods"""
        auth_success = False

        # Method 1: Try API key authentication
        if self.api_key and not auth_success:
            logger.info("   🔑 Testing API key authentication...")
            try:
                # Temporarily set API key header
                original_auth = self.session.headers.get('Authorization')
                self.session.headers['Authorization'] = f'Bearer {self.api_key}'

                response = self.session.get(
                    f"{self.url}/api/org",
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                if response.status_code == 200:
                    logger.info("   ✅ API key authentication successful")
                    self.organization_info = response.json()
                    logger.info(f"   Organization: {self.organization_info.get('name', 'Unknown')}")
                    auth_success = True
                    self.connection_status = "connected"
                elif response.status_code == 401:
                    logger.warning("   ❌ API key authentication failed")
                    # Remove the invalid API key header
                    if original_auth:
                        self.session.headers['Authorization'] = original_auth
                    else:
                        self.session.headers.pop('Authorization', None)
                elif response.status_code == 403:
                    logger.warning("   🚫 API key forbidden - may be IP restricted")
                else:
                    logger.info(f"   API key test returned: {response.status_code}")

            except Exception as e:
                logger.debug(f"   API key test error: {e}")

        # Method 2: Try basic authentication
        if not auth_success and self.username and self.password:
            logger.info(f"   🔐 Testing basic authentication ({self.username})...")
            try:
                # Set basic auth
                self.session.auth = (self.username, self.password)

                response = self.session.get(
                    f"{self.url}/api/org",
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                if response.status_code == 200:
                    logger.info("   ✅ Basic authentication successful")
                    self.organization_info = response.json()
                    logger.info(f"   Organization: {self.organization_info.get('name', 'Unknown')}")

                    # Try to get user info
                    try:
                        user_response = self.session.get(f"{self.url}/api/user", timeout=10)
                        if user_response.status_code == 200:
                            self.user_info = user_response.json()
                            logger.info(f"   User: {self.user_info.get('login', 'Unknown')}")
                    except:
                        pass

                    auth_success = True
                    self.connection_status = "connected"
                elif response.status_code == 401:
                    logger.warning("   ❌ Basic authentication failed - check credentials")
                    self.session.auth = None
                elif response.status_code == 403:
                    logger.warning("   🚫 Basic auth forbidden - may be IP restricted")
                else:
                    logger.info(f"   Basic auth test returned: {response.status_code}")

            except Exception as e:
                logger.debug(f"   Basic auth test error: {e}")

        # Method 3: Try without authentication (public access)
        if not auth_success:
            logger.info("   🌐 Testing public access...")
            try:
                # Remove any authentication
                self.session.auth = None
                self.session.headers.pop('Authorization', None)

                response = self.session.get(
                    f"{self.url}/api/org",
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                if response.status_code == 200:
                    logger.info("   ✅ Public access available")
                    self.organization_info = response.json()
                    auth_success = True
                    self.connection_status = "connected"
                else:
                    logger.debug(f"   Public access denied: {response.status_code}")

            except Exception as e:
                logger.debug(f"   Public access test error: {e}")

        # Final status update
        if auth_success:
            self.connection_tested = True
            logger.info("✅ Grafana connection and authentication successful")
            return True
        else:
            self.connection_status = "auth_failed"
            self.last_error = "All authentication methods failed"
            self.connection_tested = True
            logger.warning("❌ Grafana authentication failed")
            logger.info("   💡 Verify credentials or check IP restrictions")
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information"""
        return {
            'url': self.url,
            'status': self.connection_status,
            'tested': self.connection_tested,
            'last_test_time': self.last_test_time,
            'last_error': self.last_error,
            'auth_method': self._get_auth_method(),
            'username': self.username,
            'has_password': bool(self.password and self.password != 'admin'),
            'has_api_key': bool(self.api_key),
            'timeout_settings': {
                'connect': self.connect_timeout,
                'read': self.read_timeout,
                'total': self.timeout
            },
            'organization_info': self.organization_info,
            'user_info': self.user_info,
            'requests_available': REQUESTS_AVAILABLE
        }

    def get_status_message(self) -> str:
        """Get human-readable status message"""
        status_messages = {
            "connected": "✅ Connected and authenticated",
            "ip_restricted": "🚫 Service detected but IP access restricted",
            "auth_failed": "🔐 Service reachable but authentication failed",
            "connection_failed": "❌ Cannot reach service",
            "timeout": "⏰ Connection timeout",
            "requests_unavailable": "📦 HTTP client not available",
            "error": "❌ Connection error",
            "unknown": "❓ Status unknown"
        }
        return status_messages.get(self.connection_status, f"Status: {self.connection_status}")

    def is_service_detected(self) -> bool:
        """Check if Grafana service was detected (even if not fully accessible)"""
        detected_statuses = ["connected", "ip_restricted", "auth_failed"]
        return self.connection_status in detected_statuses

    def is_accessible(self) -> bool:
        """Check if Grafana is fully accessible"""
        return self.connection_status == "connected"

    def _can_make_requests(self) -> bool:
        """Check if we can make authenticated API requests"""
        if not REQUESTS_AVAILABLE or not self.session:
            return False

        if not self.connection_tested:
            self.test_connection()

        return self.connection_status == "connected"

    def get_datasources(self) -> List[Dict]:
        """Get list of existing datasources"""
        if not self._can_make_requests():
            logger.debug("Cannot get datasources - not authenticated")
            return []

        try:
            response = self.session.get(
                f"{self.url}/api/datasources",
                timeout=(self.connect_timeout, self.read_timeout)
            )

            if response.status_code == 200:
                datasources = response.json()
                logger.debug(f"Retrieved {len(datasources)} datasources")
                return datasources
            else:
                logger.debug(f"Failed to get datasources: {response.status_code}")
                return []

        except Exception as e:
            logger.debug(f"Failed to get datasources: {e}")
            return []

    def create_datasource(self, name: str, prometheus_url: str) -> bool:
        """Create Prometheus datasource in Grafana"""
        if not self._can_make_requests():
            logger.warning("Cannot create datasource - not authenticated to Grafana")
            return False

        try:
            # Check if datasource already exists
            existing_datasources = self.get_datasources()
            for ds in existing_datasources:
                if ds.get('name') == name:
                    logger.info(f"✅ Grafana datasource '{name}' already exists")
                    return True

            # Create new datasource
            datasource_config = {
                "name": name,
                "type": "prometheus",
                "url": prometheus_url,
                "access": "proxy",
                "isDefault": True,
                "basicAuth": False,
                "withCredentials": False,
                "jsonData": {
                    "httpMethod": "POST",
                    "queryTimeout": "60s",
                    "timeInterval": "15s",
                    "manageAlerts": True,
                    "alertmanagerUid": ""
                }
            }

            response = self.session.post(
                f"{self.url}/api/datasources",
                json=datasource_config,
                timeout=(self.connect_timeout, self.read_timeout)
            )

            if response.status_code in [200, 409]:  # 409 = already exists
                logger.info(f"✅ Grafana datasource '{name}' configured successfully")
                return True
            else:
                logger.warning(f"Failed to create datasource '{name}': {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False

        except Exception as e:
            logger.warning(f"Failed to create Grafana datasource '{name}': {e}")
            return False

    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> bool:
        """Create a dashboard in Grafana"""
        if not self._can_make_requests():
            logger.warning("Cannot create dashboard - not authenticated to Grafana")
            return False

        try:
            dashboard_payload = {
                "dashboard": dashboard_config,
                "overwrite": True,
                "message": "Created by Universal Neural System"
            }

            response = self.session.post(
                f"{self.url}/api/dashboards/db",
                json=dashboard_payload,
                timeout=(self.connect_timeout, self.read_timeout)
            )

            if response.status_code == 200:
                result = response.json()
                dashboard_url = f"{self.url}{result.get('url', '')}"
                logger.info(f"✅ Grafana dashboard created: {dashboard_url}")
                return True
            else:
                logger.warning(f"Failed to create dashboard: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False

        except Exception as e:
            logger.warning(f"Failed to create Grafana dashboard: {e}")
            return False

    def create_universal_neural_dashboard(self, datasource_name: str = "Universal-Prometheus") -> bool:
        """Create a comprehensive dashboard for Universal Neural System metrics"""
        dashboard_config = {
            "id": None,
            "uid": "universal-neural-system",
            "title": "Universal Neural System Dashboard",
            "description": "Comprehensive monitoring for Intel Universal Neural System",
            "tags": ["universal-neural", "ai", "monitoring", "intel"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {
                "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
            },
            "panels": [
                {
                    "id": 1,
                    "title": "System Overview",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "universal_neural_tasks_total",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 80}
                                ]
                            },
                            "unit": "short"
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Average Confidence Score",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": "universal_neural_confidence_avg",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 0.5},
                                    {"color": "green", "value": 0.8}
                                ]
                            },
                            "unit": "percentunit"
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
                },
                {
                    "id": 3,
                    "title": "System Health Score",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": "universal_neural_system_health",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 0.6},
                                    {"color": "green", "value": 0.8}
                                ]
                            },
                            "unit": "percentunit"
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
                },
                {
                    "id": 4,
                    "title": "Active Tasks",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "universal_neural_active_tasks",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 50},
                                    {"color": "red", "value": 100}
                                ]
                            },
                            "unit": "short"
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
                },
                {
                    "id": 5,
                    "title": "Domain Expertise Levels",
                    "type": "bargauge",
                    "targets": [
                        {
                            "expr": "universal_neural_domain_expertise",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "color": {"mode": "continuous-GrYlRd"},
                            "unit": "percentunit"
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                },
                {
                    "id": 6,
                    "title": "Execution Time Trend",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "universal_neural_execution_time_avg",
                            "refId": "A",
                            "datasource": {"type": "prometheus", "uid": datasource_name}
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "s"
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                }
            ]
        }

        return self.create_dashboard(dashboard_config)

    def setup_universal_neural_monitoring(self, prometheus_url: str) -> bool:
        """Setup complete monitoring for Universal Neural System"""
        if not self._can_make_requests():
            logger.info("🎯 Grafana detected but not accessible for automatic setup")
            logger.info(f"   Dashboard URL: {self.url}")
            logger.info("   Manual configuration required")
            return False

        try:
            logger.info("🔧 Setting up Universal Neural System monitoring in Grafana...")

            # Step 1: Create Prometheus datasource
            datasource_created = self.create_datasource("Universal-Prometheus", prometheus_url)

            # Step 2: Create Universal Neural dashboard
            dashboard_created = self.create_universal_neural_dashboard("Universal-Prometheus")

            success = datasource_created and dashboard_created

            if success:
                logger.info("✅ Universal Neural System monitoring setup complete in Grafana")
                logger.info(f"   Dashboard: {self.url}/d/universal-neural-system")
                logger.info(f"   Datasource: Universal-Prometheus -> {prometheus_url}")
            else:
                logger.warning("⚠️ Partial Grafana setup - some components may need manual configuration")

            return success

        except Exception as e:
            logger.warning(f"Grafana monitoring setup failed: {e}")
            logger.info("🎯 Grafana is available for manual configuration")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get Grafana system information"""
        if not self._can_make_requests():
            return {
                "status": self.connection_status,
                "url": self.url,
                "message": "Authentication required for system info",
                "accessible": False
            }

        try:
            # Try to get basic system info
            info = {
                "status": self.connection_status,
                "url": self.url,
                "accessible": True,
                "organization": self.organization_info,
                "user": self.user_info
            }

            # Try to get additional system info if we have admin access
            try:
                response = self.session.get(f"{self.url}/api/admin/settings", timeout=10)
                if response.status_code == 200:
                    info["admin_settings"] = response.json()
                else:
                    info["admin_access"] = False
            except:
                info["admin_access"] = False

            return info

        except Exception as e:
            return {
                "status": "error",
                "url": self.url,
                "error": str(e),
                "accessible": False
            }

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test various API endpoints to determine available functionality"""
        if not self._can_make_requests():
            return {"error": "Not authenticated"}

        endpoints_to_test = [
            ("/api/org", "Organization info"),
            ("/api/user", "User info"),
            ("/api/datasources", "Datasources"),
            ("/api/dashboards/home", "Home dashboard"),
            ("/api/admin/settings", "Admin settings"),
            ("/api/health", "Health check")
        ]

        results = {}

        for endpoint, description in endpoints_to_test:
            try:
                response = self.session.get(f"{self.url}{endpoint}", timeout=10)
                results[endpoint] = {
                    "status_code": response.status_code,
                    "description": description,
                    "accessible": response.status_code == 200
                }
            except Exception as e:
                results[endpoint] = {
                    "status_code": "error",
                    "description": description,
                    "accessible": False,
                    "error": str(e)
                }

        return results

    def __str__(self) -> str:
        """String representation of the Grafana integration"""
        return f"GrafanaIntegration(url={self.url}, status={self.connection_status})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"GrafanaIntegration(url='{self.url}', "
                f"status='{self.connection_status}', "
                f"auth_method='{self._get_auth_method()}', "
                f"tested={self.connection_tested})")
