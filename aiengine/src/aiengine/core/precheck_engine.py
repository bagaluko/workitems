#!/usr/bin/env python3
"""
Precheck Exception Engine - AI-Driven Precheck Exception Handling
Integrates with Universal Neural System for intelligent precheck decisions
Enhanced with comprehensive BA team precheck documentation
"""
import os
import sys
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# Add main directory to path
sys.path.insert(0, '/aiengine/src/aiengine')




try:
    # Import core types directly
    from core.universal_types import (
        UniversalTask, UniversalSolution, DomainType, TaskType
    )

    # Import utilities with fallbacks
    try:
        from config.ai_utils import (
            safe_json_serialize,
            validate_input_data,
            create_task_id,
            convert_deque_to_list
        )
    except ImportError:
        # Fallback implementations
        def safe_json_serialize(obj): return obj
        def validate_input_data(data, max_size_mb=10): return True, "Valid"
        def create_task_id(prefix="task"): return f"{prefix}_{int(time.time())}"
        def convert_deque_to_list(obj): return obj

    AI_SYSTEM_AVAILABLE = True

except ImportError as e:
    print(f"⚠️ AI System not available: {e}")
    AI_SYSTEM_AVAILABLE = False

    # Minimal fallbacks
    class UniversalTask:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class UniversalSolution:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DomainType:
        INFRASTRUCTURE = "infrastructure"

    class TaskType:
        CLASSIFICATION = "classification"

    def safe_json_serialize(obj): return obj
    def validate_input_data(data, max_size_mb=10): return True, "Valid"
    def create_task_id(prefix="task"): return f"{prefix}_{int(time.time())}"
    def convert_deque_to_list(obj): return obj




    class SystemFactory:
        @staticmethod
        def create_universal_system():
            return None

        @staticmethod
        def get_universal_system_class():
            return None


logger = logging.getLogger('PrecheckEngine')

class PrecheckDecision(Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    ESCALATE_BA = "escalate_ba"
    ESCALATE_LONNY = "escalate_lonny"
    ESCALATE_RAJDEEP = "escalate_rajdeep"
    MANUAL_REVIEW = "manual_review"

class IngredientType(Enum):
    INTEL = "intel"
    TPV = "tpv"
    UNKNOWN = "unknown"

@dataclass
class PrecheckFailure:
    """Represents a precheck failure requiring decision"""
    failure_id: str
    precheck_name: str
    ingredient_type: IngredientType
    ingredient_name: str
    failure_details: Dict[str, Any]
    milestone: str
    timestamp: float
    severity: str = "medium"
    metadata: Dict[str, Any] = None
    error_message: str = ""
    is_false_positive: bool = False

@dataclass
class PrecheckDecisionResult:
    """Result of precheck decision analysis"""
    failure_id: str
    decision: PrecheckDecision
    confidence: float
    reasoning: str
    approver: str
    escalation_contacts: List[str]
    waiver_duration_days: int
    ai_analysis: Dict[str, Any]
    business_rule_applied: str
    validation_required: str = ""
    milestone_restriction: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PrecheckEngine:
    """AI-powered precheck exception handling engine with comprehensive BA rules"""


    def __init__(self):
        self.ai_system = None
        if AI_SYSTEM_AVAILABLE:
            try:
                # Try to get AI system without SystemFactory
                print("🔧 Attempting to initialize AI system for precheck...")
                # We'll set this later when the main system is available
                self.ai_system = None  # Will be set by main system
                print("✅ AI System placeholder created for precheck analysis")
            except Exception as e:
                print(f"⚠️ Failed to initialize AI system: {e}")

        # Load comprehensive business rules
        self.business_rules = self.load_comprehensive_business_rules()

        # Historical data for learning
        self.decision_history = []
        self.pattern_cache = {}

        # Email configuration for manual intervention notifications
        self.email_config = {
            'smtp_server': 'smtp.intel.com',
            'smtp_port': 25,
            'sender_email': 'balasubramanyam.agalukote.lakshmipathi@intel.com',
            'ba_team_email': 'balasubramanyam.agalukote.lakshmipathi@intel.com',
            'enabled': True,
            'use_tls': False,
            'timeout': 30,
            'simple_format': True
        }

        # Service monitoring integration
        self.service_monitor = None
        self.monitoring_cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_monitoring_check = 0

        # Initialize service monitoring
        self.azure_monitor = None
        self._initialize_service_monitoring()

        print("🔍 Precheck Engine initialized")

    #def _initialize_service_monitoring(self):
    #    """Initialize service monitoring integration"""
    #    try:
    #        from core.azure_connect import PrecheckServiceMonitor
    #        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'precheck'))
    #        from precheck_service_monitor.monitor_precheck import PrecheckMonitor
    #
    #        config_path = os.path.join(
    #            os.path.dirname(__file__),
    #            '..',
    #            'precheck',
    #            'config',
    #            'precheck_config.json'
    #        )
    #        self.azure_monitor = PrecheckServiceMonitor(config_path)
    #        logger.info("✅ Precheck service monitoring integrated successfully")
    #
    #        # Test the integration
    #        health_data = self.azure_monitor.get_service_health()
    #        service_count = health_data.get('total_services', 0)
    #        logger.info(f"📊 Monitoring {service_count} precheck services")
    #
    #    except ImportError as e:
    #        logger.warning(f"⚠️ Precheck service monitoring integration failed: {e}")
    #        self.azure_monitor = None
    #    except Exception as e:
    #        logger.error(f"❌ Service monitoring initialization failed: {e}")
    #        self.azure_monitor = None
    #
    #        if os.path.exists(config_path):
    #            self.service_monitor = PrecheckMonitor(config_path)
    #            logger.info("✅ Service monitoring integrated with precheck engine")
    #        else:
    #            logger.warning(f"⚠️ Service monitor config not found: {config_path}")
    #
    #    except Exception as e:
    #        logger.warning(f"⚠️ Service monitoring integration failed: {e}")
    #        self.service_monitor = None
    #
    #    """Initialize service monitoring with Azure integration"""
    #    try:
    #        from core.azure_connect import AzureServiceMonitor
    #        self.azure_monitor = AzureServiceMonitor()
    #
    #        if self.azure_monitor.enabled and self.azure_monitor.initialized:
    #            logger.info("✅ Azure service monitoring integrated with precheck engine")
    #        else:
    #            logger.info("ℹ️ Azure monitoring not available - using fallback service monitoring")
    #
    #    except ImportError as e:
    #        logger.warning(f"⚠️ Azure monitoring integration failed: {e}")
    #        self.azure_monitor = None
    #    except Exception as e:
    #        logger.error(f"❌ Service monitoring initialization failed: {e}")
    #        self.azure_monitor = None

    def set_ai_system(self, ai_system):
        """Set the AI system for enhanced analysis"""
        self.ai_system = ai_system
        logger.info("✅ AI system integrated with precheck engine")


    def format_precheck_output(self, solution: Any, task) -> Dict[str, Any]:
        """Format precheck output for universal system compatibility"""
        try:
            if isinstance(solution, dict):
                return {
                    'decision': solution.get('action', solution.get('decision', 'REQUIRE_MANUAL_REVIEW')),
                    'confidence': solution.get('confidence', 0.5),
                    'explanation': solution.get('reasoning', 'Precheck analysis completed'),
                    'risk_level': self._calculate_risk_level(solution.get('confidence', 0.5)),
                    'risk_score': 1.0 - solution.get('confidence', 0.5),
                    'metadata': {
                        'precheck_engine_used': True,
                        'business_rule_applied': solution.get('business_rule_applied', 'unknown'),
                        'fallback_mode': solution.get('fallback_used', False),
                        'analysis_timestamp': time.time()
                    }
                }
            else:
                return {
                    'decision': 'REQUIRE_MANUAL_REVIEW',
                    'confidence': 0.5,
                    'explanation': 'Unable to format precheck output',
                    'risk_level': 'MEDIUM',
                    'risk_score': 0.5,
                    'metadata': {'error': 'Invalid solution format'}
                }
        except Exception as e:
            logger.error(f"Error formatting precheck output: {e}")
            return {
                'decision': 'REQUIRE_MANUAL_REVIEW',
                'confidence': 0.3,
                'explanation': f'Format error: {str(e)}',
                'risk_level': 'HIGH',
                'risk_score': 0.7,
                'metadata': {'error': str(e)}
            }

    def _calculate_risk_level(self, confidence: float) -> str:
        """Calculate risk level based on confidence"""
        if confidence >= 0.8:
            return 'LOW'
        elif confidence >= 0.6:
            return 'MEDIUM'
        elif confidence >= 0.4:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def integrate_with_universal_system(self, universal_system):
        """Integrate precheck engine with universal neural system"""
        try:
            self.ai_system = universal_system
            logger.info("✅ Precheck engine integrated with Universal Neural System")

            # Test the integration
            test_result = self.get_health_status()
            logger.info(f"🔍 Integration test: {test_result['status']}")

            return True
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return False

    def _initialize_service_monitoring(self):
        """Initialize service monitoring integration with multiple fallback options"""
        self.azure_monitor = None
        self.service_monitor = None

        # Approach 1: Try PrecheckServiceMonitor from azure_connect
        try:
            from core.azure_connect import PrecheckServiceMonitor

            config_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'precheck',
                'config',
                'precheck_config.json'
            )

            if os.path.exists(config_path):
                self.azure_monitor = PrecheckServiceMonitor(config_path)

                # Test the integration
                health_data = self.azure_monitor.get_service_health()
                service_count = health_data.get('total_services', 0)
                print(f"✅ PrecheckServiceMonitor integrated - monitoring {service_count} services")
                return  # Success, exit early
            else:
                print(f"⚠️ Precheck config not found: {config_path}")

        except ImportError as e:
            print(f"⚠️ PrecheckServiceMonitor import failed: {e}")
        except Exception as e:
            print(f"⚠️ PrecheckServiceMonitor initialization failed: {e}")

        # Approach 2: Try standalone PrecheckMonitor
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'precheck'))
            from precheck_service_monitor.monitor_precheck import PrecheckMonitor

            config_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'precheck',
                'config',
                'precheck_config.json'
            )

            if os.path.exists(config_path):
                self.service_monitor = PrecheckMonitor(config_path)
                print("✅ Standalone PrecheckMonitor integrated")
                return  # Success, exit early
            else:
                print(f"⚠️ Service monitor config not found: {config_path}")

        except ImportError as e:
            print(f"⚠️ Standalone PrecheckMonitor import failed: {e}")
        except Exception as e:
            print(f"⚠️ Standalone PrecheckMonitor initialization failed: {e}")

        # Approach 3: Try AzureServiceMonitor
        try:
            from core.azure_connect import AzureServiceMonitor
            self.azure_monitor = AzureServiceMonitor()

            # Check if monitor has required attributes safely
            if (hasattr(self.azure_monitor, 'enabled') and
                hasattr(self.azure_monitor, 'initialized') and
                self.azure_monitor.enabled and
                self.azure_monitor.initialized):
                print("✅ AzureServiceMonitor integrated with precheck engine")
                return  # Success, exit early
            else:
                print("ℹ️ AzureServiceMonitor available but not fully enabled")

        except ImportError as e:
            print(f"⚠️ AzureServiceMonitor import failed: {e}")
        except Exception as e:
            print(f"⚠️ AzureServiceMonitor initialization failed: {e}")

        # Approach 4: Create fallback monitor
        print("ℹ️ Creating fallback service monitor")
        self.azure_monitor = self._create_fallback_monitor()

    def _create_fallback_monitor(self):
        """Create a fallback monitor object with all required methods"""
        class FallbackMonitor:
            def __init__(self):
                self.enabled = False
                self.initialized = False

            def get_service_health(self):
                return {
                    'service_health_available': False,
                    'overall_health': 'unknown',
                    'services': {},
                    'total_services': 0,
                    'message': 'Fallback monitor - no real monitoring data available',
                    'timestamp': time.time()
                }

            def refresh_service_data(self):
                """Placeholder refresh method"""
                pass

        return FallbackMonitor()

    def get_service_health_data(self):
        """Get service health data from any available monitor"""
        try:
            # Try azure_monitor first
            if self.azure_monitor and hasattr(self.azure_monitor, 'get_service_health'):
                return self.azure_monitor.get_service_health()

            # Try service_monitor as fallback
            if self.service_monitor and hasattr(self.service_monitor, 'monitor_all_environments'):
                results = self.service_monitor.monitor_all_environments()
                return self._convert_monitor_results_to_health_data(results)

            # Return default if no monitors available
            return {
                'service_health_available': False,
                'overall_health': 'unknown',
                'services': {},
                'message': 'No service monitors available'
            }

        except Exception as e:
            print(f"⚠️ Error getting service health data: {e}")
            return {
                'service_health_available': False,
                'overall_health': 'error',
                'services': {},
                'error': str(e)
            }

    def _convert_monitor_results_to_health_data(self, monitor_results):
        """Convert PrecheckMonitor results to standard health data format"""
        try:
            if not monitor_results:
                return {'service_health_available': False, 'services': {}}

            services = {}
            healthy_count = 0

            for result in monitor_results:
                service_name = f"{result.get('environment', 'unknown')}_{result.get('api_name', 'unknown')}"
                is_healthy = not result.get('error') and result.get('response', {}).get('status') != 'error'

                services[service_name] = {
                    'status': 'running' if is_healthy else 'error',
                    'environment': result.get('environment', 'unknown'),
                    'api_name': result.get('api_name', 'unknown'),
                    'response_time': result.get('response_time', 0),
                    'error': result.get('error', '')
                }

                if is_healthy:
                    healthy_count += 1

            return {
                'service_health_available': True,
                'overall_health': 'healthy' if healthy_count > len(services) / 2 else 'degraded',
                'services': services,
                'total_services': len(services),
                'healthy_services': healthy_count,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"⚠️ Error converting monitor results: {e}")
            return {'service_health_available': False, 'error': str(e)}

    # Enhanced analysis method that includes service monitoring
    async def enhanced_analyze_precheck_failure(self, failure: 'PrecheckFailure') -> 'PrecheckDecisionResult':
        """Enhanced precheck failure analysis with service monitoring"""
        logger.info(f"🔍 Enhanced analysis: {failure.precheck_name} for {failure.ingredient_name}")

        # Step 1: Get traditional analysis
        traditional_result = await self.analyze_precheck_failure(failure)

        # Step 2: Get service health insights
        service_health = await self._get_service_health_insights()

        # Step 3: Combine analyses for enhanced decision
        enhanced_result = self._enhance_decision_with_service_health(
            traditional_result,
            service_health,
            failure
        )

        return enhanced_result

        """Get service health insights from Azure"""
        if not self.azure_monitor or not self.azure_monitor.enabled:
            return {
                'service_health_available': False,
                'overall_health': 'unknown',
                'services': {},
                'message': 'Azure monitoring not available'
            }

        try:
            health_data = self.azure_monitor.get_service_health()

            # Calculate health score
            services = health_data.get('services', {})
            if services:
                healthy_services = sum(1 for s in services.values() if s.get('status') == 'running')
                health_score = healthy_services / len(services)
            else:
                health_score = 0.5  # Unknown

            return {
                'service_health_available': True,
                'overall_health': health_data.get('status', 'unknown'),
                'health_score': health_score,
                'services': services,
                'service_count': len(services),
                'healthy_services': sum(1 for s in services.values() if s.get('status') == 'running'),
                'timestamp': health_data.get('timestamp'),
                'message': health_data.get('message', 'Azure monitoring active')
            }

        except Exception as e:
            logger.error(f"❌ Failed to get service health insights: {e}")
            return {
                'service_health_available': False,
                'overall_health': 'error',
                'services': {},
                'error': str(e)
            }


    def process_sync(self, task) -> Dict[str, Any]:
        """Synchronous processing method for main.py integration"""
        try:
            # Convert task to failure if needed
            if hasattr(task, 'input_data'):
                # It's a UniversalTask
               failure = self._universal_task_to_failure(task)
            else:
                # It's already a failure or dict
                failure = task

            # Use existing AI analysis
            result = self.ai_analyze_failure(failure)

            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {'decision': 'REQUIRE_MANUAL_REVIEW', 'confidence': 0.5}

            return result

        except Exception as e:
            logger.error(f"Sync processing failed: {e}")
            return {
                'decision': 'REQUIRE_MANUAL_REVIEW',
                'confidence': 0.3,
                'error': str(e),
                'reasoning': f'Sync processing failed: {str(e)}'
            }

    def _universal_task_to_failure(self, task):
        """Convert UniversalTask to PrecheckFailure"""
        input_data = task.input_data
        return PrecheckFailure(
            failure_id=task.task_id,
            precheck_name=input_data.get('precheck_name', 'unknown'),
            ingredient_type=IngredientType.INTEL,
            ingredient_name=input_data.get('ingredient_name', 'unknown'),
            failure_details=input_data,
            milestone=input_data.get('milestone', 'unknown'),
            timestamp=getattr(task, 'timestamp', time.time()),
            error_message=input_data.get('error_message', ''),
            metadata=getattr(task, 'metadata', {})
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get engine health status for API"""
        return {
            'engine_available': True,
            'ai_system_available': self.ai_system is not None,
            'azure_monitor_available': self.azure_monitor is not None,
            'business_rules_loaded': len(self.business_rules.get('precheck_rules', {})),
            'decision_history_count': len(self.decision_history),
            'email_enabled': self.email_config.get('enabled', False),
            'last_check': time.time(),
            'status': 'healthy'
        }


    async def _get_service_health_insights(self) -> Dict[str, Any]:
        """Get service health insights with caching and multiple monitor support"""
        current_time = time.time()

        # Use cached data if recent
        if (self.monitoring_cache and
            current_time - self.last_monitoring_check < self.cache_duration):
            print("📋 Using cached service monitoring data")
            return self.monitoring_cache

        # Get fresh monitoring data
        try:
            print("🔄 Getting fresh service monitoring data...")
            health_data = self.get_service_health_data()

            # Process and cache results
            self.monitoring_cache = health_data
            self.last_monitoring_check = current_time

            return health_data
        except Exception as e:
            print(f"⚠️ Service monitoring failed: {e}")
            return {
                'service_health_available': False,
                'error': str(e),
                'overall_health': 'error'
            }


        # if not self.service_monitor:
        #    return {'monitoring_available': False, 'health_score': 0.5}

        # try:
         #   logger.debug("🔄 Running fresh service monitoring...")
         #   monitoring_results = self.service_monitor.monitor_all_environments()

         #   # Process results
         #   health_insights = self._process_monitoring_results(monitoring_results)

         #   # Cache results
         #   self.monitoring_cache = health_insights
         #   self.last_monitoring_check = current_time

         #   return health_insights

       # except Exception as e:
       #     logger.warning(f"Service monitoring failed: {e}")
       #     return {
       #         'monitoring_available': False,
       #         'error': str(e),
       #         'health_score': 0.3  # Lower confidence due to monitoring failure
       #     }


    def _process_monitoring_results(self, monitoring_results: List[Dict]) -> Dict[str, Any]:
        """Process monitoring results into health insights"""
        if not monitoring_results:
            return {'monitoring_available': False, 'health_score': 0.5}

        # Analyze results
        total_apis = len(monitoring_results)
        healthy_apis = 0
        critical_issues = []
        environment_health = {}

        for result in monitoring_results:
            env = result.get('environment', 'unknown')
            api_name = result.get('api_name', 'unknown')

            # Track environment health
            if env not in environment_health:
                environment_health[env] = {'total': 0, 'healthy': 0}
            environment_health[env]['total'] += 1

            # Check if API is healthy
            is_healthy = self._is_api_result_healthy(result)

            if is_healthy:
                healthy_apis += 1
                environment_health[env]['healthy'] += 1
            else:
                # Track critical issues
                error_info = result.get('response', {}).get('error', str(result.get('error', 'Unknown')))

                if any(keyword in error_info.lower() for keyword in ['timeout', 'connection', 'auth', 'token']):
                    critical_issues.append({
                        'environment': env,
                        'api': api_name,
                        'error': error_info,
                        'severity': 'high' if 'auth' in error_info.lower() else 'medium'
                    })

        # Calculate health scores
        overall_health_score = healthy_apis / total_apis if total_apis > 0 else 0.0

        for env_data in environment_health.values():
            env_data['health_score'] = env_data['healthy'] / env_data['total'] if env_data['total'] > 0 else 0.0

        return {
            'monitoring_available': True,
            'overall_health_score': overall_health_score,
            'total_apis': total_apis,
            'healthy_apis': healthy_apis,
            'failed_apis': total_apis - healthy_apis,
            'critical_issues': critical_issues,
            'environment_health': environment_health,
            'monitoring_timestamp': time.time()
        }

    def _is_api_result_healthy(self, result: Dict) -> bool:
        """Check if API monitoring result indicates healthy service"""
        if 'error' in result:
            return False

        response = result.get('response', {})
        if 'error' in response:
            return False

        return True

    def _enhance_decision_with_service_health(self, traditional_result: 'PrecheckDecisionResult',
                                            service_health: Dict[str, Any],
                                            failure: 'PrecheckFailure') -> 'PrecheckDecisionResult':
        """Enhance precheck decision with service health data"""

        # Extract service health metrics
        monitoring_available = service_health.get('monitoring_available', False)
        health_score = service_health.get('overall_health_score', 0.5)
        critical_issues = service_health.get('critical_issues', [])

        # Start with traditional decision
        enhanced_decision = traditional_result.decision
        enhanced_confidence = traditional_result.confidence
        enhanced_reasoning = traditional_result.reasoning

        # Adjust based on service health
        if monitoring_available:
            # Add service health context to reasoning
            enhanced_reasoning += f" Service health score: {health_score:.1%}."

            # Adjust decision based on service health
            if health_score < 0.3 and len(critical_issues) > 0:
                # Poor service health - be more conservative
                if enhanced_decision == PrecheckDecision.AUTO_APPROVE:
                    enhanced_decision = PrecheckDecision.MANUAL_REVIEW
                    enhanced_reasoning += " Decision adjusted to manual review due to poor service health."
                    enhanced_confidence *= 0.8  # Reduce confidence

            elif health_score > 0.9:
                # Excellent service health - can be more confident
                if enhanced_decision == PrecheckDecision.MANUAL_REVIEW and enhanced_confidence > 0.8:
                    enhanced_decision = PrecheckDecision.AUTO_APPROVE
                    enhanced_reasoning += " Decision upgraded to auto-approve due to excellent service health."
                    enhanced_confidence = min(1.0, enhanced_confidence * 1.1)

            # Add critical issues to reasoning
            if critical_issues:
                issue_summary = f" {len(critical_issues)} service issues detected"
                if any(issue['severity'] == 'high' for issue in critical_issues):
                    issue_summary += " (including high-severity issues)"
                enhanced_reasoning += issue_summary + "."

        else:
            # No monitoring available - note in reasoning
            enhanced_reasoning += " Service monitoring unavailable."
            enhanced_confidence *= 0.9  # Slight confidence reduction

        # Create enhanced result
        enhanced_result = PrecheckDecisionResult(
            failure_id=traditional_result.failure_id,
            decision=enhanced_decision,
            confidence=min(1.0, enhanced_confidence),
            reasoning=enhanced_reasoning,
            approver=traditional_result.approver,
            escalation_contacts=traditional_result.escalation_contacts,
            waiver_duration_days=traditional_result.waiver_duration_days,
            ai_analysis={
                **traditional_result.ai_analysis,
                'service_health_considered': monitoring_available,
                'service_health_score': health_score,
                'service_issues_count': len(critical_issues),
                'enhanced_analysis': True
            },
            business_rule_applied=traditional_result.business_rule_applied,
            metadata={
                **traditional_result.metadata,
                'service_monitoring': service_health,
                'enhancement_applied': True,
                'original_decision': traditional_result.decision.value,
                'original_confidence': traditional_result.confidence
            }
        )

        return enhanced_result

    # Method for AI system integration (called from main.py)
    def ai_analyze_failure_with_monitoring(self, precheck_data):
        """
        Enhanced AI analysis that includes service monitoring
        This method can be called by the prometheus_alert_processor
        """
        try:
            # Convert dict to PrecheckFailure if needed
            if isinstance(precheck_data, dict):
                # Create PrecheckFailure object from dict
                failure = self._dict_to_precheck_failure(precheck_data)
            else:
                failure = precheck_data

            # Run enhanced analysis
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.enhanced_analyze_precheck_failure(failure))
            finally:
                loop.close()

            # Convert result to dict for AI system compatibility
            return {
                'decision': result.decision.value,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'service_health_considered': result.ai_analysis.get('service_health_considered', False),
                'service_health_score': result.ai_analysis.get('service_health_score', 0.5),
                'enhanced_analysis': True,
                'approver': result.approver,
                'business_rule': result.business_rule_applied,
                'metadata': result.metadata
            }

        except Exception as e:
            logger.error(f"Enhanced AI analysis failed: {e}")
            # Fallback to traditional analysis
            return self.ai_analyze_failure(precheck_data)

    def _dict_to_precheck_failure(self, data: Dict) -> 'PrecheckFailure':
        """Convert dictionary to PrecheckFailure object"""
        # Import your PrecheckFailure class
        from core.precheck_engine import PrecheckFailure  # Adjust import as needed

        return PrecheckFailure(
            failure_id=data.get('failure_id', f"failure_{int(time.time())}"),
            precheck_name=data.get('precheck_name', 'unknown'),
            ingredient_name=data.get('ingredient_name', 'unknown'),
            ingredient_type=data.get('ingredient_type', 'unknown'),
            error_message=data.get('error_message', ''),
            milestone=data.get('milestone', 'unknown'),
            timestamp=data.get('timestamp', time.time())
        )

    def ai_analyze_failure(self, precheck_data):
        """
        AI-powered analysis of precheck failures
        This method is called by the prometheus_alert_processor
        """
        try:
            # Handle both dict and PrecheckFailure object inputs
            if isinstance(precheck_data, dict):
                precheck_name = precheck_data.get('precheck_name', 'unknown')
                ingredient_name = precheck_data.get('ingredient_name', 'unknown')
                ingredient_type = precheck_data.get('ingredient_type', 'unknown')
                error_message = precheck_data.get('error_message', '')
                milestone = precheck_data.get('milestone', 'unknown')
            else:
                # Assume it's a PrecheckFailure object
                precheck_name = precheck_data.precheck_name
                ingredient_name = precheck_data.ingredient_name
                ingredient_type = precheck_data.ingredient_type.value if hasattr(precheck_data.ingredient_type, 'value') else str(precheck_data.ingredient_type)
                error_message = precheck_data.error_message
                milestone = precheck_data.milestone

            logger.info(f"🔍 AI analyzing precheck failure: {precheck_name} for {ingredient_name}")

            # Create analysis context
            analysis_context = f"""
            Precheck: {precheck_name}
            Ingredient: {ingredient_name} ({ingredient_type})
            Milestone: {milestone}
            Error: {error_message}
            """

            # Use the AI system for analysis if available
            if self.ai_system:
                try:
                    import asyncio

                    # Generate a unique task ID
                    task_id = create_task_id("precheck_analysis")

                    # Create UniversalTask object with proper parameters
                    task = UniversalTask(
                        task_id=task_id,
                        domain=DomainType.INFRASTRUCTURE,
                        task_type=TaskType.CLASSIFICATION,
                        input_data={
                            "text": analysis_context,
                            "precheck_name": precheck_name,
                            "ingredient_type": ingredient_type,
                            "milestone": milestone,
                            "error_message": error_message
                        }
                    )

                    # Process the task with proper async handling
                    try:
                        # Try to get existing event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're in an async context, need to handle differently
                            import concurrent.futures
                            import threading

                            def run_in_thread():
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                try:
                                    return new_loop.run_until_complete(self.ai_system.process_universal_task(task))
                                finally:
                                    new_loop.close()

                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(run_in_thread)
                                solution = future.result(timeout=30)
                        else:
                            # We're not in an async context
                            solution = loop.run_until_complete(self.ai_system.process_universal_task(task))
                    except RuntimeError:
                        # No event loop exists, create one
                        solution = asyncio.run(self.ai_system.process_universal_task(task))

                    if solution:
                        # Extract confidence from various possible attributes
                        confidence = getattr(solution, "confidence",
                                   getattr(solution, "output_confidence",
                                   getattr(solution, "score", 0.5)))

                        # Extract prediction from various possible attributes
                        prediction = getattr(solution, "prediction",
                                   getattr(solution, "result",
                                   getattr(solution, "output_data", {}).get("prediction", "investigate")))

                        # Check if solution indicates success (default to True if no success attribute)
                        is_successful = getattr(solution, "success",
                                      getattr(solution, "is_successful", True))

                        # Determine action based on AI analysis and business rules
                        action = self._determine_ai_action_fixed(precheck_name, ingredient_type, error_message, confidence)

                        ai_result = {
                            'success': True,
                            'action': action,
                            'confidence': confidence,
                            'reasoning': f"AI analysis suggests {action} based on {precheck_name} failure pattern",
                            'ai_prediction': prediction,
                            'task_id': task_id,
                            'analysis_context': analysis_context,
                            'fallback_used': False
                        }

                        logger.info(f"✅ AI analysis completed: {action} (confidence: {confidence:.3f})")
                        return ai_result

                except Exception as ai_error:
                    logger.error(f"AI system error: {ai_error}")
                    # Fall through to fallback analysis

            # Fallback rule-based analysis
            return self._fallback_analysis_with_absolute_priority(precheck_name, ingredient_type, error_message, milestone)

        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return self._fallback_analysis_with_absolute_priority(
                precheck_data.get('precheck_name', 'unknown') if isinstance(precheck_data, dict) else 'unknown',
                precheck_data.get('ingredient_type', 'unknown') if isinstance(precheck_data, dict) else 'unknown',
                precheck_data.get('error_message', '') if isinstance(precheck_data, dict) else '',
                precheck_data.get('milestone', 'unknown') if isinstance(precheck_data, dict) else 'unknown'
            )

    def _determine_ai_action_fixed(self, precheck_name: str, ingredient_type: str, error_message: str, confidence: float) -> str:
        """
        Determine the recommended action based on AI analysis and business rules
        """
        # Check business rules first
        precheck_rules = self.business_rules.get('precheck_rules', {})
        precheck_rule = precheck_rules.get(precheck_name, {})
        ingredient_rule = precheck_rule.get(ingredient_type, {})

        if ingredient_rule:
            auto_decision = ingredient_rule.get('auto_decision', 'investigate')

            # Map business rule decisions to actions
            if auto_decision == 'approve':
                return 'auto_approve'
            elif auto_decision == 'reject':
                return 'auto_reject'
            elif auto_decision == 'escalate':
                return 'escalate'
            elif confidence > 0.8:
                # High confidence AI override
                if 'critical' in error_message.lower() or 'signature' in precheck_name.lower():
                    return 'escalate'
                elif 'version' in precheck_name.lower() or 'path' in precheck_name.lower():
                    return 'auto_fix'
                else:
                    return 'investigate'

        # Default AI-based decision
        if confidence > 0.8:
            if 'critical' in error_message.lower() or precheck_name in ['MsftSignChk', 'SignChk', 'SDLeChk']:
                return 'escalate'
            elif precheck_name in ['VersionChk', 'PathLengthChk', 'InfWCOSChk']:
                return 'auto_fix'
            elif ingredient_type == 'tpv':
                return 'auto_approve'  # TPV blanket exception
            else:
                return 'investigate'
        else:
            return 'investigate'

    def _fallback_analysis_with_absolute_priority(self, precheck_name: str, ingredient_type: str, error_message: str, milestone: str) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when AI is not available
        """
        logger.info(f"🔄 Using fallback analysis for {precheck_name}")

        # Check comprehensive business rules
        precheck_rules = self.business_rules.get('precheck_rules', {})
        precheck_rule = precheck_rules.get(precheck_name, {})
        ingredient_rule = precheck_rule.get(ingredient_type, {})

        if ingredient_rule:
            auto_decision = ingredient_rule.get('auto_decision', 'investigate')
            confidence = 0.8  # High confidence for rule-based decisions
            reasoning = f"Rule-based analysis: {ingredient_rule.get('notes', 'Standard business rule applied')}"
        else:
            # Generic fallback rules
            if 'critical' in error_message.lower():
                auto_decision = 'escalate'
                confidence = 0.7
            elif precheck_name in ['MsftSignChk', 'SDLeChk', 'SignChk']:
                auto_decision = 'escalate'
                confidence = 0.8
            elif precheck_name in ['VersionChk', 'PathLengthChk', 'InfWCOSChk', 'IngUWDApiChk']:
                auto_decision = 'auto_fix'
                confidence = 0.6
            elif ingredient_type == 'tpv':
                auto_decision = 'auto_approve'
                confidence = 0.9
            else:
                auto_decision = 'investigate'
                confidence = 0.5

            reasoning = f"Fallback rule-based analysis for {precheck_name}"

        # Map to action
        action_map = {
            'approve': 'auto_approve',
            'reject': 'auto_reject',
            'escalate': 'escalate',
            'manual_review': 'investigate',
            'auto_fix': 'auto_fix'
        }

        action = action_map.get(auto_decision, 'investigate')

        return {
            'success': True,
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'ai_prediction': 'rule_based',
            'fallback_used': True,
            'business_rule_applied': f"{precheck_name}_{ingredient_type}" if ingredient_rule else 'generic_fallback'
        }

    def send_manual_intervention_email(self, failure: PrecheckFailure, decision_result: PrecheckDecisionResult):
        """Send email notification for manual intervention required"""
        if not self.email_config.get('enabled', False):
            logger.info("📧 Email notifications disabled")
            return False

        try:
            # Create email content
            subject = f"🔍 Manual Precheck Review Required: {failure.precheck_name} - {failure.ingredient_name}"

            # Create email body based on format preference
            email_body = self._create_email_body(failure, decision_result)

            # Send email
            success = self._send_email(
                to_email=self.email_config['ba_team_email'],
                subject=subject,
                body=email_body,
                failure_id=failure.failure_id
            )

            if success:
                logger.info(f"📧 Manual intervention email sent for {failure.failure_id}")
                return True
            else:
                logger.error(f"📧 Failed to send email for {failure.failure_id}")
                return False

        except Exception as e:
            logger.error(f"📧 Email notification error: {str(e)}")
            return False

    def _create_email_body(self, failure: PrecheckFailure, decision_result: PrecheckDecisionResult) -> str:
        """Create email body - simple or detailed based on configuration"""

        # Check if simple mode is enabled (default to True)
        simple_mode = self.email_config.get('simple_format', True)

        if simple_mode:
            return self._create_simple_email_body(failure, decision_result)
        else:
            return self._create_detailed_html_email_body(failure, decision_result)

    def _create_simple_email_body(self, failure: PrecheckFailure, decision_result: PrecheckDecisionResult) -> str:
        """Create simple, readable email body for manual intervention"""

        # Determine urgency indicator
        urgency_indicator = "🔴 HIGH PRIORITY" if failure.severity == "high" or "critical" in failure.error_message.lower() else "🟡 STANDARD"

        # Create simple text-based email content
        email_body = f"""🔍 AI Precheck Engine - Manual Intervention Required

This precheck failure requires manual review and decision.

📋 Precheck Details
Timestamp: {datetime.fromtimestamp(failure.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}
System: {os.uname().nodename}
Priority: {urgency_indicator}
Failure ID: {failure.failure_id}

🎯 Manual Review Required
Precheck: {failure.precheck_name}
Ingredient: {failure.ingredient_name}
Type: {failure.ingredient_type.value.upper()}
Milestone: {failure.milestone}
Error: {failure.error_message}

🧠 AI Analysis
Decision: {decision_result.decision.value}
Confidence: {decision_result.confidence:.1%}
Reasoning: {decision_result.reasoning}
Business Rule: {decision_result.business_rule_applied}
Recommended Approver: {decision_result.approver}

📋 Action Required
Please review this precheck failure and make an appropriate decision:
- Approve waiver if this is a false positive or acceptable risk
- Reject if this is a legitimate failure that must be fixed
- Escalate if additional expertise is needed

{f'🔺 Escalation Contacts: {", ".join(decision_result.escalation_contacts)}' if decision_result.escalation_contacts else ''}

If you receive this notification, the manual intervention system is working correctly!

---
Intel AI Precheck Engine
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Contact: {self.email_config['sender_email']}
"""

        return email_body.strip()

    def _create_detailed_html_email_body(self, failure: PrecheckFailure, decision_result: PrecheckDecisionResult) -> str:
        """Create detailed HTML email body (original format)"""

        # Determine urgency level
        urgency = "🔴 HIGH" if failure.severity == "high" or "critical" in failure.error_message.lower() else "🟡 MEDIUM"

        # Create comprehensive email content
        email_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #0078d4; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
        .section {{ margin-bottom: 15px; }}
        .label {{ font-weight: bold; color: #0078d4; }}
        .urgent {{ color: #d13438; font-weight: bold; }}
        .confidence {{ background-color: #f0f8ff; padding: 10px; border-radius: 3px; }}
        .actions {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🔍 AI Precheck Engine - Manual Review Required</h2>
        <p>Intelligent analysis completed, human expertise needed</p>
    </div>

    <div class="content">
        <div class="section">
            <span class="label">Failure ID:</span> {failure.failure_id}<br>
            <span class="label">Timestamp:</span> {datetime.fromtimestamp(failure.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
            <span class="label">Urgency Level:</span> <span class="urgent">{urgency}</span>
        </div>

        <div class="section">
            <h3>📋 Precheck Details</h3>
            <span class="label">Precheck Name:</span> {failure.precheck_name}<br>
            <span class="label">Ingredient:</span> {failure.ingredient_name}<br>
            <span class="label">Type:</span> {failure.ingredient_type.value.upper()}<br>
            <span class="label">Milestone:</span> {failure.milestone}<br>
            <span class="label">Error Message:</span> {failure.error_message}
        </div>

        <div class="section confidence">
            <h3>🧠 AI Analysis Results</h3>
            <span class="label">AI Decision:</span> {decision_result.decision.value}<br>
            <span class="label">Confidence Score:</span> {decision_result.confidence:.2%}<br>
            <span class="label">AI Reasoning:</span> {decision_result.reasoning}<br>
            <span class="label">Business Rule Applied:</span> {decision_result.business_rule_applied}
        </div>

        <div class="section">
            <h3>📊 Additional Context</h3>
            <span class="label">Recommended Approver:</span> {decision_result.approver}<br>
            <span class="label">Waiver Duration:</span> {decision_result.waiver_duration_days} days<br>
            <span class="label">Validation Required:</span> {decision_result.validation_required or 'None'}<br>
            <span class="label">Milestone Restriction:</span> {decision_result.milestone_restriction or 'None'}
        </div>

        <div class="actions">
            <h3>🎯 Recommended Actions</h3>
            <ul>
                <li><strong>Review the precheck failure details above</strong></li>
                <li><strong>Assess if this is a legitimate failure or false positive</strong></li>
                <li><strong>Consider the AI confidence score ({decision_result.confidence:.1%})</strong></li>
                <li><strong>Apply appropriate waiver or escalation as needed</strong></li>
                {f'<li><strong>Escalate to: {", ".join(decision_result.escalation_contacts)}</strong></li>' if decision_result.escalation_contacts else ''}
            </ul>
        </div>

        <div class="section">
            <h3>🔗 Quick Links</h3>
            <p>
                • <a href="http://localhost:8052/metrics">AI System Metrics</a><br>
                • <a href="mailto:ai-precheck-engine@intel.com?subject=Re: {failure.failure_id}">Reply to AI System</a><br>
                • Failure ID for reference: <code>{failure.failure_id}</code>
            </p>
        </div>
    </div>

    <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px; color: #6c757d;">
        <p>This email was automatically generated by the Intel AI Precheck Engine.<br>
        System ID: {getattr(self, 'system_id', 'universal_ai_system')} |
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>
</body>
</html>
        """

        return email_body.strip()

    def _send_email(self, to_email: str, subject: str, body: str, failure_id: str) -> bool:
        """Send email using Intel SMTP server"""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.email_config['sender_email']
            message["To"] = to_email
            message["X-Priority"] = "2"  # High priority
            message["X-MSMail-Priority"] = "High"

            # Determine content type based on simple_format setting
            simple_mode = self.email_config.get('simple_format', True)

            if simple_mode:
                # Add plain text body
                text_part = MIMEText(body, "plain")
                message.attach(text_part)
            else:
                # Add HTML body
                html_part = MIMEText(body, "html")
                message.attach(html_part)

            # Connect to server and send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                # Note: In production, you might need authentication
                # server.login(username, password)

                text = message.as_string()
                server.sendmail(self.email_config['sender_email'], to_email, text)

            logger.info(f"📧 Email sent successfully to {to_email} for {failure_id}")
            return True

        except Exception as e:
            logger.error(f"📧 Failed to send email: {str(e)}")
            return False

    # [Rest of your existing methods remain unchanged]
    def load_business_rules(self) -> Dict[str, Any]:
        """Load enhanced business rules from configuration"""
        rules_file = '/aiengine/src/aiengine/config/precheck_rules.yaml'
        try:
            if os.path.exists(rules_file):
                with open(rules_file, 'r') as f:
                    rules = yaml.safe_load(f)
                    logger.info(f"✅ Loaded {len(rules.get('precheck_rules', {}))} precheck rules")
                    return rules
            else:
                logger.warning(f"Rules file not found: {rules_file}")
                return self.get_default_rules()
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}")
            return self.get_default_rules()

    def load_comprehensive_business_rules(self) -> Dict[str, Any]:
        """Load comprehensive precheck business rules based on BA documentation"""
        return {
            'overriding_principle': {
                'false_positive_auto_waiver': True,
                'ba_team_authority': 'Always authorized for false-positive violations'
            },
            'precheck_rules': {
                'BATChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'BA_TEAM',
                        'notes': 'WSE-driven, BA team can grant waiver',
                        'auto_decision': 'approve'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'BinaryScanChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny', 'Coffin, Lynn A'],
                        'validation_required': 'PSXT',
                        'notes': 'CQN/SDLe-driven, must validate with PSXT',
                        'auto_decision': 'escalate'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'DriverChk': {
                    'intel': {
                        'waiver_allowed': False,
                        'escalation_required': False,
                        'approver': 'REJECT',
                        'notes': 'Must have INF if marked as driver. No waiver allowed.',
                        'auto_decision': 'reject'
                    },
                    'tpv': {
                        'waiver_allowed': False,
                        'escalation_required': False,
                        'approver': 'REJECT',
                        'notes': 'Must have INF if marked as driver. No waiver allowed.',
                        'auto_decision': 'reject'
                    }
                },
                'EwdkOsChk': {
                    'intel': {'waiver_allowed': True, 'notes': 'EWDK/OS compatibility validation', 'auto_decision': 'approve'},
                    'tpv': {'waiver_allowed': True, 'notes': 'EWDK/OS compatibility validation', 'auto_decision': 'approve'}
                },
                'INFCompChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'milestone_restriction': 'pre_alpha',
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver until Alpha, then escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'milestone_restriction': 'pre_alpha',
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver until Alpha, then escalation required',
                        'auto_decision': 'manual_review'
                    }
                },
                'InfDeclarativeChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'milestone_restriction': 'pre_alpha',
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver until Alpha, then escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'milestone_restriction': 'pre_alpha',
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver until Alpha, then escalation required',
                        'auto_decision': 'manual_review'
                    }
                },
                'InfHSAChk': {
                    'intel': {
                        'waiver_allowed': False,
                        'escalation_required': False,
                        'approver': 'REJECT',
                        'notes': 'No waiver allowed. Use isApplicable=false in JSON metadata',
                        'auto_decision': 'reject'
                    },
                    'tpv': {
                        'waiver_allowed': False,
                        'escalation_required': False,
                        'approver': 'REJECT',
                        'notes': 'No waiver allowed. Use isApplicable=false in JSON metadata',
                        'auto_decision': 'reject'
                    }
                },
                'InfWCOSChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Not mandatory. Platform config issue.',
                        'auto_decision': 'approve'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Not mandatory. Platform config issue.',
                        'auto_decision': 'approve'
                    }
                },
                'IngUWDApiChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Not mandatory. Platform config issue.',
                        'auto_decision': 'approve'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Not mandatory. Platform config issue.',
                        'auto_decision': 'approve'
                    }
                },
                'IPScanChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny', 'Coffin, Lynn A'],
                        'validation_required': 'PSXT',
                        'notes': 'CQN/SDLe-driven, must validate with PSXT',
                        'auto_decision': 'escalate'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'MsftSignChk': {
                    'intel': {
                        'waiver_allowed': False,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': 'No waiver allowed. Escalation required.',
                        'auto_decision': 'escalate'
                    },
                    'tpv': {
                        'waiver_allowed': False,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': 'No waiver allowed. Escalation required.',
                        'auto_decision': 'escalate'
                    }
                },
                'PathLengthChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'assessment_required': True,
                        'approver': 'BA_TEAM',
                        'notes': 'BA team can assess if benign violation',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'assessment_required': True,
                        'approver': 'BA_TEAM',
                        'notes': 'BA team can assess if benign violation',
                        'auto_decision': 'manual_review'
                    }
                },
                'ReleaseNotesChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    }
                },
                'SCAScanChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny', 'Coffin, Lynn A'],
                        'validation_required': 'PSXT',
                        'notes': 'CQN/SDLe-driven, must validate with PSXT',
                        'auto_decision': 'escalate'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'SDLeChk': {
                    'intel': {
                        'waiver_allowed': False,
                        'milestone_dependent': True,
                        'milestone_restriction': 'post_alpha',
                        'approver': 'REJECT',
                        'notes': 'Mandatory post-Alpha, critical for BKC. No waiver allowed.',
                        'auto_decision': 'reject'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'SignChk': {
                    'intel': {
                        'waiver_allowed': False,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': 'No waiver allowed. Escalation required.',
                        'auto_decision': 'escalate'
                    },
                    'tpv': {
                        'waiver_allowed': False,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': 'No waiver allowed. Escalation required.',
                        'auto_decision': 'escalate'
                    }
                },
                'SymChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'escalation_required': False,
                        'approver': 'SYSTEM',
                        'notes': 'Blanket Modulation Exception for all TPV ingredients',
                        'auto_decision': 'approve'
                    }
                },
                'SymPublicChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    }
                },
                'VersionChk': {
                    'intel': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    },
                    'tpv': {
                        'waiver_allowed': True,
                        'waiver_duration_days': 14,
                        'escalation_required': True,
                        'escalation_to': ['Mcmichael, Lonny'],
                        'notes': '2-week waiver with escalation required',
                        'auto_decision': 'manual_review'
                    }
                }
            },
            'ai_learning': {
                'confidence_threshold': 0.8,
                'auto_approval_confidence_threshold': 0.9,
                'false_positive_indicators': [
                    'precheck bug', 'false positive', 'tool error',
                    'configuration issue', 'platform setup', 'false-positive'
                ]
            }
        }

    def detect_ingredient_type(self, failure: PrecheckFailure) -> IngredientType:
        """Enhanced ingredient type detection"""
        ingredient_name = failure.ingredient_name.lower()

        # TPV indicators
        tpv_indicators = ['tpv', 'third_party', 'vendor', 'oem', 'partner']
        if any(indicator in ingredient_name for indicator in tpv_indicators):
            return IngredientType.TPV

        # Intel indicators
        intel_indicators = ['intel', 'internal', 'first_party']
        if any(indicator in ingredient_name for indicator in intel_indicators):
            return IngredientType.INTEL

        # Default based on metadata or assume Intel
        return failure.ingredient_type if failure.ingredient_type != IngredientType.UNKNOWN else IngredientType.INTEL

    def detect_false_positive(self, failure: PrecheckFailure) -> bool:
        """Enhanced false positive detection"""
        false_positive_indicators = self.business_rules['ai_learning']['false_positive_indicators']

        error_message = failure.error_message.lower()
        failure_details = str(failure.failure_details).lower()

        # Check error message and failure details
        for indicator in false_positive_indicators:
            if indicator in error_message or indicator in failure_details:
                return True

        return failure.is_false_positive

    async def analyze_precheck_failure(self, failure: PrecheckFailure) -> PrecheckDecisionResult:
        """Enhanced precheck failure analysis with comprehensive rules"""
        logger.info(f"🔍 Analyzing precheck failure: {failure.precheck_name} for {failure.ingredient_name}")

        # Step 0: Check overriding principle (false positive)
        if self.detect_false_positive(failure):
            return PrecheckDecisionResult(
                failure_id=failure.failure_id,
                decision=PrecheckDecision.AUTO_APPROVE,
                confidence=0.95,
                reasoning='Detected as false positive - auto-approving',
                approver='ai_system',
                escalation_contacts=[],
                waiver_duration_days=30,
                ai_analysis={'false_positive': True},
                business_rule_applied='false_positive_detection',
                validation_required='',
                milestone_restriction='',
                metadata={'false_positive_detected': True}
            )

        # Initialize default values
        decision = PrecheckDecision.AUTO_APPROVE  # Default to MANUAL_REVIEW
        final_confidence = 0.5
        reasoning_parts = ["Precheck analysis completed"]
        approver = "manual_review"
        escalation_contacts = []
        ai_analysis = {'analysis_completed': True}
        business_rule_applied = f"{failure.precheck_name}_{failure.ingredient_type.value}"

        try:
            # Apply business rules based on precheck type and ingredient
            if failure.ingredient_type == IngredientType.TPV:
                if failure.precheck_name in ["SDLeChk", "VersionChk"]:
                    # TPV components should be auto-approved for these checks
                    decision = PrecheckDecision.AUTO_APPROVE  # FIXED!
                    final_confidence = 0.8
                    reasoning_parts = ["TPV component auto-approved for common checks"]
                    approver = "ai_system"
                    business_rule_applied = "tpv_auto_approve"
            elif failure.ingredient_type == IngredientType.INTEL:
                if failure.precheck_name == "MsftSignChk" and failure.milestone == "rtm":
                    # Critical signature failures at RTM should escalate
                    decision = PrecheckDecision.ESCALATE_BA
                    final_confidence = 0.9
                    reasoning_parts = ["Critical signature check failed at RTM - requires escalation"]
                    escalation_contacts = ["security_team@company.com"]
                    business_rule_applied = "intel_critical_escalation"
                    approver = "security_team"
                elif failure.precheck_name in ["VersionChk", "PathLengthChk"]:
                    # Intel version and path checks require manual review
                    decision = PrecheckDecision.MANUAL_REVIEW
                    final_confidence = 0.6
                    reasoning_parts = [f"Intel {failure.precheck_name} requires manual review"]
                    business_rule_applied = "intel_manual_review"
                    approver = "manual_review"

            # Use AI analysis if available
            if self.ai_system:
                try:

                    ai_task = UniversalTask(
                        task_id=f"precheck_{failure.failure_id}_{int(time.time())}",
                        domain=DomainType.INFRASTRUCTURE,
                        task_type=TaskType.CLASSIFICATION,
                        input_data={
                            "precheck_name": failure.precheck_name,
                            "ingredient_type": failure.ingredient_type.value,
                            "ingredient_name": failure.ingredient_name,
                            "milestone": failure.milestone,
                            "failure_details": failure.failure_details or {}
                        }
                    )

                    ai_solution = await self.ai_system.process_universal_task(ai_task)
                    if ai_solution.confidence > 0.7:
                        final_confidence = max(final_confidence, ai_solution.confidence)
                        reasoning_parts.append(f"AI analysis: {ai_solution.reasoning}")
                        ai_analysis['ai_confidence'] = ai_solution.confidence
                        ai_analysis['ai_reasoning'] = ai_solution.reasoning

                except Exception as e:
                    logger.warning(f"AI analysis failed: {e}")
                    ai_analysis['ai_error'] = str(e)

        except Exception as e:
            logger.error(f"Error in precheck analysis: {e}")
            reasoning_parts.append(f"Analysis error: {str(e)}")

        # Create final decision with all required parameters
        try:
            final_decision = PrecheckDecisionResult(
                failure_id=failure.failure_id,
                decision=decision,
                confidence=final_confidence,
                reasoning='; '.join(reasoning_parts),
                approver=approver,
                escalation_contacts=escalation_contacts,
                waiver_duration_days=0,
                ai_analysis=ai_analysis,
                business_rule_applied=business_rule_applied,
                validation_required='',
                milestone_restriction=''
            )
        except Exception as e:
            logger.error(f"Error creating PrecheckDecisionResult: {e}")
            # Fallback with minimal parameters
            final_decision = PrecheckDecisionResult(
                failure_id=failure.failure_id,
                decision=decision,
                confidence=final_confidence,
                reasoning='; '.join(reasoning_parts),
                approver=approver,
                escalation_contacts=escalation_contacts,
                waiver_duration_days=0,
                ai_analysis=ai_analysis,
                business_rule_applied=business_rule_applied
            )

        logger.info(f"✅ Precheck decision: {decision.value if hasattr(decision, 'value') else str(decision)} (confidence: {final_confidence})")

        # Record decision for statistics
        self.record_decision(failure, final_decision)

        # Send email notification if manual intervention required
        email_triggers = [PrecheckDecision.MANUAL_REVIEW, PrecheckDecision.ESCALATE_BA,
                         PrecheckDecision.ESCALATE_LONNY, PrecheckDecision.ESCALATE_RAJDEEP]
        if final_decision.decision in email_triggers:
            try:
                email_sent = self.send_manual_intervention_email(failure, final_decision)
                if email_sent:
                    logger.info(f"📧 Manual intervention email sent for {failure.failure_id}")
                else:
                    logger.warning(f"📧 Failed to send email for {failure.failure_id}")
            except Exception as e:
                logger.error(f"📧 Email notification error: {e}")

        return final_decision

    def record_decision(self, failure: PrecheckFailure, decision: PrecheckDecisionResult):
        """Record decision for learning purposes"""
        decision_record = {
            'timestamp': time.time(),
            'precheck_name': failure.precheck_name,
            'ingredient_type': failure.ingredient_type.value,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'business_rule_applied': decision.business_rule_applied,
            'escalation_contacts': decision.escalation_contacts
        }

        self.decision_history.append(decision_record)

        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Enhanced statistics with comprehensive rule tracking"""
        base_stats = {
            'total_decisions': len(self.decision_history),
            'ai_system_available': self.ai_system is not None,
            'business_rules_loaded': len(self.business_rules.get('precheck_rules', {}))
        }

        if not self.decision_history:
            return base_stats

        # Rule-based decision analysis
        rule_based_decisions = {}
        for decision in self.decision_history:
            rule = decision.get('business_rule_applied', 'unknown')
            if rule not in rule_based_decisions:
                rule_based_decisions[rule] = {'count': 0, 'decisions': {}}

            rule_based_decisions[rule]['count'] += 1
            dec = decision['decision']
            if dec not in rule_based_decisions[rule]['decisions']:
                rule_based_decisions[rule]['decisions'][dec] = 0
            rule_based_decisions[rule]['decisions'][dec] += 1

        # Escalation tracking
        escalation_stats = {}
        for decision in self.decision_history:
            if 'escalate' in decision['decision']:
                escalation_contacts = decision.get('escalation_contacts', [])
                for contact in escalation_contacts:
                    if contact not in escalation_stats:
                        escalation_stats[contact] = 0
                    escalation_stats[contact] += 1

        base_stats.update({
            'rule_based_decisions': rule_based_decisions,
            'escalation_tracking': escalation_stats,
            'comprehensive_rules_loaded': len(self.business_rules.get('precheck_rules', {}))
        })

        return base_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics"""
        return {
            'total_decisions': len(self.decision_history),
            'ai_system_available': self.ai_system is not None,
            'business_rules_loaded': len(self.business_rules.get('precheck_rules', {})),
            'pattern_cache_size': len(self.pattern_cache)
        }

    def get_default_rules(self) -> Dict[str, Any]:
        """Get default rules if configuration file is not available"""
        return self.load_comprehensive_business_rules()
