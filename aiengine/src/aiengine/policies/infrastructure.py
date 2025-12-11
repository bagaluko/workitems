root@balu-AHV:/oldaiengine/policies# cat infrastructure.py
# policies\infrastructure.py

"""
Secure Infrastructure Optimization Policy
"""

import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Import base policy class with fallback
try:
    from .base import SecureAIPolicyBase
except ImportError:
    logger.warning("Base policy not available - creating fallback")

    class SecureAIPolicyBase:
        """Fallback base policy class"""
        def __init__(self, name: str, state_dim: int, device: str = 'cpu'):
            self.name = name
            self.state_dim = state_dim
            self.device = device
            self.performance_history = deque(maxlen=1000)
            self.replay_buffer = deque(maxlen=10000)
            self.training_step = 0
            self.max_training_iterations = 10000

            # Circuit breaker fallback
            self.circuit_breaker = type('CircuitBreaker', (), {
                'call': lambda self, func, *args, **kwargs: func(*args, **kwargs)
            })()

        def select_action(self, state):
            """Override in subclass"""
            raise NotImplementedError

        def update(self, state, action, result, next_state):
            """Override in subclass"""
            pass

        def save_model(self, path: str):
            """Override in subclass"""
            pass

        def load_model(self, path: str):
            """Override in subclass"""
            pass

# Import data structures with fallbacks
try:
    from models.data_structures import Action, ActionType, SystemType, EventSeverity, SystemState, ActionResult
except ImportError:
    logger.warning("Data structures not available - using fallbacks", safe_float_convert)

    class ActionType:
        NO_ACTION = "no_action"
        SCALE_UP_VERTICAL = "scale_up_vertical"
        SCALE_DOWN_VERTICAL = "scale_down_vertical"
        SCALE_OUT_HORIZONTAL = "scale_out_horizontal"
        SCALE_IN_HORIZONTAL = "scale_in_horizontal"
        ENABLE_AUTO_SCALING = "enable_auto_scaling"
        OPTIMIZE_RESOURCE_ALLOCATION = "optimize_resource_allocation"
        TUNE_APPLICATION_PARAMS = "tune_application_params"
        OPTIMIZE_DATABASE_CONFIG = "optimize_database_config"
        ADJUST_CACHE_STRATEGY = "adjust_cache_strategy"
        OPTIMIZE_LOAD_BALANCING = "optimize_load_balancing"
        HUMAN_ESCALATION = "human_escalation"

    class SystemType:
        INFRASTRUCTURE = "infrastructure"
        CICD_PIPELINE = "cicd_pipeline"

    class EventSeverity:
        LOW = 0
        MEDIUM = 1
        HIGH = 2
        CRITICAL = 3

    class Action:
        def __init__(self, action_type, target_services=None, parameters=None,
                     confidence=0.5, expected_impact=0.0, estimated_cost=0.0,
                     risk_score=0.5, execution_time_estimate=0):
            self.action_type = action_type
            self.target_services = target_services or []
            self.parameters = parameters or {}
            self.confidence = confidence
            self.expected_impact = expected_impact
            self.estimated_cost = estimated_cost
            self.risk_score = risk_score
            self.execution_time_estimate = execution_time_estimate

    class SystemState:
        def __init__(self):
            self.affected_services = []
            self.severity = EventSeverity.MEDIUM
            self.is_business_hours = True
            self.deployment_in_progress = False
            self.maintenance_window = False

            # Mock metrics - ADD MISSING ATTRIBUTES
            self.metrics = type('Metrics', (), {
                'cpu_utilization': 0.5,
                'memory_usage': 0.5,
                'disk_io': 0.3,  # ADD THIS
                'network_io': 0.3,  # ADD THIS
                'throughput': 500,
                'response_time': 500,
                'error_rate': 0.01,  # ADD THIS
                'availability': 0.99,  # ADD THIS
                'cache_hit_rate': 0.8,
                'active_connections': 50,
                'queue_depth': 10,  # ADD THIS
                'build_duration': 20.0,  # ADD THIS
                'test_duration': 15.0,  # ADD THIS
                'deployment_duration': 10.0,  # ADD THIS
                'success_rate': 0.95,  # ADD THIS
                'cost_per_hour': 50.0,  # ADD THIS
                'user_satisfaction': 0.8  # ADD THIS
            })()

        def to_feature_vector(self):
            """Convert state to feature vector for neural networks"""
            import numpy as np

            # Create a comprehensive feature vector
            features = [
                # Resource metrics
                self.metrics.cpu_utilization,
                self.metrics.memory_usage,
                getattr(self.metrics, 'disk_io', 0.3),  # Safe access
                getattr(self.metrics, 'network_io', 0.3),  # Safe access

                # Performance metrics
                min(self.metrics.response_time / 1000.0, 10.0),  # Normalize
                min(self.metrics.throughput / 1000.0, 100.0),   # Normalize
                getattr(self.metrics, 'error_rate', 0.01),  # Safe access
                getattr(self.metrics, 'availability', 0.99),  # Safe access

                # Application metrics
                min(self.metrics.active_connections / 100.0, 10.0),  # Normalize
                min(getattr(self.metrics, 'queue_depth', 10) / 50.0, 10.0),  # Safe access
                self.metrics.cache_hit_rate,

                # CI/CD metrics
                min(getattr(self.metrics, 'build_duration', 20.0) / 60.0, 5.0),
                min(getattr(self.metrics, 'test_duration', 15.0) / 60.0, 2.0),
                min(getattr(self.metrics, 'deployment_duration', 10.0) / 60.0, 1.0),
                getattr(self.metrics, 'success_rate', 0.95),

                # Business metrics
                min(getattr(self.metrics, 'cost_per_hour', 50.0) / 100.0, 10.0),
                getattr(self.metrics, 'user_satisfaction', 0.8),

                # Context features - FIXED
                1.0 / 3.0,  # severity normalized (fixed)
                12.0 / 23.0,  # time_of_day (normalized to noon)
                1.0 / 6.0,    # day_of_week (normalized to Monday)
                safe_float_convert(self.is_business_hours),
                2.0 / 10.0,   # recent_incidents (normalized)
                min(len(self.affected_services) / 10.0, 1.0),  # Normalize
                safe_float_convert(self.deployment_in_progress),
                safe_float_convert(self.maintenance_window),
                0.0,  # high_traffic_expected

                # Historical trend features (placeholders)
                0.0, 0.0, 0.0, 0.0, 0.0,  # 5 trend features

                # System type one-hot encoding (infrastructure=1, cicd=0, others=0)
                1.0, 0.0, 0.0, 0.0, 0.0   # 5 system type features
            ]

            # Ensure we have exactly 36 features (matching state_dim)
            while len(features) < 36:
                features.append(0.0)

            return np.array(features[:36], dtype=np.float32)

    class ActionResult:
        def __init__(self, success=True):
            self.success = success

        def calculate_reward(self):
            return 1.0 if self.success else -1.0

# Import neural networks with fallbacks
try:
    from models.neural_networks import ActionPolicyNetwork, ValueNetwork, AnomalyDetector
except ImportError:
    logger.warning("Neural networks not available - using fallbacks")

    class ActionPolicyNetwork:
        def __init__(self, state_dim, num_actions):
            pass
        def to(self, device):
            return self
        def parameters(self):
            return []
        def state_dict(self):
            return {}
        def load_state_dict(self, state_dict):
            pass
        def __call__(self, x):
            # Return dummy outputs: logits, value, confidence
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return (np.random.random((batch_size, 11)),
                   np.random.random((batch_size, 1)),
                   np.random.random((batch_size, 1)))

    class ValueNetwork:
        def __init__(self, state_dim):
            pass
        def to(self, device):
            return self
        def parameters(self):
            return []
        def state_dict(self):
            return {}
        def load_state_dict(self, state_dict):
            pass
        def __call__(self, x):
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return np.random.random((batch_size, 1))

    class AnomalyDetector:
        def __init__(self, state_dim):
            pass
        def to(self, device):
            return self
        def parameters(self):
            return []
        def state_dict(self):
            return {}
        def load_state_dict(self, state_dict):
            pass
        def __call__(self, x):
            return x

# PyTorch imports with fallbacks
try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - using fallbacks")
    TORCH_AVAILABLE = False

    class torch:
        @staticmethod
        def FloatTensor(data):
            return np.array(data)

        @staticmethod
        def LongTensor(data):
            return np.array(data)

        @staticmethod
        def multinomial(probs, num_samples):
            return type('Result', (), {'item': lambda: np.random.randint(0, len(probs))})()

        @staticmethod
        def isnan(x):
            return np.isnan(x) if isinstance(x, np.ndarray) else False

        @staticmethod
        def isinf(x):
            return np.isinf(x) if isinstance(x, np.ndarray) else False

        class nn:
            class utils:
                @staticmethod
                def clip_grad_norm_(parameters, max_norm):
                    pass

    class F:
        @staticmethod
        def softmax(x, dim=-1):
            if isinstance(x, np.ndarray):
                exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
                return exp_x / np.sum(exp_x, axis=dim, keepdims=True)
            return x

        @staticmethod
        def log_softmax(x, dim=-1):
            return np.log(F.softmax(x, dim) + 1e-8)

        @staticmethod
        def mse_loss(input, target):
            return 0.0

    class optim:
        class Adam:
            def __init__(self, parameters, lr=0.001, weight_decay=0):
                pass
            def zero_grad(self):
                pass
            def step(self):
                pass

# Scikit-learn imports with fallbacks
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.warning("Scikit-learn not available - using fallback")

    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return self

        def transform(self, X):
            if self.mean_ is not None and self.scale_ is not None:
                return (X - self.mean_) / (self.scale_ + 1e-8)
            return X

class SecureInfrastructureOptimizationPolicy(SecureAIPolicyBase):
    """Secure AI policy for infrastructure optimization"""

    def __init__(self, state_dim: int, device: str = 'cpu'):
        super().__init__("SecureInfrastructureOptimization", state_dim, device)

        # Infrastructure-specific actions
        self.infrastructure_actions = [
            ActionType.SCALE_UP_VERTICAL,
            ActionType.SCALE_DOWN_VERTICAL,
            ActionType.SCALE_OUT_HORIZONTAL,
            ActionType.SCALE_IN_HORIZONTAL,
            ActionType.ENABLE_AUTO_SCALING,
            ActionType.OPTIMIZE_RESOURCE_ALLOCATION,
            ActionType.TUNE_APPLICATION_PARAMS,
            ActionType.OPTIMIZE_DATABASE_CONFIG,
            ActionType.ADJUST_CACHE_STRATEGY,
            ActionType.OPTIMIZE_LOAD_BALANCING,
            ActionType.NO_ACTION
        ]

        self.num_actions = len(self.infrastructure_actions)

        # Neural networks with security enhancements
        try:
            self.policy_network = ActionPolicyNetwork(state_dim, self.num_actions).to(self.device)
            self.value_network = ValueNetwork(state_dim).to(self.device)
            self.anomaly_detector = AnomalyDetector(state_dim).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize neural networks: {e}")
            raise

        # Optimizers with gradient clipping
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001, weight_decay=1e-5)
        self.anomaly_optimizer = optim.Adam(self.anomaly_detector.parameters(), lr=0.0005, weight_decay=1e-5)

        # Experience replay buffer with size limit
        self.replay_buffer = deque(maxlen=10000)

        # Preprocessing with validation
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Training metrics
        self.training_iterations = 0
        self.last_training_time = time.time()

        logger.info(f"✅ {self.name} policy initialized with {self.num_actions} actions")

    def select_action(self, state: SystemState) -> Action:
        """Select action using secure neural network policy"""

        try:
            return self.circuit_breaker.call(self._select_action_internal, state)
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Return safe fallback action
            return Action(
                action_type=ActionType.NO_ACTION,
                target_services=[],
                parameters={'reason': 'selection_failed', 'error': str(e)},
                confidence=0.0,
                expected_impact=0.0,
                estimated_cost=0.0,
                risk_score=1.0,
                execution_time_estimate=0
            )

    def _select_action_internal(self, state: SystemState) -> Action:
        """Internal action selection logic"""

        # Convert state to feature vector with validation
        try:
            state_vector = state.to_feature_vector()
        except Exception as e:
            logger.error(f"Failed to convert state to feature vector: {e}")
            raise

        # Validate feature vector
        if np.isnan(state_vector).any() or np.isinf(state_vector).any():
            raise ValueError("Invalid feature vector: contains NaN or Inf values")

        # Normalize features
        if self.scaler_fitted:
            try:
                state_vector = self.scaler.transform(state_vector.reshape(1, -1)).flatten()
            except Exception as e:
                logger.warning(f"Feature scaling failed: {e}")
                # Continue without scaling

        # Convert to tensor with validation
        try:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Failed to create state tensor: {e}")
            raise

        # Get policy output with timeout
        try:
            with torch.no_grad():
                policy_logits, value, confidence = self.policy_network(state_tensor)

                # Validate outputs
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    raise ValueError("Invalid policy logits")

                # Apply temperature scaling for exploration
                temperature = self._get_exploration_temperature(state)
                policy_probs = F.softmax(policy_logits / temperature, dim=-1)

                # Sample action with validation
                action_idx = torch.multinomial(policy_probs, 1).item()
                if not (0 <= action_idx < len(self.infrastructure_actions)):
                    raise ValueError(f"Invalid action index: {action_idx}")

                action_type = self.infrastructure_actions[action_idx]

                # Get confidence and value estimates
                confidence_score = safe_float_convert(confidence.item())
                value_estimate = safe_float_convert(value.item())

                # Validate outputs
                confidence_score = np.clip(confidence_score, 0.0, 1.0)
                value_estimate = np.clip(value_estimate, -100.0, 100.0)

        except Exception as e:
            logger.error(f"Neural network inference failed: {e}")
            raise

        # Generate action parameters based on state and action type
        try:
            parameters = self._generate_action_parameters(state, action_type)
        except Exception as e:
            logger.error(f"Parameter generation failed: {e}")
            parameters = {}

        # Calculate expected impact and risk with validation
        try:
            expected_impact = self._estimate_impact(state, action_type, parameters)
            risk_score = self._calculate_risk_score(state, action_type, parameters)
            estimated_cost = self._estimate_cost(action_type, parameters)

            # Validate calculated values
            expected_impact = np.clip(expected_impact, 0.0, 1.0)
            risk_score = np.clip(risk_score, 0.0, 1.0)
            estimated_cost = max(0.0, estimated_cost)

        except Exception as e:
            logger.error(f"Impact/risk calculation failed: {e}")
            expected_impact = 0.1
            risk_score = 0.8
            estimated_cost = 0.0

        try:
            action = Action(
                action_type=action_type,
                target_services=state.affected_services[:10],  # Limit services
                parameters=parameters,
                confidence=confidence_score,
                expected_impact=expected_impact,
                estimated_cost=estimated_cost,
                risk_score=risk_score,
                execution_time_estimate=self._estimate_execution_time(action_type)
            )

            return action

        except Exception as e:
            logger.error(f"Action creation failed: {e}")
            raise

    def _get_exploration_temperature(self, state: SystemState) -> float:
        """Dynamic exploration temperature based on state"""
        base_temp = 1.0

        try:
            # Higher temperature (more exploration) for:
            # - Critical situations (but less exploration)
            # - New/unseen states
            # - Low confidence scenarios

            if state.severity == EventSeverity.CRITICAL:
                base_temp *= 0.5  # Less exploration for critical issues
            elif state.severity == EventSeverity.LOW:
                base_temp *= 1.5  # More exploration for low severity

            # Adjust based on recent performance
            if len(self.performance_history) > 10:
                recent_performance = list(self.performance_history)[-10:]
                recent_success_rate = sum(recent_performance) / len(recent_performance)
                if recent_success_rate < 0.7:
                    base_temp *= 1.3  # More exploration if recent performance is poor

            return np.clip(base_temp, 0.1, 3.0)

        except Exception as e:
            logger.warning(f"Temperature calculation failed: {e}")
            return 1.0

    def _generate_action_parameters(self, state: SystemState, action_type: ActionType) -> Dict[str, Any]:
        """Generate intelligent action parameters based on state"""

        parameters = {}

        try:
            if action_type == ActionType.SCALE_UP_VERTICAL:
                # Intelligent scaling based on current utilization
                cpu_util = state.metrics.cpu_utilization
                mem_util = state.metrics.memory_usage

                if cpu_util > 0.9 or mem_util > 0.9:
                    scale_factor = 2.0
                elif cpu_util > 0.8 or mem_util > 0.8:
                    scale_factor = 1.5
                else:
                    scale_factor = 1.2

                parameters = {
                    "scale_factor": min(scale_factor, 3.0),  # Cap scaling
                    "resource_type": "cpu_memory",
                    "gradual_scaling": True,
                    "max_instances": 10
                }

            elif action_type == ActionType.SCALE_OUT_HORIZONTAL:
                current_load = min(state.metrics.throughput / 1000.0, 10.0)  # Cap load
                target_instances = min(8, max(2, int(current_load * 2)))

                parameters = {
                    "target_instances": target_instances,
                    "scaling_policy": "gradual",
                    "health_check_grace_period": 300
                }

            elif action_type == ActionType.OPTIMIZE_DATABASE_CONFIG:
                parameters = {
                    "connection_pool_size": min(100, max(10, state.metrics.active_connections * 2)),
                    "query_timeout": 30000,
                    "cache_size": "512MB",
                    "enable_query_optimization": True
                }

            elif action_type == ActionType.ADJUST_CACHE_STRATEGY:
                hit_rate = state.metrics.cache_hit_rate
                if hit_rate < 0.7:
                    parameters = {
                        "cache_size_increase": min(2.0, 1.5),
                        "ttl_adjustment": min(2.0, 1.2),
                        "eviction_policy": "LRU",
                        "preload_strategy": "predictive"
                    }
                else:
                    parameters = {
                        "cache_optimization": "fine_tune",
                        "ttl_adjustment": 0.9
                    }

            elif action_type == ActionType.ENABLE_AUTO_SCALING:
                parameters = {
                    "min_instances": 2,
                    "max_instances": 10,
                    "target_cpu_utilization": 70,
                    "scale_up_cooldown": 300,
                    "scale_down_cooldown": 600
                }

            # Validate parameters size
            if len(json.dumps(parameters)) > 5000:  # 5KB limit
                logger.warning("Parameters too large, truncating")
                parameters = {"truncated": True}

        except Exception as e:
            logger.warning(f"Parameter generation error: {e}")
            parameters = {"error": "parameter_generation_failed"}

        return parameters

    def _estimate_impact(self, state: SystemState, action_type: ActionType, parameters: Dict[str, Any]) -> float:
        """Estimate expected impact of action"""

        try:
            base_impact = 0.5

            # Impact based on action type and current state
            if action_type == ActionType.SCALE_UP_VERTICAL:
                if state.metrics.cpu_utilization > 0.8:
                    base_impact = 0.8
                elif state.metrics.memory_usage > 0.8:
                    base_impact = 0.7

            elif action_type == ActionType.SCALE_OUT_HORIZONTAL:
                if state.metrics.throughput > 800:
                    base_impact = 0.9

            elif action_type == ActionType.OPTIMIZE_DATABASE_CONFIG:
                if state.metrics.response_time > 1000:
                    base_impact = 0.7

            elif action_type == ActionType.ADJUST_CACHE_STRATEGY:
                if state.metrics.cache_hit_rate < 0.6:
                    base_impact = 0.8

            elif action_type == ActionType.NO_ACTION:
                base_impact = 0.1

            # Adjust based on severity
            severity_multiplier = {
                EventSeverity.LOW: 0.8,
                EventSeverity.MEDIUM: 1.0,
                EventSeverity.HIGH: 1.2,
                EventSeverity.CRITICAL: 1.5
            }

            impact = base_impact * severity_multiplier.get(state.severity, 1.0)
            return np.clip(impact, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Impact estimation error: {e}")
            return 0.5

    def _calculate_risk_score(self, state: SystemState, action_type: ActionType, parameters: Dict[str, Any]) -> float:
        """Calculate risk score for action"""

        try:
            base_risk = 0.3

            # Risk based on action type
            risk_map = {
                ActionType.SCALE_UP_VERTICAL: 0.2,
                ActionType.SCALE_DOWN_VERTICAL: 0.4,
                ActionType.SCALE_OUT_HORIZONTAL: 0.3,
                ActionType.SCALE_IN_HORIZONTAL: 0.5,
                ActionType.OPTIMIZE_DATABASE_CONFIG: 0.6,
                ActionType.TUNE_APPLICATION_PARAMS: 0.7,
                ActionType.NO_ACTION: 0.0
            }

            base_risk = risk_map.get(action_type, 0.5)

            # Increase risk during business hours
            if state.is_business_hours:
                base_risk *= 1.3

            # Increase risk if deployment in progress
            if state.deployment_in_progress:
                base_risk *= 1.5

            # Decrease risk during maintenance window
            if state.maintenance_window:
                base_risk *= 0.7

            return np.clip(base_risk, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Risk calculation error: {e}")
            return 0.5

    def _estimate_cost(self, action_type: ActionType, parameters: Dict[str, Any]) -> float:
        """Estimate cost of action"""

        try:
            cost_map = {
                ActionType.SCALE_UP_VERTICAL: 50.0,
                ActionType.SCALE_OUT_HORIZONTAL: 100.0,
                ActionType.SCALE_DOWN_VERTICAL: -20.0,
                ActionType.SCALE_IN_HORIZONTAL: -40.0,
                ActionType.OPTIMIZE_DATABASE_CONFIG: 10.0,
                ActionType.ADJUST_CACHE_STRATEGY: 15.0,
                ActionType.NO_ACTION: 0.0
            }

            base_cost = cost_map.get(action_type, 25.0)

            # Adjust based on parameters
            if action_type == ActionType.SCALE_UP_VERTICAL:
                scale_factor = parameters.get("scale_factor", 1.0)
                base_cost *= min(scale_factor, 5.0)  # Cap cost multiplier

            elif action_type == ActionType.SCALE_OUT_HORIZONTAL:
                target_instances = parameters.get("target_instances", 1)
                base_cost *= min(target_instances, 10)  # Cap instances

            return max(0.0, base_cost)

        except Exception as e:
            logger.warning(f"Cost estimation error: {e}")
            return 25.0

    def _estimate_execution_time(self, action_type: ActionType) -> int:
        """Estimate execution time in minutes"""

        time_map = {
            ActionType.SCALE_UP_VERTICAL: 3,
            ActionType.SCALE_DOWN_VERTICAL: 2,
            ActionType.SCALE_OUT_HORIZONTAL: 5,
            ActionType.SCALE_IN_HORIZONTAL: 3,
            ActionType.OPTIMIZE_DATABASE_CONFIG: 8,
            ActionType.ADJUST_CACHE_STRATEGY: 4,
            ActionType.TUNE_APPLICATION_PARAMS: 10,
            ActionType.NO_ACTION: 0
        }

        return time_map.get(action_type, 5)

    def update(self, state: SystemState, action: Action, result: ActionResult, next_state: SystemState):
        """Update policy based on experience with security checks"""

        try:
            # Validate inputs
            if not isinstance(result, ActionResult):
                raise ValueError("Invalid action result")

            # Add to replay buffer with size check
            if len(self.replay_buffer) >= 10000:
                logger.warning("Replay buffer full, removing oldest entries")

            experience = {
                'state': state.to_feature_vector(),
                'action': self.infrastructure_actions.index(action.action_type),
                'reward': result.calculate_reward(),
                'next_state': next_state.to_feature_vector(),
                'done': True,
                'timestamp': time.time()
            }

            # Validate experience data
            if (np.isnan(experience['state']).any() or
                np.isnan(experience['next_state']).any() or
                np.isnan(experience['reward'])):
                logger.warning("Invalid experience data, skipping update")
                return

            self.replay_buffer.append(experience)

            # Update performance history
            self.performance_history.append(1.0 if result.success else 0.0)

            # Train networks if enough data and not too frequent
            current_time = time.time()
            if (len(self.replay_buffer) >= 32 and
                current_time - self.last_training_time > 60):  # At least 1 minute between training

                self._train_networks()
                self.last_training_time = current_time

        except Exception as e:
            logger.error(f"Policy update failed: {e}")

    def _train_networks(self, batch_size: int = 32):
        """Train neural networks using experience replay with security checks"""

        try:
            if len(self.replay_buffer) < batch_size:
                return

            # Check training limits
            if self.training_iterations >= self.max_training_iterations:
                logger.warning("Maximum training iterations reached")
                return

            # Sample batch
            batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in batch_indices]

            # Prepare tensors with validation
            try:
                states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
                actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
                rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
                next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)

                # Validate tensors
                for tensor in [states, next_states, rewards]:
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logger.warning("Invalid training data, skipping batch")
                        return

            except Exception as e:
                logger.error(f"Failed to prepare training tensors: {e}")
                return

            # Fit scaler if not fitted
            if not self.scaler_fitted:
                try:
                    all_states = np.vstack([states.cpu().numpy(), next_states.cpu().numpy()])
                    self.scaler.fit(all_states)
                    self.scaler_fitted = True
                except Exception as e:
                    logger.warning(f"Scaler fitting failed: {e}")

            # Normalize states
            if self.scaler_fitted:
                try:
                    states_np = self.scaler.transform(states.cpu().numpy())
                    next_states_np = self.scaler.transform(next_states.cpu().numpy())

                    states = torch.FloatTensor(states_np).to(self.device)
                    next_states = torch.FloatTensor(next_states_np).to(self.device)
                except Exception as e:
                    logger.warning(f"State normalization failed: {e}")

            # Train value network
            try:
                current_values = self.value_network(states).squeeze()
                next_values = self.value_network(next_states).squeeze()
                target_values = rewards + 0.99 * next_values  # Discount factor = 0.99

                value_loss = F.mse_loss(current_values, target_values.detach())

                # Gradient clipping
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                self.value_optimizer.step()

            except Exception as e:
                logger.warning(f"Value network training failed: {e}")

            # Train policy network
            try:
                policy_logits, _, confidence = self.policy_network(states)

                # Calculate advantages
                advantages = (target_values - current_values).detach()

                # Policy loss (PPO-style)
                log_probs = F.log_softmax(policy_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

                policy_loss = -(selected_log_probs * advantages).mean()

                # Confidence loss (encourage high confidence for good actions)
                confidence_targets = torch.sigmoid(advantages)  # Convert advantages to [0,1]
                confidence_loss = F.mse_loss(confidence.squeeze(), confidence_targets)

                total_policy_loss = policy_loss + 0.1 * confidence_loss

                # Gradient clipping
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                self.policy_optimizer.step()

            except Exception as e:
                logger.warning(f"Policy network training failed: {e}")

            # Train anomaly detector (unsupervised)
            try:
                anomaly_loss = F.mse_loss(self.anomaly_detector(states), states)

                self.anomaly_optimizer.zero_grad()
                anomaly_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.anomaly_detector.parameters(), max_norm=1.0)
                self.anomaly_optimizer.step()

            except Exception as e:
                logger.warning(f"Anomaly detector training failed: {e}")

            self.training_iterations += 1

        except Exception as e:
            logger.error(f"Network training failed: {e}")

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for secure saving"""
        return {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'anomaly_detector': self.anomaly_detector.state_dict(),
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'performance_history': list(self.performance_history),
            'replay_buffer': list(self.replay_buffer)[-1000:],  # Limit size
            'training_iterations': self.training_iterations,
            'infrastructure_actions': [action.value for action in self.infrastructure_actions]
        }

    def _load_model_state(self, model_state: Dict[str, Any]):
        """Load model state securely"""
        try:
            self.policy_network.load_state_dict(model_state['policy_network'])
            self.value_network.load_state_dict(model_state['value_network'])
            self.anomaly_detector.load_state_dict(model_state['anomaly_detector'])
            self.scaler = model_state['scaler']
            self.scaler_fitted = model_state['scaler_fitted']
            self.performance_history = deque(model_state['performance_history'], maxlen=1000)
            self.replay_buffer = deque(model_state['replay_buffer'], maxlen=10000)
            self.training_iterations = model_state.get('training_iterations', 0)

            # Validate loaded actions match current actions
            loaded_actions = model_state.get('infrastructure_actions', [])
            current_actions = [action.value for action in self.infrastructure_actions]
            if loaded_actions != current_actions:
                logger.warning("Loaded actions don't match current actions")

        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            raise
