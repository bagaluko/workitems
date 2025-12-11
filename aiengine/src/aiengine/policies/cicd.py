# policies\cicd.py

"""
Secure CI/CD Optimization Policy
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

# Import data structures with fallbacks
try:
    from models.data_structures import Action, ActionType, SystemType, EventSeverity, SystemState, ActionResult
except ImportError:
    logger.warning("Data structures not available - using fallbacks", safe_float_convert)

    class ActionType:
        NO_ACTION = "no_action"
        OPTIMIZE_BUILD_PIPELINE = "optimize_build_pipeline"
        PARALLELIZE_TESTS = "parallelize_tests"
        OPTIMIZE_DEPLOYMENT_STRATEGY = "optimize_deployment_strategy"
        ENABLE_SMART_CACHING = "enable_smart_caching"
        ADJUST_RESOURCE_LIMITS = "adjust_resource_limits"
        OPTIMIZE_ARTIFACT_MANAGEMENT = "optimize_artifact_management"
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
            self.deployment_in_progress = False

            # Mock metrics for CI/CD
            self.metrics = type('Metrics', (), {
                'build_duration': 20.0,
                'test_duration': 15.0,
                'deployment_duration': 10.0,
                'success_rate': 0.95
            })()

        def to_feature_vector(self):
            return np.random.random(50)  # Fallback feature vector

    class ActionResult:
        def __init__(self, success=True):
            self.success = success

        def calculate_reward(self):
            return 1.0 if self.success else -1.0

# Import neural networks with fallbacks
try:
    from models.neural_networks import ActionPolicyNetwork, ValueNetwork
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
            return (np.random.random((batch_size, 7)),
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

class SecureCICDOptimizationPolicy(SecureAIPolicyBase):
    """Secure AI policy for CI/CD pipeline optimization"""

    def __init__(self, state_dim: int, device: str = 'cpu'):
        super().__init__("SecureCICDOptimization", state_dim, device)

        # CI/CD-specific actions
        self.cicd_actions = [
            ActionType.OPTIMIZE_BUILD_PIPELINE,
            ActionType.PARALLELIZE_TESTS,
            ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY,
            ActionType.ENABLE_SMART_CACHING,
            ActionType.ADJUST_RESOURCE_LIMITS,
            ActionType.OPTIMIZE_ARTIFACT_MANAGEMENT,
            ActionType.NO_ACTION
        ]

        self.num_actions = len(self.cicd_actions)

        # Neural networks with security enhancements
        try:
            self.policy_network = ActionPolicyNetwork(state_dim, self.num_actions).to(self.device)
            self.value_network = ValueNetwork(state_dim).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize CI/CD neural networks: {e}")
            raise

        # Optimizers with gradient clipping
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001, weight_decay=1e-5)

        # Experience replay with size limit
        self.replay_buffer = deque(maxlen=5000)

        # Preprocessing with validation
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Training metrics
        self.training_iterations = 0
        self.last_training_time = time.time()

        logger.info(f"✅ {self.name} policy initialized with {self.num_actions} actions")

    def select_action(self, state: SystemState) -> Action:
        """Select CI/CD optimization action securely"""

        try:
            return self.circuit_breaker.call(self._select_action_internal, state)
        except Exception as e:
            logger.error(f"CI/CD action selection failed: {e}")
            # Return safe fallback action
            return Action(
                action_type=ActionType.NO_ACTION,
                target_services=[],
                parameters={'reason': 'cicd_selection_failed', 'error': str(e)},
                confidence=0.0,
                expected_impact=0.0,
                estimated_cost=0.0,
                risk_score=1.0,
                execution_time_estimate=0
            )

    def _select_action_internal(self, state: SystemState) -> Action:
        """Internal CI/CD action selection logic"""

        # Convert state to feature vector with validation
        try:
            state_vector = state.to_feature_vector()
        except Exception as e:
            logger.error(f"Failed to convert CI/CD state to feature vector: {e}")
            raise

        # Validate feature vector
        if np.isnan(state_vector).any() or np.isinf(state_vector).any():
            raise ValueError("Invalid CI/CD feature vector: contains NaN or Inf values")

        # Normalize features
        if self.scaler_fitted:
            try:
                state_vector = self.scaler.transform(state_vector.reshape(1, -1)).flatten()
            except Exception as e:
                logger.warning(f"CI/CD feature scaling failed: {e}")

        # Convert to tensor
        try:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Failed to create CI/CD state tensor: {e}")
            raise

        # Get policy output
        try:
            with torch.no_grad():
                policy_logits, value, confidence = self.policy_network(state_tensor)

                # Validate outputs
                if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                    raise ValueError("Invalid CI/CD policy logits")

                # Apply temperature scaling (less exploration for CI/CD)
                temperature = 0.8
                policy_probs = F.softmax(policy_logits / temperature, dim=-1)

                # Sample action
                action_idx = torch.multinomial(policy_probs, 1).item()
                if not (0 <= action_idx < len(self.cicd_actions)):
                    raise ValueError(f"Invalid CI/CD action index: {action_idx}")

                action_type = self.cicd_actions[action_idx]
                confidence_score = np.clip(safe_float_convert(confidence.item()), 0.0, 1.0)

        except Exception as e:
            logger.error(f"CI/CD neural network inference failed: {e}")
            raise

        # Generate CI/CD-specific parameters
        try:
            parameters = self._generate_cicd_parameters(state, action_type)
        except Exception as e:
            logger.error(f"CI/CD parameter generation failed: {e}")
            parameters = {}

        # Calculate metrics
        try:
            expected_impact = self._estimate_cicd_impact(state, action_type)
            risk_score = self._calculate_cicd_risk(state, action_type)
            estimated_cost = self._estimate_cicd_cost(action_type, parameters)

            # Validate calculated values
            expected_impact = np.clip(expected_impact, 0.0, 1.0)
            risk_score = np.clip(risk_score, 0.0, 1.0)
            estimated_cost = max(0.0, estimated_cost)

        except Exception as e:
            logger.error(f"CI/CD impact/risk calculation failed: {e}")
            expected_impact = 0.1
            risk_score = 0.5
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
                execution_time_estimate=self._estimate_cicd_time(action_type)
            )

            return action

        except Exception as e:
            logger.error(f"CI/CD action creation failed: {e}")
            raise

    def _generate_cicd_parameters(self, state: SystemState, action_type: ActionType) -> Dict[str, Any]:
        """Generate CI/CD-specific parameters securely"""

        parameters = {}

        try:
            if action_type == ActionType.OPTIMIZE_BUILD_PIPELINE:
                build_duration = state.metrics.build_duration or 20.0

                if build_duration > 30:
                    parameters = {
                        "enable_parallel_builds": True,
                        "parallel_jobs": min(8, 4),  # Cap parallel jobs
                        "cache_strategy": "aggressive",
                        "incremental_builds": True,
                        "build_optimization_level": "high"
                    }
                else:
                    parameters = {
                        "enable_parallel_builds": True,
                        "parallel_jobs": min(4, 2),
                        "cache_strategy": "moderate",
                        "build_optimization_level": "medium"
                    }

            elif action_type == ActionType.PARALLELIZE_TESTS:
                test_duration = state.metrics.test_duration or 15.0

                parameters = {
                    "parallel_test_runners": min(8, max(2, int(test_duration / 5))),
                    "test_sharding": True,
                    "smart_test_selection": True,
                    "test_prioritization": "risk_based"
                }

            elif action_type == ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY:
                parameters = {
                    "deployment_strategy": "blue_green" if state.severity == EventSeverity.CRITICAL else "rolling",
                    "health_check_timeout": min(600, 300),  # Cap timeout
                    "rollback_threshold": max(0.01, 0.05),  # Minimum threshold
                    "canary_percentage": min(50, 10)  # Cap canary percentage
                }

            elif action_type == ActionType.ENABLE_SMART_CACHING:
                parameters = {
                    "cache_layers": ["dependencies", "build_artifacts", "test_results"],
                    "cache_ttl": "24h",
                    "cache_strategy": "content_hash",
                    "distributed_cache": True
                }

            elif action_type == ActionType.ADJUST_RESOURCE_LIMITS:
                parameters = {
                    "cpu_limit": "2000m",
                    "memory_limit": "4Gi",
                    "auto_scaling": True,
                    "resource_requests": "optimized"
                }

            # Validate parameters size
            if len(json.dumps(parameters)) > 3000:  # 3KB limit for CI/CD
                logger.warning("CI/CD parameters too large, truncating")
                parameters = {"truncated": True}

        except Exception as e:
            logger.warning(f"CI/CD parameter generation error: {e}")
            parameters = {"error": "cicd_parameter_generation_failed"}

        return parameters

    def _estimate_cicd_impact(self, state: SystemState, action_type: ActionType) -> float:
        """Estimate CI/CD optimization impact"""

        try:
            base_impact = 0.5

            if action_type == ActionType.OPTIMIZE_BUILD_PIPELINE:
                build_duration = state.metrics.build_duration or 20.0
                if build_duration > 30:
                    base_impact = 0.8
                elif build_duration > 20:
                    base_impact = 0.6

            elif action_type == ActionType.PARALLELIZE_TESTS:
                test_duration = state.metrics.test_duration or 15.0
                if test_duration > 25:
                    base_impact = 0.9
                elif test_duration > 15:
                    base_impact = 0.7

            elif action_type == ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY:
                if state.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                    base_impact = 0.8

            return np.clip(base_impact, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"CI/CD impact estimation error: {e}")
            return 0.5

    def _calculate_cicd_risk(self, state: SystemState, action_type: ActionType) -> float:
        """Calculate CI/CD risk score"""

        try:
            base_risk = 0.2  # CI/CD changes are generally lower risk

            risk_map = {
                ActionType.OPTIMIZE_BUILD_PIPELINE: 0.3,
                ActionType.PARALLELIZE_TESTS: 0.2,
                ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY: 0.4,
                ActionType.ENABLE_SMART_CACHING: 0.1,
                ActionType.ADJUST_RESOURCE_LIMITS: 0.3,
                ActionType.NO_ACTION: 0.0
            }

            base_risk = risk_map.get(action_type, 0.3)

            # Increase risk if deployment in progress
            if state.deployment_in_progress:
                base_risk *= 1.8

            return np.clip(base_risk, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"CI/CD risk calculation error: {e}")
            return 0.3

    def _estimate_cicd_cost(self, action_type: ActionType, parameters: Dict[str, Any]) -> float:
        """Estimate CI/CD cost"""

        try:
            cost_map = {
                ActionType.OPTIMIZE_BUILD_PIPELINE: 20.0,
                ActionType.PARALLELIZE_TESTS: 30.0,
                ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY: 15.0,
                ActionType.ENABLE_SMART_CACHING: 25.0,
                ActionType.ADJUST_RESOURCE_LIMITS: 10.0,
                ActionType.NO_ACTION: 0.0
            }

            return max(0.0, cost_map.get(action_type, 15.0))

        except Exception as e:
            logger.warning(f"CI/CD cost estimation error: {e}")
            return 15.0

    def _estimate_cicd_time(self, action_type: ActionType) -> int:
        """Estimate CI/CD execution time"""

        time_map = {
            ActionType.OPTIMIZE_BUILD_PIPELINE: 5,
            ActionType.PARALLELIZE_TESTS: 3,
            ActionType.OPTIMIZE_DEPLOYMENT_STRATEGY: 8,
            ActionType.ENABLE_SMART_CACHING: 4,
            ActionType.ADJUST_RESOURCE_LIMITS: 2,
            ActionType.NO_ACTION: 0
        }

        return time_map.get(action_type, 4)

    def update(self, state: SystemState, action: Action, result: ActionResult, next_state: SystemState):
        """Update CI/CD policy securely"""

        try:
            # Validate inputs
            if not isinstance(result, ActionResult):
                raise ValueError("Invalid CI/CD action result")

            experience = {
                'state': state.to_feature_vector(),
                'action': self.cicd_actions.index(action.action_type),
                'reward': result.calculate_reward(),
                'next_state': next_state.to_feature_vector(),
                'done': True,
                'timestamp': time.time()
            }

            # Validate experience data
            if (np.isnan(experience['state']).any() or
                np.isnan(experience['next_state']).any() or
                np.isnan(experience['reward'])):
                logger.warning("Invalid CI/CD experience data, skipping update")
                return

            self.replay_buffer.append(experience)
            self.performance_history.append(1.0 if result.success else 0.0)

            # Train networks if enough data and not too frequent
            current_time = time.time()
            if (len(self.replay_buffer) >= 16 and
                current_time - self.last_training_time > 60):

                self._train_networks()
                self.last_training_time = current_time

        except Exception as e:
            logger.error(f"CI/CD policy update failed: {e}")

    def _train_networks(self, batch_size: int = 16):
        """Train CI/CD networks securely"""

        try:
            if len(self.replay_buffer) < batch_size:
                return

            # Check training limits
            if self.training_iterations >= self.max_training_iterations:
                logger.warning("Maximum CI/CD training iterations reached")
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
                        logger.warning("Invalid CI/CD training data, skipping batch")
                        return

            except Exception as e:
                logger.error(f"Failed to prepare CI/CD training tensors: {e}")
                return

            # Fit scaler if needed
            if not self.scaler_fitted:
                try:
                    all_states = np.vstack([states.cpu().numpy(), next_states.cpu().numpy()])
                    self.scaler.fit(all_states)
                    self.scaler_fitted = True
                except Exception as e:
                    logger.warning(f"CI/CD scaler fitting failed: {e}")

            # Normalize
            if self.scaler_fitted:
                try:
                    states_np = self.scaler.transform(states.cpu().numpy())
                    next_states_np = self.scaler.transform(next_states.cpu().numpy())

                    states = torch.FloatTensor(states_np).to(self.device)
                    next_states = torch.FloatTensor(next_states_np).to(self.device)
                except Exception as e:
                    logger.warning(f"CI/CD state normalization failed: {e}")

            # Train networks (similar to infrastructure policy but with CI/CD specific handling)
            try:
                current_values = self.value_network(states).squeeze()
                next_values = self.value_network(next_states).squeeze()
                target_values = rewards + 0.99 * next_values

                value_loss = F.mse_loss(current_values, target_values.detach())

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                self.value_optimizer.step()

            except Exception as e:
                logger.warning(f"CI/CD value network training failed: {e}")

            # Policy training
            try:
                policy_logits, _, confidence = self.policy_network(states)
                advantages = (target_values - current_values).detach()

                log_probs = F.log_softmax(policy_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

                policy_loss = -(selected_log_probs * advantages).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                self.policy_optimizer.step()

            except Exception as e:
                logger.warning(f"CI/CD policy network training failed: {e}")

            self.training_iterations += 1

        except Exception as e:
            logger.error(f"CI/CD network training failed: {e}")

    def _get_model_state(self) -> Dict[str, Any]:
        """Get CI/CD model state for secure saving"""
        return {
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'performance_history': list(self.performance_history),
            'replay_buffer': list(self.replay_buffer)[-500:],  # Limit size
            'training_iterations': self.training_iterations,
            'cicd_actions': [action.value for action in self.cicd_actions]
        }

    def _load_model_state(self, model_state: Dict[str, Any]):
        """Load CI/CD model state securely"""
        try:
            self.policy_network.load_state_dict(model_state['policy_network'])
            self.value_network.load_state_dict(model_state['value_network'])
            self.scaler = model_state['scaler']
            self.scaler_fitted = model_state['scaler_fitted']
            self.performance_history = deque(model_state['performance_history'], maxlen=1000)
            self.replay_buffer = deque(model_state['replay_buffer'], maxlen=5000)
            self.training_iterations = model_state.get('training_iterations', 0)

            # Validate loaded actions
            loaded_actions = model_state.get('cicd_actions', [])
            current_actions = [action.value for action in self.cicd_actions]
            if loaded_actions != current_actions:
                logger.warning("Loaded CI/CD actions don't match current actions")

        except Exception as e:
            logger.error(f"Failed to load CI/CD model state: {e}")
            raise
