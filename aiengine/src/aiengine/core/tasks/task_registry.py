"""
Task Registry - Centralized task type definitions and metadata
"""

from ..universal_types import TaskType
from typing import Dict, List, Any

# Task metadata and configuration
TASK_METADATA = {
    TaskType.PRECHECK_ANALYSIS: {
        'description': 'Analyze precheck requirements and make deployment decisions',
        'input_schema': {
            'test_coverage': {'type': 'float', 'range': [0.0, 1.0], 'required': True},
            'failed_tests': {'type': 'int', 'range': [0, 100], 'required': True},
            'security_issues': {'type': 'int', 'range': [0, 50], 'required': True},
            'deployment_target': {'type': 'str', 'options': ['dev', 'staging', 'production'], 'required': False}
        },
        'output_schema': {
            'decision': {'type': 'str', 'options': ['APPROVE', 'APPROVE_WITH_CONDITIONS', 'REQUIRE_MANUAL_REVIEW', 'REJECT', 'ESCALATE']},
            'confidence': {'type': 'float', 'range': [0.0, 1.0]},
            'risk_level': {'type': 'str', 'options': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']},
            'reasoning': {'type': 'str'},
            'recommendations': {'type': 'list'}
        },
        'processing_steps': ['validate_input', 'analyze_metrics', 'apply_rules', 'generate_decision'],
        'confidence_factors': ['test_coverage', 'security_status', 'historical_performance']
    },

    TaskType.ANOMALY_DETECTION: {
        'description': 'Detect anomalies in time series or structured data',
        'input_schema': {
            'metrics': {'type': 'list', 'required': True},
            'threshold': {'type': 'float', 'range': [0.0, 1.0], 'required': False},
            'window_size': {'type': 'int', 'range': [1, 100], 'required': False}
        },
        'output_schema': {
            'anomaly_detected': {'type': 'bool'},
            'anomaly_score': {'type': 'float', 'range': [0.0, 1.0]},
            'affected_metrics': {'type': 'list'},
            'severity': {'type': 'str', 'options': ['low', 'medium', 'high', 'critical']}
        },
        'processing_steps': ['normalize_data', 'apply_detection_algorithm', 'score_anomalies'],
        'confidence_factors': ['data_quality', 'pattern_consistency', 'historical_accuracy']
    },

    TaskType.CLASSIFICATION: {
        'description': 'Classify input data into predefined categories',
        'input_schema': {
            'features': {'type': 'dict', 'required': True},
            'num_classes': {'type': 'int', 'range': [2, 100], 'required': False}
        },
        'output_schema': {
            'predicted_class': {'type': 'int'},
            'class_probabilities': {'type': 'list'},
            'confidence': {'type': 'float', 'range': [0.0, 1.0]}
        },
        'processing_steps': ['feature_extraction', 'normalization', 'classification', 'probability_calculation'],
        'confidence_factors': ['feature_quality', 'model_certainty', 'class_separation']
    },

    TaskType.TIME_SERIES_FORECASTING: {
        'description': 'Predict future values based on historical time series data',
        'input_schema': {
            'time_series': {'type': 'list', 'required': True},
            'forecast_horizon': {'type': 'int', 'range': [1, 365], 'required': True},
            'seasonality': {'type': 'str', 'options': ['daily', 'weekly', 'monthly', 'yearly'], 'required': False}
        },
        'output_schema': {
            'forecast': {'type': 'list'},
            'confidence_intervals': {'type': 'dict'},
            'trend': {'type': 'str', 'options': ['increasing', 'decreasing', 'stable']},
            'accuracy_metrics': {'type': 'dict'}
        },
        'processing_steps': ['data_preprocessing', 'trend_analysis', 'forecasting', 'confidence_calculation'],
        'confidence_factors': ['data_completeness', 'trend_stability', 'seasonal_patterns']
    },

    TaskType.SENTIMENT_ANALYSIS: {
        'description': 'Analyze sentiment in text data',
        'input_schema': {
            'text': {'type': 'str', 'required': True},
            'language': {'type': 'str', 'options': ['en', 'es', 'fr', 'de'], 'required': False}
        },
        'output_schema': {
            'sentiment': {'type': 'str', 'options': ['positive', 'negative', 'neutral']},
            'confidence': {'type': 'float', 'range': [0.0, 1.0]},
            'sentiment_score': {'type': 'float', 'range': [-1.0, 1.0]},
            'key_phrases': {'type': 'list'}
        },
        'processing_steps': ['text_preprocessing', 'feature_extraction', 'sentiment_classification'],
        'confidence_factors': ['text_clarity', 'sentiment_strength', 'context_consistency']
    }
}

def get_task_config(task_type: TaskType) -> Dict[str, Any]:
    """Get configuration for a specific task type"""
    return TASK_METADATA.get(task_type, {
        'description': f'Generic {task_type.value} task',
        'input_schema': {},
        'output_schema': {},
        'processing_steps': ['process_input', 'generate_output'],
        'confidence_factors': ['data_quality']
    })

def validate_task_input(task_type: TaskType, input_data: Any) -> tuple[bool, str]:
    """Validate input data against task schema"""
    config = get_task_config(task_type)
    input_schema = config.get('input_schema', {})

    if not input_schema:
        return True, "No validation schema defined"

    try:
        if isinstance(input_data, dict):
            # Check required fields
            for field, schema in input_schema.items():
                if schema.get('required', False) and field not in input_data:
                    return False, f"Required field '{field}' missing"

                if field in input_data:
                    value = input_data[field]
                    expected_type = schema.get('type')

                    # Type validation
                    if expected_type == 'float' and not isinstance(value, (int, float)):
                        return False, f"Field '{field}' must be numeric"
                    elif expected_type == 'int' and not isinstance(value, int):
                        return False, f"Field '{field}' must be integer"
                    elif expected_type == 'str' and not isinstance(value, str):
                        return False, f"Field '{field}' must be string"
                    elif expected_type == 'list' and not isinstance(value, list):
                        return False, f"Field '{field}' must be list"

                    # Range validation
                    if 'range' in schema and isinstance(value, (int, float)):
                        min_val, max_val = schema['range']
                        if not (min_val <= value <= max_val):
                            return False, f"Field '{field}' must be between {min_val} and {max_val}"

                    # Options validation
                    if 'options' in schema and value not in schema['options']:
                        return False, f"Field '{field}' must be one of {schema['options']}"

        return True, "Validation passed"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_processing_steps(task_type: TaskType) -> List[str]:
    """Get processing steps for a task type"""
    config = get_task_config(task_type)
    return config.get('processing_steps', ['process_input', 'generate_output'])

def get_confidence_factors(task_type: TaskType) -> List[str]:
    """Get confidence factors for a task type"""
    config = get_task_config(task_type)
    return config.get('confidence_factors', ['data_quality'])

def get_all_task_types() -> List[TaskType]:
    """Get list of all configured task types"""
    return list(TASK_METADATA.keys())

def get_task_stats() -> Dict[str, int]:
    """Get statistics about configured tasks"""
    return {
        'total_tasks': len(TASK_METADATA),
        'tasks_with_validation': len([t for t in TASK_METADATA.values() if t.get('input_schema')]),
        'avg_processing_steps': sum(len(t.get('processing_steps', [])) for t in TASK_METADATA.values()) // len(TASK_METADATA)
    }
