
"""
Domain Registry - Centralized domain definitions and metadata
"""

from ..universal_types import DomainType
from typing import Dict, List, Any

# Domain metadata and configuration
DOMAIN_METADATA = {
    DomainType.PRECHECK_VALIDATION: {
        'description': 'Precheck validation and compliance analysis',
        'compatible_tasks': [
            'waiver_decision', 'exception_routing', 'compliance_validation',
            'risk_assessment', 'approval_prediction', 'precheck_analysis',
            'classification', 'decision_support'
        ],
        'output_formats': ['decision', 'confidence', 'risk_level', 'recommendations'],
        'confidence_threshold': 0.7,
        'feature_weights': {
            'test_coverage': 1.0,
            'failed_tests': 0.9,
            'security_issues': 0.95,
            'build_quality': 0.8,
            'deployment_target': 0.7
        },
        'decision_mapping': {
            0: "APPROVE",
            1: "APPROVE_WITH_CONDITIONS",
            2: "REQUIRE_MANUAL_REVIEW",
            3: "REJECT",
            4: "ESCALATE"
        }
    },

    DomainType.INFRASTRUCTURE: {
        'description': 'Infrastructure monitoring and optimization',
        'compatible_tasks': [
            'anomaly_detection', 'resource_optimization', 'monitoring',
            'classification', 'time_series_forecasting', 'control_systems'
        ],
        'output_formats': ['alert', 'recommendation', 'metrics', 'action'],
        'confidence_threshold': 0.8,
        'feature_weights': {
            'cpu': 1.0, 'memory': 0.9, 'network': 0.8, 'disk': 0.7,
            'cpu_usage': 1.0, 'memory_usage': 0.9, 'network_latency': 0.8
        },
        'action_mapping': {
            0: 'scale_up', 1: 'scale_down', 2: 'optimize', 3: 'maintain', 4: 'alert'
        }
    },

    DomainType.FINANCE: {
        'description': 'Financial analysis and prediction',
        'compatible_tasks': [
            'time_series_forecasting', 'regression', 'classification',
            'risk_assessment', 'anomaly_detection'
        ],
        'output_formats': ['prediction', 'risk_level', 'recommendation', 'confidence_interval'],
        'confidence_threshold': 0.75,
        'feature_weights': {
            'price': 1.0, 'volume': 0.8, 'risk': 0.9, 'return': 0.95,
            'volatility': 0.85, 'liquidity': 0.7
        }
    },

    DomainType.HEALTHCARE: {
        'description': 'Healthcare analysis and medical decision support',
        'compatible_tasks': [
            'classification', 'anomaly_detection', 'pattern_recognition', 'decision_support'
        ],
        'output_formats': ['assessment', 'confidence', 'recommendations', 'urgency'],
        'confidence_threshold': 0.85,
        'feature_weights': {
            'temperature': 0.9, 'heart_rate': 0.85, 'blood_pressure': 0.9,
            'symptoms': 1.0, 'age': 0.7, 'vital_signs': 0.95
        },
        'assessment_mapping': {
            0: 'normal', 1: 'mild_concern', 2: 'moderate_concern',
            3: 'severe_concern', 4: 'critical'
        }
    },

    DomainType.NATURAL_LANGUAGE: {
        'description': 'Natural language processing and analysis',
        'compatible_tasks': [
            'sentiment_analysis', 'text_generation', 'classification',
            'natural_language_processing'
        ],
        'output_formats': ['sentiment', 'confidence', 'key_phrases', 'summary'],
        'confidence_threshold': 0.7,
        'feature_weights': {
            'word_frequency': 0.8, 'sentiment_words': 1.0, 'context': 0.9
        }
    }
}

def get_domain_config(domain: DomainType) -> Dict[str, Any]:
    """Get configuration for a specific domain"""
    return DOMAIN_METADATA.get(domain, {
        'description': f'Generic {domain.value} domain',
        'compatible_tasks': ['classification', 'regression'],
        'output_formats': ['result', 'confidence'],
        'confidence_threshold': 0.6,
        'feature_weights': {}
    })

def get_domain_feature_weight(domain: DomainType, feature_name: str) -> float:
    """Get feature weight for a domain"""
    config = get_domain_config(domain)
    feature_weights = config.get('feature_weights', {})

    # Check for exact match or partial match
    for key, weight in feature_weights.items():
        if key.lower() in feature_name.lower():
            return weight

    return 0.5  # Default weight

def is_task_compatible_with_domain(domain: DomainType, task_type: str) -> bool:
    """Check if a task type is compatible with a domain"""
    config = get_domain_config(domain)
    compatible_tasks = config.get('compatible_tasks', [])
    return task_type in compatible_tasks

def get_all_domains() -> List[DomainType]:
    """Get list of all configured domains"""
    return list(DOMAIN_METADATA.keys())

def get_domain_stats() -> Dict[str, int]:
    """Get statistics about configured domains"""
    return {
        'total_domains': len(DOMAIN_METADATA),
        'domains_with_weights': len([d for d in DOMAIN_METADATA.values() if d.get('feature_weights')]),
        'avg_compatible_tasks': sum(len(d.get('compatible_tasks', [])) for d in DOMAIN_METADATA.values()) // len(DOMAIN_METADATA)
    }
