# config/integrations.py
import yaml
import os

class IntegrationConfig:
    def __init__(self, config_file="config/integrations.yaml"):
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        """Load integration configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def get_prometheus_config(self):
        return self.config.get('prometheus', {
            'url': 'http://localhost:9090',
            'polling_interval': 30,
            'queries': {
                'cpu_high': 'avg(cpu_usage) > 80',
                'memory_high': 'avg(memory_usage) > 85'
            }
        })

    def get_kafka_config(self):
        return self.config.get('kafka', {
            'bootstrap_servers': ['localhost:9092'],
            'topics': ['infrastructure-events', 'cicd-events'],
            'group_id': 'ai-system'
        })

    def get_universal_neural_config(self):
        """Get Universal Neural System specific configuration"""
        return self.config.get('universal_neural', {
            'monitoring': {
                'enabled': True,
                'dashboard_refresh': 30,
                'metrics_retention': '7d'
            },
            'learning': {
                'continuous_enabled': True,
                'save_interval': 100,
                'checkpoint_retention': 10
            },
            'api': {
                'rate_limit': '1000 per minute',
                'authentication': False,
                'cors_enabled': True
            }
        })

    def get_webhooks_config(self):
        """Get webhooks configuration"""
        return self.config.get('webhooks', {
            'enabled': True,
            'authentication_required': False,
            'rate_limit': '100 per minute'
        })
