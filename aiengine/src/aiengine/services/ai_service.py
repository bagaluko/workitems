"""
AI Utility functions for the Universal AI Engine
Renamed to avoid conflicts with existing utils directory
"""

import os
import sys
import time
import json
import logging
import random
import string
from collections import deque
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling special types"""
    try:
        if isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: safe_json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [safe_json_serialize(item) for item in obj]
        else:
            return obj
    except Exception as e:
        logger.warning(f"JSON serialization failed for {type(obj)}: {e}")
        return str(obj)

def validate_input_data(data, max_size_mb=10):
    """Validate input data for safety and size limits"""
    try:
        # Size check
        data_str = str(data)
        size_mb = len(data_str.encode('utf-8')) / (1024 * 1024)

        if size_mb > max_size_mb:
            logger.warning(f"Input data size ({size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)")
            return False, f"Data too large: {size_mb:.2f}MB"

        # Type validation
        allowed_types = (str, int, float, list, dict, tuple, bool, type(None))

        def check_type_recursive(obj, depth=0):
            if depth > 10:  # Prevent infinite recursion
                return False

            if isinstance(obj, allowed_types):
                if isinstance(obj, (list, tuple)):
                    return all(check_type_recursive(item, depth + 1) for item in obj)
                elif isinstance(obj, dict):
                    return all(
                        isinstance(k, (str, int, float)) and check_type_recursive(v, depth + 1)
                        for k, v in obj.items()
                    )
                return True
            return False

        if not check_type_recursive(data):
            return False, "Invalid data types detected"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False, f"Validation error: {e}"

def create_task_id(prefix="task", include_timestamp=True, include_random=True):
    """Create a unique task ID"""
    parts = [prefix]

    if include_timestamp:
        parts.append(str(int(time.time())))

    if include_random:
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        parts.append(random_suffix)

    return "_".join(parts)

def setup_directories():
    """Setup required directories for the system"""
    directories = [
        "logs",
        "models",
        "knowledge_base",
        "exports",
        "backups",
        "cache",
        "temp"
    ]

    created_dirs = []

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
        except Exception as e:
            logger.warning(f"Failed to create directory {directory}: {e}")

    logger.info(f"📁 Directory setup complete: {len(created_dirs)}/{len(directories)} directories ready")
    return created_dirs

def cleanup_temp_files(max_age_hours=24):
    """Clean up temporary files older than specified hours"""
    try:
        temp_dirs = ["temp", "cache"]
        cleaned_files = 0

        cutoff_time = time.time() - (max_age_hours * 3600)

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    filepath = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                            os.remove(filepath)
                            cleaned_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to clean {filepath}: {e}")

        if cleaned_files > 0:
            logger.info(f"🧹 Cleaned up {cleaned_files} temporary files")

        return cleaned_files

    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")
        return 0

def get_system_resources():
    """Get current system resource usage"""
    try:
        import psutil

        resources = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'process_count': len(psutil.pids())
        }

        return resources

    except ImportError:
        logger.warning("psutil not available - using basic resource info")
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'available_memory_gb': 0,
            'process_count': 0
        }
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        return {}

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def health_check():
    """Perform a basic system health check"""
    health_status = {
        'timestamp': time.time(),
        'status': 'healthy',
        'issues': [],
        'warnings': []
    }

    try:
        # Check disk space
        try:
            import psutil
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 90:
                health_status['issues'].append(f"Disk usage critical: {disk_usage}%")
                health_status['status'] = 'critical'
            elif disk_usage > 80:
                health_status['warnings'].append(f"Disk usage high: {disk_usage}%")
        except ImportError:
            pass

        # Check memory
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 95:
                health_status['issues'].append(f"Memory usage critical: {memory_usage}%")
                health_status['status'] = 'critical'
            elif memory_usage > 85:
                health_status['warnings'].append(f"Memory usage high: {memory_usage}%")
        except ImportError:
            pass

        # Check required directories
        required_dirs = ["logs", "models", "knowledge_base"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                health_status['warnings'].append(f"Missing directory: {directory}")

        # Set final status
        if health_status['issues']:
            health_status['status'] = 'critical'
        elif health_status['warnings']:
            health_status['status'] = 'warning'

    except Exception as e:
        health_status['issues'].append(f"Health check failed: {e}")
        health_status['status'] = 'error'

    return health_status

def convert_deque_to_list(obj):
    """Convert deque objects to lists for JSON serialization"""
    if isinstance(obj, deque):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_deque_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_deque_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_deque_to_list(item) for item in obj)
    return obj

def parse_port_config(port_config):
    """Parse port configuration from registry format"""
    if isinstance(port_config, str) and ':' in port_config:
        try:
            host, port = port_config.split(':', 1)
            return int(port)
        except (ValueError, IndexError):
            return 8000  # Default port
    elif isinstance(port_config, (int, str)):
        try:
            return int(port_config)
        except ValueError:
            return 8000  # Default port
    else:
        return 8000  # Default port
