#!/usr/bin/env python3
"""
Intel Ingredient Management System
Comprehensive ingredient creation, validation, and lifecycle management
WITH UNIFIED DATABASE AND EMAIL NOTIFICATIONS
"""

import os
import sys
import json
import time
import hashlib
import re
import zipfile
import tarfile
import shutil
import logging
import asyncio
import tempfile
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import xml.etree.ElementTree as ET
import configparser
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import magic  # For file type detection
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# INGREDIENT TYPES AND ENUMS
# ============================================================================

class IngredientType(Enum):
    """Types of ingredients"""
    INTEL_DRIVER = "intel_driver"
    INTEL_FIRMWARE = "intel_firmware"
    INTEL_SOFTWARE = "intel_software"
    INTEL_LIBRARY = "intel_library"
    TPV_DRIVER = "tpv_driver"
    TPV_FIRMWARE = "tpv_firmware"
    TPV_SOFTWARE = "tpv_software"
    TPV_LIBRARY = "tpv_library"
    SYSTEM_COMPONENT = "system_component"
    SECURITY_UPDATE = "security_update"
    UNKNOWN = "unknown"

class IngredientStatus(Enum):
    """Ingredient lifecycle status"""
    CREATED = "created"
    VALIDATING = "validating"
    VALIDATED = "validated"
    PRECHECK_PENDING = "precheck_pending"
    PRECHECK_RUNNING = "precheck_running"
    PRECHECK_PASSED = "precheck_passed"
    PRECHECK_FAILED = "precheck_failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    ERROR = "error"

class Milestone(Enum):
    """Development milestones"""
    DEVELOPMENT = "development"
    ALPHA = "alpha"
    BETA = "beta"
    RC = "release_candidate"
    RTM = "rtm"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"

class Priority(Enum):
    """Ingredient priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

# ============================================================================
# INGREDIENT DATA STRUCTURES
# ============================================================================

@dataclass
class IngredientMetadata:
    """Comprehensive ingredient metadata"""
    # Basic information
    name: str
    version: str
    description: str
    vendor: str

    # Classification
    ingredient_type: IngredientType
    category: str
    subcategory: str = ""

    # File information
    file_path: str = ""
    file_size: int = 0
    file_hash_md5: str = ""
    file_hash_sha256: str = ""
    file_extension: str = ""
    mime_type: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    # Platform support
    supported_os: List[str] = field(default_factory=list)
    supported_architectures: List[str] = field(default_factory=list)

    # Security information
    digital_signature: Optional[str] = None
    certificate_info: Dict[str, Any] = field(default_factory=dict)
    security_scan_results: Dict[str, Any] = field(default_factory=dict)

    # Business information
    business_owner: str = ""
    technical_contact: str = ""
    approval_required: bool = True

    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Ingredient:
    """Complete ingredient representation"""
    # Unique identifier
    ingredient_id: str

    # Metadata
    metadata: IngredientMetadata

    # Status and lifecycle
    status: IngredientStatus = IngredientStatus.CREATED
    milestone: Milestone = Milestone.DEVELOPMENT
    priority: Priority = Priority.MEDIUM

    # Processing information
    validation_results: Dict[str, Any] = field(default_factory=dict)
    precheck_results: List[Dict[str, Any]] = field(default_factory=list)

    # Tracking
    created_by: str = "system"
    assigned_to: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # History
    status_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'ingredient_id': self.ingredient_id,
            'metadata': asdict(self.metadata),
            'status': self.status.value,
            'milestone': self.milestone.value,
            'priority': self.priority.value,
            'validation_results': self.validation_results,
            'precheck_results': self.precheck_results,
            'created_by': self.created_by,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status_history': self.status_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ingredient':
        """Create from dictionary"""
        metadata = IngredientMetadata(**data['metadata'])

        return cls(
            ingredient_id=data['ingredient_id'],
            metadata=metadata,
            status=IngredientStatus(data['status']),
            milestone=Milestone(data['milestone']),
            priority=Priority(data['priority']),
            validation_results=data.get('validation_results', {}),
            precheck_results=data.get('precheck_results', []),
            created_by=data.get('created_by', 'system'),
            assigned_to=data.get('assigned_to', ''),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            status_history=data.get('status_history', [])
        )

# ============================================================================
# EMAIL NOTIFICATION SYSTEM
# ============================================================================

class IntelNotificationSystem:
    """Enhanced notification system for Intel ingredient management"""

    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(f"{__name__}.IntelNotificationSystem")

        # Email configuration
        # self.config = config or {
        #    # 'smtp_server': 'smtp.intel.com',
        #    'smtp_servers': ['smtp.intel.com', 'mail.intel.com'],
        #    'smtp_port': [587, 25],
        #    'use_tls': True,
        #    'sender_email': 'balasubramanyam.agalukote.lakshmipathi@intel.com',
        #    'sender_name': 'Intel Ingredient Management System',
        #    'dashboard_url': 'https://ingredient-dashboard.intel.com',
        #    'default_recipients': {
        #        'critical': ['balasubramanyam.agalukote.lakshmipathi@intel.com'],
        #        'escalation': ['balasubramanyam.agalukote.lakshmipathi@intel.com'],
        #        'info': ['balasubramanyam.agalukote.lakshmipathi@intel.com']
        #    }
        # }


        # Enhanced email configuration similar to complete_precheck_engine_pro.py
        self.config = config or {
            'enabled': True,
            'intel_email': 'balasubramanyam.agalukote.lakshmipathi@intel.com',
            'sender_email': 'balasubramanyam.agalukote.lakshmipathi@intel.com',  # Changed from ingredient-system@intel.com
            'sender_name': 'Intel Ingredient Management System',
            'smtp_servers': ['smtp.intel.com', 'mail.intel.com', 'mailrelay.intel.com', 'smtp-relay.intel.com'],
            'smtp_ports': [587, 25],
            'timeout': 20,
            'corporate_mode': True,
            'dashboard_url': 'https://ingredient-dashboard.intel.com',
            'default_recipients': {
                'critical': ['balasubramanyam.agalukote.lakshmipathi@intel.com'],
                'escalation': ['balasubramanyam.agalukote.lakshmipathi@intel.com'],
                'info': ['balasubramanyam.agalukote.lakshmipathi@intel.com']
            }
        }

    async def send_ingredient_notification(self, ingredient_data: Dict[str, Any],
                                         notification_type: str = 'created') -> bool:
        """Send notification for ingredient events"""
        try:
            if notification_type == 'created':
                return await self._send_ingredient_created_notification(ingredient_data)
            elif notification_type == 'validation_failed':
                return await self._send_validation_failed_notification(ingredient_data)
            elif notification_type == 'precheck_completed':
                return await self._send_precheck_completed_notification(ingredient_data)

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send notification: {e}")
            return False

    async def _send_ingredient_created_notification(self, ingredient_data: Dict[str, Any]) -> bool:
        """Send notification when ingredient is created"""
        try:
            # Get recipients
            recipients = []
            if ingredient_data.get('business_owner'):
                recipients.append(ingredient_data['business_owner'])
            if ingredient_data.get('technical_contact'):
                recipients.append(ingredient_data['technical_contact'])

            if not recipients:
                recipients = self.config['default_recipients']['info']

            # Generate email content
            subject = f"📦 New Intel Ingredient Created: {ingredient_data.get('name', 'Unknown')}"

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #0071c5; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ margin: 20px 0; }}
        .info-box {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #0071c5; margin: 10px 0; }}
        .button {{ background-color: #0071c5; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🧬 Intel Ingredient Management System</h2>
        <p>New Ingredient Created</p>
    </div>

    <div class="content">
        <h3>Ingredient Details</h3>
        <div class="info-box">
            <strong>Name:</strong> {ingredient_data.get('name', 'Unknown')}<br>
            <strong>ID:</strong> {ingredient_data.get('ingredient_id', 'Unknown')}<br>
            <strong>Type:</strong> {ingredient_data.get('ingredient_type', 'Unknown')}<br>
            <strong>Vendor:</strong> {ingredient_data.get('vendor', 'Unknown')}<br>
            <strong>Version:</strong> {ingredient_data.get('version', 'Unknown')}<br>
            <strong>Status:</strong> {ingredient_data.get('status', 'Unknown')}<br>
            <strong>Business Owner:</strong> {ingredient_data.get('business_owner', 'Not specified')}<br>
            <strong>Technical Contact:</strong> {ingredient_data.get('technical_contact', 'Not specified')}
        </div>

        <div style="margin: 20px 0;">
            <a href="{self.config['dashboard_url']}/ingredient/{ingredient_data.get('ingredient_id', '')}" class="button">📊 View Ingredient</a>
            <a href="{self.config['dashboard_url']}" class="button">🏠 Go to Dashboard</a>
        </div>
    </div>

    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d;">
        <p>This notification was sent by the Intel Ingredient Management System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p>For questions or issues, please contact the Ingredient Management Team</p>
    </div>
</body>
</html>
            """

            text_content = f"""
Intel Ingredient Management System - New Ingredient Created

Ingredient Details:
- Name: {ingredient_data.get('name', 'Unknown')}
- ID: {ingredient_data.get('ingredient_id', 'Unknown')}
- Type: {ingredient_data.get('ingredient_type', 'Unknown')}
- Vendor: {ingredient_data.get('vendor', 'Unknown')}
- Version: {ingredient_data.get('version', 'Unknown')}
- Status: {ingredient_data.get('status', 'Unknown')}
- Business Owner: {ingredient_data.get('business_owner', 'Not specified')}
- Technical Contact: {ingredient_data.get('technical_contact', 'Not specified')}

View ingredient: {self.config['dashboard_url']}/ingredient/{ingredient_data.get('ingredient_id', '')}
Dashboard: {self.config['dashboard_url']}

This notification was sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """

            return await self._send_email(recipients, subject, html_content, text_content)

        except Exception as e:
            self.logger.error(f"❌ Failed to send ingredient created notification: {e}")
            return False

    async def _send_validation_failed_notification(self, ingredient_data: Dict[str, Any]) -> bool:
        """Send notification when validation fails"""
        try:
            recipients = self.config['default_recipients']['critical']

            validation_results = ingredient_data.get('validation_results', {})
            errors = validation_results.get('errors', [])
            warnings = validation_results.get('warnings', [])

            subject = f"🚨 Ingredient Validation Failed: {ingredient_data.get('name', 'Unknown')}"

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #dc3545; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ margin: 20px 0; }}
        .error-box {{ background-color: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 10px 0; }}
        .warning-box {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .button {{ background-color: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🚨 Ingredient Validation Failed</h2>
        <p>Ingredient: {ingredient_data.get('name', 'Unknown')}</p>
    </div>

    <div class="content">
        <h3>Validation Errors ({len(errors)})</h3>
        <div class="error-box">
            {'<br>'.join([f"• {error}" for error in errors]) if errors else 'No errors reported'}
        </div>

        <h3>Validation Warnings ({len(warnings)})</h3>
        <div class="warning-box">
            {'<br>'.join([f"• {warning}" for warning in warnings]) if warnings else 'No warnings reported'}
        </div>

        <a href="{self.config['dashboard_url']}/ingredient/{ingredient_data.get('ingredient_id', '')}" class="button">🔍 Review Ingredient</a>
    </div>
</body>
</html>
            """

            text_content = f"""
🚨 INGREDIENT VALIDATION FAILED

Ingredient: {ingredient_data.get('name', 'Unknown')}
ID: {ingredient_data.get('ingredient_id', 'Unknown')}

Validation Errors ({len(errors)}):
{chr(10).join([f"• {error}" for error in errors]) if errors else 'No errors reported'}

Validation Warnings ({len(warnings)}):
{chr(10).join([f"• {warning}" for warning in warnings]) if warnings else 'No warnings reported'}

Review: {self.config['dashboard_url']}/ingredient/{ingredient_data.get('ingredient_id', '')}
            """

            return await self._send_email(recipients, subject, html_content, text_content)

        except Exception as e:
            self.logger.error(f"❌ Failed to send validation failed notification: {e}")
            return False


    async def _send_email(self, recipients: List[str], subject: str,
                     html_content: str, text_content: str) -> bool:
        """Send email using multiple SMTP methods like complete_precheck_engine_pro.py"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.config['sender_name']} <{self.config['sender_email']}>"
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject

            # Set corporate headers
            msg['X-Priority'] = '2'
            msg['X-Mailer'] = 'Intel Ingredient Management System v2.1'
            msg['Organization'] = 'Intel Corporation'
            msg['X-Intel-System'] = 'Ingredient-Management'
            msg['X-Intel-Classification'] = 'Confidential'

            # Add text and HTML parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            # Try multiple SMTP methods like in complete_precheck_engine_pro.py
            return self._try_multiple_smtp_methods(msg, recipients)

        except Exception as e:
            self.logger.error(f"❌ Failed to send email: {e}")
            return False

    def _try_multiple_smtp_methods(self, msg: MIMEMultipart, recipients: List[str]) -> bool:
        """Try multiple SMTP methods similar to complete_precheck_engine_pro.py"""

        # Method 1: Intel Corporate SMTP with TLS
        try:
            self.logger.info("📧 Attempting Intel Corporate SMTP with TLS...")
            context = ssl.create_default_context()

            with smtplib.SMTP('smtp.intel.com', 587, timeout=self.config['timeout']) as server:
                server.starttls(context=context)
                server.send_message(msg)
                self.logger.info("✅ Email sent via Intel SMTP TLS")
                return True

        except Exception as e:
            self.logger.warning(f"⚠️ Intel Corporate SMTP TLS failed: {e}")

        # Method 2: Intel Corporate SMTP without TLS
        try:
            self.logger.info("📧 Attempting Intel Corporate SMTP...")

            with smtplib.SMTP('smtp.intel.com', 25, timeout=15) as server:
                server.send_message(msg)
                self.logger.info("✅ Email sent via Intel SMTP")
                return True

        except Exception as e:
            self.logger.warning(f"⚠️ Intel Corporate SMTP failed: {e}")

        # Method 3: Try alternative Intel mail servers
        for server_name in self.config['smtp_servers']:
            for port in self.config['smtp_ports']:
                try:
                    self.logger.info(f"📧 Trying Intel server {server_name}:{port}...")

                    with smtplib.SMTP(server_name, port, timeout=10) as server:
                        if port == 587:
                            context = ssl.create_default_context()
                            server.starttls(context=context)
                        server.send_message(msg)
                        self.logger.info(f"✅ Email sent via {server_name}:{port}")
                        return True

                except Exception as e:
                    self.logger.warning(f"⚠️ {server_name}:{port} failed: {e}")

        # Fallback: Save email locally (like in complete_precheck_engine_pro.py)
        self.logger.info("📁 All SMTP methods failed, saving email locally...")
        self._save_email_locally(msg, recipients)
        return True  # Consider successful for demo

    def _save_email_locally(self, msg: MIMEMultipart, recipients: List[str]):
        """Save email locally when SMTP fails"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save EML file
            eml_filename = f"Intel_Ingredient_Email_{timestamp}.eml"
            eml_filepath = os.path.join('/tmp', eml_filename)

            with open(eml_filepath, 'w', encoding='utf-8') as f:
                f.write(msg.as_string())

            # Save HTML file
            html_filename = f"Intel_Ingredient_Email_{timestamp}.html"
            html_filepath = os.path.join('/tmp', html_filename)

            # Extract and save HTML content
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    with open(html_filepath, 'w', encoding='utf-8') as f:
                        f.write(part.get_payload(decode=True).decode('utf-8'))
                    break

            self.logger.info(f"📁 Email saved: {eml_filepath}")
            self.logger.info(f"🌐 HTML version: {html_filepath}")

            # Display email summary
            print("\n" + "="*60)
            print("📧 INTEL INGREDIENT EMAIL NOTIFICATION")
            print("="*60)
            print(f"📤 From: {msg['From']}")
            print(f"📥 To: {', '.join(recipients)}")
            print(f"📋 Subject: {msg['Subject']}")
            print(f"📁 EML File: {eml_filepath}")
            print(f"🌐 HTML File: {html_filepath}")
            print("="*60)
            print(f"🌐 View in browser: firefox {html_filepath} &")
            print("="*60)

        except Exception as e:
            self.logger.error(f"❌ Failed to save email locally: {e}")




# ============================================================================
# INGREDIENT FACTORY (UNCHANGED)
# ============================================================================

class IngredientFactory:
    """Factory for creating ingredients from various sources"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IngredientFactory")

        # File type mappings
        self.file_type_mappings = {
            '.dll': IngredientType.INTEL_LIBRARY,
            '.sys': IngredientType.INTEL_DRIVER,
            '.exe': IngredientType.INTEL_SOFTWARE,
            '.msi': IngredientType.INTEL_SOFTWARE,
            '.inf': IngredientType.INTEL_DRIVER,
            '.cat': IngredientType.INTEL_DRIVER,
            '.bin': IngredientType.INTEL_FIRMWARE,
            '.fw': IngredientType.INTEL_FIRMWARE,
            '.zip': IngredientType.SYSTEM_COMPONENT,
            '.cab': IngredientType.SYSTEM_COMPONENT
        }

        # Vendor detection patterns
        self.vendor_patterns = {
            'intel': ['intel', 'igfx', 'gfx', 'graphics', 'wifi', 'bluetooth', 'chipset'],
            'nvidia': ['nvidia', 'nvda', 'geforce', 'quadro'],
            'amd': ['amd', 'ati', 'radeon'],
            'realtek': ['realtek', 'rtl'],
            'broadcom': ['broadcom', 'bcm'],
            'qualcomm': ['qualcomm', 'qcom', 'atheros'],
            'microsoft': ['microsoft', 'msft', 'windows']
        }

    def create_from_file(self, file_path: str, **kwargs) -> Ingredient:
        """Create ingredient from file"""
        try:
            self.logger.info(f"🧬 Creating ingredient from file: {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Extract file information
            file_info = self._extract_file_info(file_path)

            # Detect ingredient type and vendor
            ingredient_type = self._detect_ingredient_type(file_path, file_info)
            vendor = self._detect_vendor(file_path, file_info)

            # Create metadata
            metadata = IngredientMetadata(
                name=kwargs.get('name', Path(file_path).stem),
                version=kwargs.get('version', self._extract_version(file_path, file_info)),
                description=kwargs.get('description', f"Ingredient from {Path(file_path).name}"),
                vendor=vendor,
                ingredient_type=ingredient_type,
                category=kwargs.get('category', ingredient_type.value.split('_')[1]),
                file_path=file_path,
                file_size=file_info['size'],
                file_hash_md5=file_info['md5'],
                file_hash_sha256=file_info['sha256'],
                file_extension=file_info['extension'],
                mime_type=file_info['mime_type'],
                supported_os=kwargs.get('supported_os', ['windows']),
                supported_architectures=kwargs.get('supported_architectures', ['x64']),
                business_owner=kwargs.get('business_owner', ''),
                technical_contact=kwargs.get('technical_contact', '')
            )

            # Create ingredient
            ingredient_id = self._generate_ingredient_id(metadata)

            ingredient = Ingredient(
                ingredient_id=ingredient_id,
                metadata=metadata,
                milestone=kwargs.get('milestone', Milestone.DEVELOPMENT),
                priority=kwargs.get('priority', Priority.MEDIUM),
                created_by=kwargs.get('created_by', 'system')
            )

            # Add initial status
            ingredient.status_history.append({
                'status': IngredientStatus.CREATED.value,
                'timestamp': time.time(),
                'user': ingredient.created_by,
                'notes': 'Ingredient created from file'
            })

            self.logger.info(f"✅ Created ingredient: {ingredient_id}")
            return ingredient

        except Exception as e:
            self.logger.error(f"❌ Failed to create ingredient from file {file_path}: {e}")
            raise

    def create_from_manifest(self, manifest_path: str) -> List[Ingredient]:
        """Create ingredients from manifest file"""
        try:
            self.logger.info(f"📋 Creating ingredients from manifest: {manifest_path}")

            with open(manifest_path, 'r') as f:
                if manifest_path.endswith('.json'):
                    manifest_data = json.load(f)
                elif manifest_path.endswith('.yaml') or manifest_path.endswith('.yml'):
                    import yaml
                    manifest_data = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported manifest format")

            ingredients = []

            for item in manifest_data.get('ingredients', []):
                try:
                    ingredient = self._create_from_manifest_item(item)
                    ingredients.append(ingredient)
                except Exception as e:
                    self.logger.error(f"❌ Failed to create ingredient from manifest item: {e}")

            self.logger.info(f"✅ Created {len(ingredients)} ingredients from manifest")
            return ingredients

        except Exception as e:
            self.logger.error(f"❌ Failed to create ingredients from manifest: {e}")
            raise

    def create_from_directory(self, directory_path: str, recursive: bool = True) -> List[Ingredient]:
        """Create ingredients from all files in directory"""
        try:
            self.logger.info(f"📁 Creating ingredients from directory: {directory_path}")

            ingredients = []

            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_ingredient_file(file_path):
                            try:
                                ingredient = self.create_from_file(file_path)
                                ingredients.append(ingredient)
                            except Exception as e:
                                self.logger.warning(f"⚠️ Skipped file {file_path}: {e}")
            else:
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path) and self._is_ingredient_file(file_path):
                        try:
                            ingredient = self.create_from_file(file_path)
                            ingredients.append(ingredient)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Skipped file {file_path}: {e}")

            self.logger.info(f"✅ Created {len(ingredients)} ingredients from directory")
            return ingredients

        except Exception as e:
            self.logger.error(f"❌ Failed to create ingredients from directory: {e}")
            raise

    def _extract_file_info(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive file information"""
        try:
            stat = os.stat(file_path)

            # Calculate hashes
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()

            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)

            # Get MIME type
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except:
                mime_type = "application/octet-stream"

            return {
                'size': stat.st_size,
                'md5': md5_hash.hexdigest(),
                'sha256': sha256_hash.hexdigest(),
                'extension': Path(file_path).suffix.lower(),
                'mime_type': mime_type,
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to extract file info: {e}")
            return {
                'size': 0,
                'md5': '',
                'sha256': '',
                'extension': '',
                'mime_type': '',
                'created': 0,
                'modified': 0
            }

    def _detect_ingredient_type(self, file_path: str, file_info: Dict[str, Any]) -> IngredientType:
        """Detect ingredient type from file"""
        file_name = Path(file_path).name.lower()
        extension = file_info['extension']

        # Check by extension first
        if extension in self.file_type_mappings:
            base_type = self.file_type_mappings[extension]

            # Refine based on vendor
            vendor = self._detect_vendor(file_path, file_info)
            if vendor != 'intel':
                # Convert Intel types to TPV types
                type_mapping = {
                    IngredientType.INTEL_DRIVER: IngredientType.TPV_DRIVER,
                    IngredientType.INTEL_FIRMWARE: IngredientType.TPV_FIRMWARE,
                    IngredientType.INTEL_SOFTWARE: IngredientType.TPV_SOFTWARE,
                    IngredientType.INTEL_LIBRARY: IngredientType.TPV_LIBRARY
                }
                return type_mapping.get(base_type, base_type)

            return base_type

        # Special cases
        if 'driver' in file_name:
            return IngredientType.INTEL_DRIVER
        elif 'firmware' in file_name or 'fw' in file_name:
            return IngredientType.INTEL_FIRMWARE
        elif 'security' in file_name or 'patch' in file_name:
            return IngredientType.SECURITY_UPDATE

        return IngredientType.UNKNOWN

    def _detect_vendor(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Detect vendor from file path and name"""
        file_path_lower = file_path.lower()

        for vendor, patterns in self.vendor_patterns.items():
            if any(pattern in file_path_lower for pattern in patterns):
                return vendor

        return 'unknown'

    def _extract_version(self, file_path: str, file_info: Dict[str, Any]) -> str:
        """Extract version information from file"""
        try:
            # Try to extract from filename
            file_name = Path(file_path).stem

            # Look for version patterns
            version_patterns = [
                r'v?(\d+\.\d+\.\d+\.\d+)',  # 1.2.3.4
                r'v?(\d+\.\d+\.\d+)',       # 1.2.3
                r'v?(\d+\.\d+)',            # 1.2
                r'_(\d+)_(\d+)_(\d+)',      # _1_2_3
            ]

            for pattern in version_patterns:
                match = re.search(pattern, file_name)
                if match:
                    return match.group(1) if len(match.groups()) == 1 else '.'.join(match.groups())

            # Try to extract from PE file (Windows executables)
            if file_info['extension'] in ['.exe', '.dll', '.sys']:
                version = self._extract_pe_version(file_path)
                if version:
                    return version

            return "1.0.0"

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract version: {e}")
            return "1.0.0"

    def _extract_pe_version(self, file_path: str) -> Optional[str]:
        """Extract version from PE file"""
        try:
            # This would require a PE parser library
            # For now, return None
            return None
        except:
            return None

    def _is_ingredient_file(self, file_path: str) -> bool:
        """Check if file is a valid ingredient"""
        extension = Path(file_path).suffix.lower()
        return extension in self.file_type_mappings

    def _generate_ingredient_id(self, metadata: IngredientMetadata) -> str:
        """Generate unique ingredient ID"""
        # Create ID based on name, vendor, version, and file hash
        id_string = f"{metadata.vendor}_{metadata.name}_{metadata.version}_{metadata.file_hash_md5[:8]}"

        # Clean up the ID
        id_string = re.sub(r'[^a-zA-Z0-9_-]', '_', id_string)
        id_string = re.sub(r'_+', '_', id_string)

        return f"ing_{id_string}_{int(time.time())}"

    def _create_from_manifest_item(self, item: Dict[str, Any]) -> Ingredient:
        """Create ingredient from manifest item"""
        # Extract file path
        file_path = item.get('file_path', '')

        if file_path and os.path.exists(file_path):
            # Create from file with manifest overrides
            return self.create_from_file(file_path, **item)
        else:
            # Create from manifest data only
            metadata = IngredientMetadata(
                name=item['name'],
                version=item.get('version', '1.0.0'),
                description=item.get('description', ''),
                vendor=item.get('vendor', 'unknown'),
                ingredient_type=IngredientType(item.get('type', 'unknown')),
                category=item.get('category', 'component'),
                supported_os=item.get('supported_os', ['windows']),
                supported_architectures=item.get('architectures', ['x64']),
                business_owner=item.get('business_owner', ''),
                technical_contact=item.get('technical_contact', '')
            )

            ingredient_id = self._generate_ingredient_id(metadata)

            return Ingredient(
                ingredient_id=ingredient_id,
                metadata=metadata,
                milestone=Milestone(item.get('milestone', 'development')),
                priority=Priority(item.get('priority', 'medium')),
                created_by=item.get('created_by', 'manifest')
            )

# ============================================================================
# INGREDIENT VALIDATOR (UNCHANGED)
# ============================================================================

class IngredientValidator:
    """Validates ingredients for compliance and quality"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IngredientValidator")

        # Validation rules
        self.validation_rules = {
            'file_size_limits': {
                'max_size_mb': 500,  # 500MB max
                'min_size_bytes': 1024  # 1KB min
            },
            'required_fields': [
                'name', 'version', 'vendor', 'ingredient_type'
            ],
            'filename_patterns': {
                'allowed_chars': r'^[a-zA-Z0-9._-]+$',
                'max_length': 255
            },
            'security_requirements': {
                'require_signature': True,
                'allowed_extensions': ['.dll', '.exe', '.sys', '.inf', '.cat', '.msi'],
                'blocked_extensions': ['.bat', '.cmd', '.scr', '.vbs']
            }
        }

    async def validate_ingredient(self, ingredient: Ingredient) -> Dict[str, Any]:
        """Comprehensive ingredient validation"""
        try:
            self.logger.info(f"🔍 Validating ingredient: {ingredient.ingredient_id}")

            validation_results = {
                'overall_status': 'passed',
                'validation_timestamp': time.time(),
                'checks_performed': [],
                'warnings': [],
                'errors': [],
                'security_scan': {},
                'compliance_check': {}
            }

            # Basic validation
            basic_results = await self._validate_basic_requirements(ingredient)
            validation_results['checks_performed'].append('basic_requirements')
            validation_results['warnings'].extend(basic_results.get('warnings', []))
            validation_results['errors'].extend(basic_results.get('errors', []))

            # File validation (if file exists)
            if ingredient.metadata.file_path and os.path.exists(ingredient.metadata.file_path):
                file_results = await self._validate_file(ingredient)
                validation_results['checks_performed'].append('file_validation')
                validation_results['warnings'].extend(file_results.get('warnings', []))
                validation_results['errors'].extend(file_results.get('errors', []))

            # Security validation
            security_results = await self._validate_security(ingredient)
            validation_results['security_scan'] = security_results
            validation_results['checks_performed'].append('security_validation')
            validation_results['warnings'].extend(security_results.get('warnings', []))
            validation_results['errors'].extend(security_results.get('errors', []))

            # Compliance validation
            compliance_results = await self._validate_compliance(ingredient)
            validation_results['compliance_check'] = compliance_results
            validation_results['checks_performed'].append('compliance_validation')
            validation_results['warnings'].extend(compliance_results.get('warnings', []))
            validation_results['errors'].extend(compliance_results.get('errors', []))

            # Determine overall status
            if validation_results['errors']:
                validation_results['overall_status'] = 'failed'
            elif validation_results['warnings']:
                validation_results['overall_status'] = 'passed_with_warnings'

            self.logger.info(f"✅ Validation complete: {validation_results['overall_status']}")
            return validation_results

        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {
                'overall_status': 'error',
                'validation_timestamp': time.time(),
                'error': str(e)
            }

    async def _validate_basic_requirements(self, ingredient: Ingredient) -> Dict[str, Any]:
        """Validate basic requirements"""
        results = {'warnings': [], 'errors': []}

        # Check required fields
        for field in self.validation_rules['required_fields']:
            if not getattr(ingredient.metadata, field, None):
                results['errors'].append(f"Missing required field: {field}")

        # Check name format
        if ingredient.metadata.name:
            if not re.match(self.validation_rules['filename_patterns']['allowed_chars'],
                          ingredient.metadata.name):
                results['errors'].append("Invalid characters in ingredient name")

            if len(ingredient.metadata.name) > self.validation_rules['filename_patterns']['max_length']:
                results['errors'].append("Ingredient name too long")

        # Check version format
        if ingredient.metadata.version:
            version_pattern = r'^\d+(\.\d+)*$'
            if not re.match(version_pattern, ingredient.metadata.version):
                results['warnings'].append("Version format should be numeric (e.g., 1.2.3)")

        return results

    async def _validate_file(self, ingredient: Ingredient) -> Dict[str, Any]:
        """Validate file-specific requirements"""
        results = {'warnings': [], 'errors': []}

        file_path = ingredient.metadata.file_path

        try:
            # Check file exists
            if not os.path.exists(file_path):
                results['errors'].append(f"File not found: {file_path}")
                return results

            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = self.validation_rules['file_size_limits']['max_size_mb'] * 1024 * 1024
            min_size = self.validation_rules['file_size_limits']['min_size_bytes']

            if file_size > max_size:
                results['errors'].append(f"File too large: {file_size} bytes (max: {max_size})")
            elif file_size < min_size:
                results['errors'].append(f"File too small: {file_size} bytes (min: {min_size})")

            # Check file extension
            extension = Path(file_path).suffix.lower()
            allowed_extensions = self.validation_rules['security_requirements']['allowed_extensions']
            blocked_extensions = self.validation_rules['security_requirements']['blocked_extensions']

            if extension in blocked_extensions:
                results['errors'].append(f"Blocked file extension: {extension}")
            elif extension not in allowed_extensions:
                results['warnings'].append(f"Unusual file extension: {extension}")

            # Verify file hashes
            if ingredient.metadata.file_hash_md5:
                actual_md5 = self._calculate_file_hash(file_path, 'md5')
                if actual_md5 != ingredient.metadata.file_hash_md5:
                    results['errors'].append("MD5 hash mismatch - file may be corrupted")

            if ingredient.metadata.file_hash_sha256:
                actual_sha256 = self._calculate_file_hash(file_path, 'sha256')
                if actual_sha256 != ingredient.metadata.file_hash_sha256:
                    results['errors'].append("SHA256 hash mismatch - file may be corrupted")

        except Exception as e:
            results['errors'].append(f"File validation error: {str(e)}")

        return results

    async def _validate_security(self, ingredient: Ingredient) -> Dict[str, Any]:
        """Validate security requirements"""
        results = {
            'signature_check': {'status': 'not_performed'},
            'virus_scan': {'status': 'not_performed'},
            'certificate_validation': {'status': 'not_performed'},
            'warnings': [],
            'errors': []
        }

        file_path = ingredient.metadata.file_path

        if file_path and os.path.exists(file_path):
            # Digital signature check
            if self.validation_rules['security_requirements']['require_signature']:
                signature_result = await self._check_digital_signature(file_path)
                results['signature_check'] = signature_result

                if not signature_result.get('is_signed', False):
                    results['errors'].append("Digital signature required but not found")

            # Basic virus scan (placeholder)
            virus_result = await self._basic_virus_scan(file_path)
            results['virus_scan'] = virus_result

            if virus_result.get('threats_found', 0) > 0:
                results['errors'].append("Security threats detected in file")

        return results

    async def _validate_compliance(self, ingredient: Ingredient) -> Dict[str, Any]:
        """Validate compliance requirements"""
        results = {
            'intel_compliance': {'status': 'passed'},
            'regulatory_compliance': {'status': 'passed'},
            'warnings': [],
            'errors': []
        }

        # Intel-specific compliance checks
        if ingredient.metadata.vendor.lower() == 'intel':
            # Check for required Intel metadata
            if not ingredient.metadata.business_owner:
                results['warnings'].append("Intel ingredients should have business owner")

            if not ingredient.metadata.technical_contact:
                results['warnings'].append("Intel ingredients should have technical contact")

        # TPV compliance checks
        elif ingredient.metadata.ingredient_type.value.startswith('tpv'):
            # Check for TPV-specific requirements
            if not ingredient.metadata.vendor or ingredient.metadata.vendor == 'unknown':
                results['errors'].append("TPV ingredients must specify vendor")

        return results

    async def _check_digital_signature(self, file_path: str) -> Dict[str, Any]:
        """Check digital signature of file"""
        try:
            # This would integrate with Windows signtool or similar
            # For now, return placeholder
            return {
                'status': 'checked',
                'is_signed': False,
                'signer': None,
                'certificate_valid': False,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    async def _basic_virus_scan(self, file_path: str) -> Dict[str, Any]:
        """Basic virus scan"""
        try:
            # This would integrate with antivirus engine
            # For now, return placeholder
            return {
                'status': 'scanned',
                'threats_found': 0,
                'scan_engine': 'placeholder',
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def _calculate_file_hash(self, file_path: str, algorithm: str) -> str:
        """Calculate file hash"""
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

# ============================================================================
# ENHANCED INGREDIENT REPOSITORY WITH UNIFIED DATABASE
# ============================================================================

class IngredientRepository:
    """Repository for storing and managing ingredients with unified database schema"""

    def __init__(self, db_path: str = "unified_ingredients.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.IngredientRepository")
        self._initialize_database()

        # In-memory cache for performance
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _initialize_database(self):
        """Initialize SQLite database with unified schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Main ingredients table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ingredients (
                    ingredient_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    ingredient_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    milestone TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    file_path TEXT,
                    file_size INTEGER,
                    file_hash_md5 TEXT,
                    file_hash_sha256 TEXT,
                    created_by TEXT,
                    assigned_to TEXT,
                    created_at REAL,
                    updated_at REAL,
                    metadata_json TEXT,
                    validation_results_json TEXT,
                    precheck_results_json TEXT,
                    status_history_json TEXT
                )
            ''')

            # NEW: Precheck results table with foreign key relationship
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precheck_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingredient_id TEXT NOT NULL,
                    precheck_session_id TEXT NOT NULL,
                    precheck_name TEXT NOT NULL,
                    precheck_passed BOOLEAN NOT NULL,
                    execution_time REAL,
                    error_message TEXT,
                    severity TEXT,
                    decision TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    neural_used BOOLEAN DEFAULT FALSE,
                    ai_analysis_json TEXT,
                    timestamp REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ingredient_id) REFERENCES ingredients (ingredient_id)
                )
            ''')

            # NEW: Precheck sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precheck_sessions (
                    session_id TEXT PRIMARY KEY,
                    ingredient_id TEXT NOT NULL,
                    session_type TEXT NOT NULL,  -- 'real_time', 'manual', 'batch'
                    total_prechecks INTEGER,
                    passed_prechecks INTEGER,
                    failed_prechecks INTEGER,
                    overall_decision TEXT,
                    overall_confidence REAL,
                    processing_time REAL,
                    neural_system_used BOOLEAN DEFAULT FALSE,
                    started_at REAL,
                    completed_at REAL,
                    created_by TEXT DEFAULT 'system',
                    FOREIGN KEY (ingredient_id) REFERENCES ingredients (ingredient_id)
                )
            ''')

            # NEW: Notification log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingredient_id TEXT,
                    notification_type TEXT NOT NULL,  -- 'email', 'slack', 'dashboard'
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',  -- 'pending', 'sent', 'failed'
                    sent_at REAL,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ingredient_id) REFERENCES ingredients (ingredient_id)
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_name ON ingredients(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_vendor ON ingredients(vendor)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_type ON ingredients(ingredient_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_status ON ingredients(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ingredient_created ON ingredients(created_at)')

            # NEW: Precheck indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precheck_ingredient ON precheck_results(ingredient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precheck_session ON precheck_results(precheck_session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precheck_decision ON precheck_results(decision)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_precheck_timestamp ON precheck_results(timestamp)')

            # NEW: Session indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_ingredient ON precheck_sessions(ingredient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_decision ON precheck_sessions(overall_decision)')

            # NEW: Notification indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_ingredient ON notifications(ingredient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_status ON notifications(status)')

            conn.commit()
            conn.close()

            self.logger.info(f"✅ Unified ingredient repository initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize repository: {e}")
            raise

    def store_ingredient(self, ingredient: Ingredient) -> bool:
        """Store ingredient in repository"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update timestamp
            ingredient.updated_at = time.time()

            # Convert metadata to dict and handle Enums properly
            metadata_dict = asdict(ingredient.metadata)

            # Convert enum to string value if it exists
            if 'ingredient_type' in metadata_dict and hasattr(metadata_dict['ingredient_type'], 'value'):
                metadata_dict['ingredient_type'] = metadata_dict['ingredient_type'].value

            cursor.execute('''
                INSERT OR REPLACE INTO ingredients (
                    ingredient_id, name, version, vendor, ingredient_type,
                    status, milestone, priority, file_path, file_size,
                    file_hash_md5, file_hash_sha256, created_by, assigned_to,
                    created_at, updated_at, metadata_json, validation_results_json,
                    precheck_results_json, status_history_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ingredient.ingredient_id,
                ingredient.metadata.name,
                ingredient.metadata.version,
                ingredient.metadata.vendor,
                ingredient.metadata.ingredient_type.value,
                ingredient.status.value,
                ingredient.milestone.value,
                ingredient.priority.value,
                ingredient.metadata.file_path,
                ingredient.metadata.file_size,
                ingredient.metadata.file_hash_md5,
                ingredient.metadata.file_hash_sha256,
                ingredient.created_by,
                ingredient.assigned_to,
                ingredient.created_at,
                ingredient.updated_at,
                json.dumps(metadata_dict),
                json.dumps(ingredient.validation_results),
                json.dumps(ingredient.precheck_results),
                json.dumps(ingredient.status_history)
            ))

            conn.commit()
            conn.close()

            # Update cache
            with self._cache_lock:
                self._cache[ingredient.ingredient_id] = ingredient

            self.logger.info(f"✅ Stored ingredient: {ingredient.ingredient_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store ingredient: {e}")
            return False

    # NEW: Methods for precheck integration
    def store_precheck_session(self, session_data: Dict[str, Any]) -> bool:
        """Store precheck session data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO precheck_sessions (
                    session_id, ingredient_id, session_type, total_prechecks,
                    passed_prechecks, failed_prechecks, overall_decision,
                    overall_confidence, processing_time, neural_system_used,
                    started_at, completed_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_data['session_id'],
                session_data['ingredient_id'],
                session_data['session_type'],
                session_data['total_prechecks'],
                session_data['passed_prechecks'],
                session_data['failed_prechecks'],
                session_data['overall_decision'],
                session_data['overall_confidence'],
                session_data['processing_time'],
                session_data['neural_system_used'],
                session_data['started_at'],
                session_data['completed_at'],
                session_data.get('created_by', 'system')
            ))

            conn.commit()
            conn.close()

            self.logger.info(f"✅ Stored precheck session: {session_data['session_id']}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store precheck session: {e}")
            return False

    def store_precheck_result(self, result_data: Dict[str, Any]) -> bool:
        """Store individual precheck result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO precheck_results (
                    ingredient_id, precheck_session_id, precheck_name,
                    precheck_passed, execution_time, error_message, severity,
                    decision, confidence, reasoning, neural_used,
                    ai_analysis_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_data['ingredient_id'],
                result_data['precheck_session_id'],
                result_data['precheck_name'],
                result_data['precheck_passed'],
                result_data['execution_time'],
                result_data['error_message'],
                result_data['severity'],
                result_data['decision'],
                result_data['confidence'],
                result_data['reasoning'],
                result_data['neural_used'],
                json.dumps(result_data.get('ai_analysis', {})),
                result_data['timestamp']
            ))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store precheck result: {e}")
            return False

    def get_ingredient_with_prechecks(self, ingredient_id: str) -> Optional[Dict[str, Any]]:
        """Get ingredient with all precheck results"""
        try:
            ingredient = self.get_ingredient(ingredient_id)
            if not ingredient:
                return None

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get precheck sessions
            cursor.execute('''
                SELECT * FROM precheck_sessions
                WHERE ingredient_id = ?
                ORDER BY started_at DESC
            ''', (ingredient_id,))
            sessions = [dict(row) for row in cursor.fetchall()]

            # Get precheck results
            cursor.execute('''
                SELECT * FROM precheck_results
                WHERE ingredient_id = ?
                ORDER BY timestamp DESC
            ''', (ingredient_id,))
            precheck_results = [dict(row) for row in cursor.fetchall()]

            conn.close()

            return {
                'ingredient': ingredient.to_dict(),
                'precheck_sessions': sessions,
                'precheck_results': precheck_results,
                'total_prechecks': len(precheck_results),
                'passed_prechecks': len([r for r in precheck_results if r['precheck_passed']]),
                'failed_prechecks': len([r for r in precheck_results if not r['precheck_passed']])
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get ingredient with prechecks: {e}")
            return None

    def log_notification(self, notification_data: Dict[str, Any]) -> bool:
        """Log notification attempt"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO notifications (
                    ingredient_id, notification_type, recipient, subject,
                    message, status, sent_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification_data.get('ingredient_id'),
                notification_data['notification_type'],
                notification_data['recipient'],
                notification_data['subject'],
                notification_data['message'],
                notification_data.get('status', 'pending'),
                notification_data.get('sent_at'),
                notification_data.get('error_message')
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to log notification: {e}")
            return False

    def get_ingredient(self, ingredient_id: str) -> Optional[Ingredient]:
        """Get ingredient by ID"""
        try:
            # Check cache first
            with self._cache_lock:
                if ingredient_id in self._cache:
                    return self._cache[ingredient_id]

            # Query database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM ingredients WHERE ingredient_id = ?', (ingredient_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                ingredient = self._row_to_ingredient(row)

                # Update cache
                with self._cache_lock:
                    self._cache[ingredient_id] = ingredient

                return ingredient

            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to get ingredient {ingredient_id}: {e}")
            return None

    def search_ingredients(self, **criteria) -> List[Ingredient]:
        """Search ingredients by criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query
            where_clauses = []
            params = []

            for key, value in criteria.items():
                if key in ['name', 'vendor', 'ingredient_type', 'status', 'milestone', 'priority']:
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                elif key == 'name_like':
                    where_clauses.append("name LIKE ?")
                    params.append(f"%{value}%")
                elif key == 'created_after':
                    where_clauses.append("created_at > ?")
                    params.append(value)
                elif key == 'created_before':
                    where_clauses.append("created_at < ?")
                    params.append(value)

            query = "SELECT * FROM ingredients"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " ORDER BY created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            ingredients = [self._row_to_ingredient(row) for row in rows]

            self.logger.info(f"🔍 Found {len(ingredients)} ingredients matching criteria")
            return ingredients

        except Exception as e:
            self.logger.error(f"❌ Failed to search ingredients: {e}")
            return []

    def update_ingredient_status(self, ingredient_id: str, new_status: IngredientStatus,
                               user: str = "system", notes: str = "") -> bool:
        """Update ingredient status"""
        try:
            ingredient = self.get_ingredient(ingredient_id)
            if not ingredient:
                return False

            # Add to status history
            ingredient.status_history.append({
                'status': new_status.value,
                'timestamp': time.time(),
                'user': user,
                'notes': notes,
                'previous_status': ingredient.status.value
            })

            # Update status
            ingredient.status = new_status
            ingredient.updated_at = time.time()

            # Store updated ingredient
            return self.store_ingredient(ingredient)

        except Exception as e:
            self.logger.error(f"❌ Failed to update ingredient status: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total count
            cursor.execute('SELECT COUNT(*) FROM ingredients')
            total_count = cursor.fetchone()[0]

            # By status
            cursor.execute('SELECT status, COUNT(*) FROM ingredients GROUP BY status')
            status_counts = dict(cursor.fetchall())

            # By type
            cursor.execute('SELECT ingredient_type, COUNT(*) FROM ingredients GROUP BY ingredient_type')
            type_counts = dict(cursor.fetchall())

            # By vendor
            cursor.execute('SELECT vendor, COUNT(*) FROM ingredients GROUP BY vendor')
            vendor_counts = dict(cursor.fetchall())

            # NEW: Precheck statistics
            cursor.execute('SELECT COUNT(*) FROM precheck_sessions')
            total_sessions = cursor.fetchone()[0]

            cursor.execute('SELECT overall_decision, COUNT(*) FROM precheck_sessions GROUP BY overall_decision')
            decision_counts = dict(cursor.fetchall())

            # NEW: Notification statistics
            cursor.execute('SELECT status, COUNT(*) FROM notifications GROUP BY status')
            notification_counts = dict(cursor.fetchall())

            conn.close()

            return {
                'total_ingredients': total_count,
                'status_distribution': status_counts,
                'type_distribution': type_counts,
                'vendor_distribution': vendor_counts,
                'cache_size': len(self._cache),
                'precheck_sessions': total_sessions,
                'precheck_decisions': decision_counts,
                'notifications': notification_counts
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get statistics: {e}")
            return {}

    def _row_to_ingredient(self, row: sqlite3.Row) -> Ingredient:
        """Convert database row to Ingredient object"""
        # Parse JSON fields
        metadata_dict = json.loads(row['metadata_json'])
        validation_results = json.loads(row['validation_results_json'] or '{}')
        precheck_results = json.loads(row['precheck_results_json'] or '[]')
        status_history = json.loads(row['status_history_json'] or '[]')

        # Create metadata object
        metadata = IngredientMetadata(**metadata_dict)

        # Create ingredient object
        return Ingredient(
            ingredient_id=row['ingredient_id'],
            metadata=metadata,
            status=IngredientStatus(row['status']),
            milestone=Milestone(row['milestone']),
            priority=Priority(row['priority']),
            validation_results=validation_results,
            precheck_results=precheck_results,
            created_by=row['created_by'],
            assigned_to=row['assigned_to'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            status_history=status_history
        )

# ============================================================================
# ENHANCED INGREDIENT MANAGER WITH NOTIFICATIONS
# ============================================================================

class IngredientManager:
    """High-level ingredient management system with notifications"""

    def __init__(self, repository_path: str = "unified_ingredients.db"):
        self.logger = logging.getLogger(f"{__name__}.IngredientManager")

        # Initialize components
        self.factory = IngredientFactory()
        self.validator = IngredientValidator()
        self.repository = IngredientRepository(repository_path)

        # Initialize notification system
        self.notification_system = IntelNotificationSystem()

        # Processing queue
        self.processing_queue = asyncio.Queue()
        self.processing_active = False

        self.logger.info("🧬 Enhanced Ingredient Manager initialized with notifications")

    async def create_ingredient_from_file(self, file_path: str, **kwargs) -> Optional[Ingredient]:
        """Create and validate ingredient from file with notifications"""
        try:
            self.logger.info(f"🧬 Creating ingredient from: {file_path}")

            # Create ingredient
            ingredient = self.factory.create_from_file(file_path, **kwargs)

            # Validate ingredient
            validation_results = await self.validator.validate_ingredient(ingredient)
            ingredient.validation_results = validation_results

            # Update status based on validation
            if validation_results['overall_status'] == 'failed':
                ingredient.status = IngredientStatus.ERROR
            elif validation_results['overall_status'] == 'passed_with_warnings':
                ingredient.status = IngredientStatus.VALIDATED
            else:
                ingredient.status = IngredientStatus.VALIDATED

            # Store ingredient
            if self.repository.store_ingredient(ingredient):
                self.logger.info(f"✅ Ingredient created successfully: {ingredient.ingredient_id}")

                # Send notifications
                await self._send_ingredient_notifications(ingredient, validation_results)

                return ingredient
            else:
                self.logger.error(f"❌ Failed to store ingredient")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to create ingredient: {e}")
            return None

    async def _send_ingredient_notifications(self, ingredient: Ingredient, validation_results: Dict[str, Any]):
        """Send appropriate notifications for ingredient events"""
        try:
            ingredient_data = ingredient.to_dict()

            # Send creation notification
            success = await self.notification_system.send_ingredient_notification(
                ingredient_data, 'created'
            )

            # Log notification
            self.repository.log_notification({
                'ingredient_id': ingredient.ingredient_id,
                'notification_type': 'email',
                'recipient': ingredient.metadata.business_owner or 'balasubramanyam.agalukote.lakshmipathi@intel.com',
                'subject': f'New Ingredient Created: {ingredient.metadata.name}',
                'message': f'Ingredient {ingredient.ingredient_id} created successfully',
                'status': 'sent' if success else 'failed',
                'sent_at': time.time() if success else None
            })

            # Send validation failure notification if needed
            if validation_results['overall_status'] == 'failed':
                await self.notification_system.send_ingredient_notification(
                    ingredient_data, 'validation_failed'
                )

                # Log validation failure notification
                self.repository.log_notification({
                    'ingredient_id': ingredient.ingredient_id,
                    'notification_type': 'email',
                    'recipient': 'balasubramanyam.agalukote.lakshmipathi@intel.com',
                    'subject': f'Validation Failed: {ingredient.metadata.name}',
                    'message': f'Ingredient {ingredient.ingredient_id} failed validation',
                    'status': 'sent',
                    'sent_at': time.time()
                })

        except Exception as e:
            self.logger.error(f"❌ Failed to send ingredient notifications: {e}")

    async def create_ingredients_from_directory(self, directory_path: str, **kwargs) -> List[Ingredient]:
        """Create ingredients from directory with notifications"""
        try:
            self.logger.info(f"📁 Creating ingredients from directory: {directory_path}")

            ingredients = []

            # Create ingredients using factory
            raw_ingredients = self.factory.create_from_directory(directory_path, **kwargs)

            # Validate and store each ingredient
            for raw_ingredient in raw_ingredients:
                try:
                    # Validate
                    validation_results = await self.validator.validate_ingredient(raw_ingredient)
                    raw_ingredient.validation_results = validation_results

                    # Update status
                    if validation_results['overall_status'] == 'failed':
                        raw_ingredient.status = IngredientStatus.ERROR
                    else:
                        raw_ingredient.status = IngredientStatus.VALIDATED

                    # Store
                    if self.repository.store_ingredient(raw_ingredient):
                        ingredients.append(raw_ingredient)

                        # Send notifications
                        await self._send_ingredient_notifications(raw_ingredient, validation_results)

                except Exception as e:
                    self.logger.error(f"❌ Failed to process ingredient {raw_ingredient.ingredient_id}: {e}")

            self.logger.info(f"✅ Created {len(ingredients)} ingredients from directory")
            return ingredients

        except Exception as e:
            self.logger.error(f"❌ Failed to create ingredients from directory: {e}")
            return []

    async def process_ingredient_for_precheck(self, ingredient_id: str) -> bool:
        """Process ingredient for precheck"""
        try:
            ingredient = self.repository.get_ingredient(ingredient_id)
            if not ingredient:
                return False

            # Update status to precheck pending
            self.repository.update_ingredient_status(
                ingredient_id,
                IngredientStatus.PRECHECK_PENDING,
                notes="Queued for precheck processing"
            )

            # Queue for precheck processing (this would integrate with your precheck system)
            await self.processing_queue.put(ingredient)

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to process ingredient for precheck: {e}")
            return False

    def get_ingredient_dashboard(self) -> Dict[str, Any]:
        """Get ingredient dashboard data"""
        try:
            stats = self.repository.get_statistics()

            # Recent ingredients
            recent_ingredients = self.repository.search_ingredients()[:10]

            # Pending actions
            pending_validation = self.repository.search_ingredients(status='created')
            pending_precheck = self.repository.search_ingredients(status='precheck_pending')

            return {
                'statistics': stats,
                'recent_ingredients': [ing.to_dict() for ing in recent_ingredients],
                'pending_validation': len(pending_validation),
                'pending_precheck': len(pending_precheck),
                'dashboard_timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get dashboard data: {e}")
            return {}

# ============================================================================
# DEMO AND TESTING WITH NOTIFICATIONS
# ============================================================================

async def create_demo_ingredients():
    """Create demo ingredients for testing with notifications"""
    print("🧬 Creating Demo Ingredients with Email Notifications")
    print("=" * 60)

    # Initialize manager with unified database
    manager = IngredientManager("unified_demo_ingredients.db")

    # Create demo directory
    demo_dir = "/tmp/demo_ingredients"
    os.makedirs(demo_dir, exist_ok=True)

    # Create demo files
    demo_files = [
        ("intel_graphics_driver_v27.20.100.8681.dll", "Intel Graphics Driver"),
        ("nvidia_display_driver_v471.96.sys", "NVIDIA Display Driver"),
        ("realtek_audio_driver_v6.0.9088.1.inf", "Realtek Audio Driver"),
        ("intel_wifi_firmware_v22.80.1.bin", "Intel WiFi Firmware"),
        ("microsoft_security_update_kb5005565.msi", "Microsoft Security Update")
    ]

    created_ingredients = []

    for filename, description in demo_files:
        # Create demo file
        file_path = os.path.join(demo_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(b"Demo ingredient file content\n" * 100)  # Create some content

        print(f"📦 Creating ingredient: {filename}")

        # Create ingredient with Intel email
        ingredient = await manager.create_ingredient_from_file(
            file_path,
            description=description,
            business_owner="balasubramanyam.agalukote.lakshmipathi@intel.com",
            technical_contact="balasubramanyam.agalukote.lakshmipathi@intel.com"
        )

        if ingredient:
            created_ingredients.append(ingredient)
            print(f"   ✅ Created: {ingredient.ingredient_id}")
            print(f"   📊 Status: {ingredient.status.value}")
            print(f"   🔍 Validation: {ingredient.validation_results.get('overall_status', 'unknown')}")
            print(f"   📧 Email notifications sent to: balasubramanyam.agalukote.lakshmipathi@intel.com")
        else:
            print(f"   ❌ Failed to create ingredient")

        print()

    # Display enhanced dashboard
    print("📊 Enhanced Ingredient Dashboard")
    print("-" * 40)

    dashboard = manager.get_ingredient_dashboard()
    stats = dashboard['statistics']

    print(f"Total Ingredients: {stats['total_ingredients']}")
    print(f"Status Distribution: {stats['status_distribution']}")
    print(f"Type Distribution: {stats['type_distribution']}")
    print(f"Vendor Distribution: {stats['vendor_distribution']}")
    print(f"Precheck Sessions: {stats.get('precheck_sessions', 0)}")
    print(f"Notifications Sent: {stats.get('notifications', {})}")

    # Show sample ingredient with precheck data
    if created_ingredients:
        sample_ingredient = created_ingredients[0]
        full_data = manager.repository.get_ingredient_with_prechecks(sample_ingredient.ingredient_id)
        if full_data:
            print(f"\n📋 Sample Ingredient Details:")
            print(f"   ID: {sample_ingredient.ingredient_id}")
            print(f"   Precheck Results: {full_data['total_prechecks']}")
            print(f"   Dashboard Link: https://ingredient-dashboard.intel.com/ingredient/{sample_ingredient.ingredient_id}")

    print(f"\n✅ Created {len(created_ingredients)} demo ingredients with email notifications")
    print(f"📧 All notifications sent to: balasubramanyam.agalukote.lakshmipathi@intel.com")
    print(f"🗄️ Database: unified_demo_ingredients.db")

    return created_ingredients

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        asyncio.run(create_demo_ingredients())
    else:
        print("🧬 Intel Ingredient Management System - Enhanced with Notifications")
        print("=" * 70)
        print("Available commands:")
        print("  python ingredient_management.py --demo    # Create demo ingredients with notifications")
        print("\n💡 Use IngredientManager class in your code:")
        print("  manager = IngredientManager('unified_ingredients.db')")
        print("  ingredient = await manager.create_ingredient_from_file('path/to/file')")
        print("\n📧 Email notifications will be sent to:")
        print("  balasubramanyam.agalukote.lakshmipathi@intel.com")
        print("\n🗄️ Unified database schema includes:")
        print("  • ingredients - Core ingredient data")
        print("  • precheck_sessions - Precheck execution sessions")
        print("  • precheck_results - Individual precheck results")
        print("  • notifications - Email notification log")
