"""
Standalone GitHub Integration Service
Manages multiple repositories dynamically
"""

import yaml
import os
from typing import Dict, List
from flask import Flask, request, jsonify

class GitHubIntegrationService:
    def __init__(self, config_file="config/github_config.yaml"):
        self.config = self.load_config(config_file)
        self.repositories = {}
        self.load_repositories()

    def load_config(self, config_file):
        """Load GitHub configuration from YAML"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def add_repository(self, repo_config):
        """Dynamically add a new repository"""
        repo_key = f"{repo_config['owner']}/{repo_config['name']}"
        self.repositories[repo_key] = repo_config
        return f"Repository {repo_key} added successfully"

    def remove_repository(self, owner, name):
        """Remove a repository from monitoring"""
        repo_key = f"{owner}/{name}"
        if repo_key in self.repositories:
            del self.repositories[repo_key]
            return f"Repository {repo_key} removed"
        return f"Repository {repo_key} not found"

    def analyze_repository(self, owner, name, commit_data):
        """Analyze specific repository"""
        repo_key = f"{owner}/{name}"
        if repo_key not in self.repositories:
            return {"error": f"Repository {repo_key} not configured"}

        # Call your AI engine for analysis
        return self.perform_ai_analysis(commit_data, self.repositories[repo_key])
