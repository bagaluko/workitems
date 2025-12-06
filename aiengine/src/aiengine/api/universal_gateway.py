#!/usr/bin/env python3
"""
Universal API Gateway - Unified entry point for all AI services
Integrates FastAPI (Wiki), Flask (GitHub), and Universal Neural System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
try:
    from config.settings import get_default_host, get_env_int, logger
    CONFIG_AVAILABLE = True
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    CONFIG_AVAILABLE = False

# Import the bridge
from .universal_bridge import universal_bridge

app = FastAPI(
    title="Universal AI Gateway",
    description="Unified API gateway for all AI services including Wiki Q&A, GitHub integration, and Universal Neural System",
    version="2.0.0"
)

# Request/Response Models
class UniversalTaskRequest(BaseModel):
    domain: str
    task_type: str
    input_data: Any
    metadata: Optional[Dict] = {}
    priority: Optional[int] = 1

class WikiQuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 1000
    include_sources: Optional[bool] = True
    enhanced: Optional[bool] = True

class GitHubAnalysisRequest(BaseModel):
    owner: str
    repo: str
    commit_data: Optional[Dict] = {}
    analysis_type: Optional[str] = "standard"

class PrecheckRequest(BaseModel):
    precheck_data: Dict
    environment: Optional[str] = "development"
    enhanced_analysis: Optional[bool] = True

# ============================================================================
# UNIVERSAL NEURAL SYSTEM ENDPOINTS
# ============================================================================

@app.post("/api/universal/process_task")
async def process_universal_task(request: UniversalTaskRequest):
    """Process any task using the Universal Neural System"""

    if not universal_bridge.is_available():
        raise HTTPException(
            status_code=503,
            detail="Universal Neural System not available"
        )

    try:
        result = await universal_bridge.process_task(
            domain=request.domain,
            task_type=request.task_type,
            input_data=request.input_data,
            metadata=request.metadata
        )
        return result

    except Exception as e:
        logger.error(f"Universal task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/universal/status")
async def get_universal_status():
    """Get comprehensive Universal Neural System status"""
    return await universal_bridge.get_system_status()

@app.get("/api/universal/capabilities")
async def get_capabilities():
    """Get supported domains and task types"""
    if not universal_bridge.is_available():
        raise HTTPException(status_code=503, detail="Universal System not available")

    try:
        from core.universal_types import DomainType, TaskType
        return {
            'domains': [domain.value for domain in DomainType],
            'task_types': [task_type.value for task_type in TaskType],
            'total_domains': len(DomainType),
            'total_task_types': len(TaskType),
            'universal_system_available': True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENHANCED WIKI Q&A ENDPOINTS
# ============================================================================

@app.post("/api/wiki/ask")
async def ask_wiki_question(request: WikiQuestionRequest):
    """Enhanced wiki question answering with Universal System integration"""

    try:
        # Try enhanced processing first if requested and available
        if request.enhanced and universal_bridge.is_available():
            try:
                enhanced_result = await universal_bridge.process_wiki_question_enhanced(
                    request.question,
                    request.max_tokens,
                    request.include_sources
                )

                if enhanced_result.get('success', False):
                    enhanced_result['processing_method'] = 'universal_neural_system'
                    return enhanced_result

            except Exception as e:
                logger.warning(f"Enhanced wiki processing failed, using standard: {e}")

        # Fallback to standard wiki API
        try:
            # Import and use existing wiki API
            from .wiki_qa_api import knowledge_base

            if not knowledge_base:
                raise HTTPException(status_code=503, detail="Wiki knowledge base not initialized")

            result = await knowledge_base.answer_question(
                question=request.question,
                max_tokens=request.max_tokens,
                include_sources=request.include_sources
            )

            result['processing_method'] = 'standard_wiki_api'
            result['success'] = True
            return result

        except Exception as e:
            logger.error(f"Standard wiki processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Wiki processing failed: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wiki/info")
async def get_wiki_info():
    """Get wiki knowledge base information"""
    try:
        from .wiki_qa_api import knowledge_base

        if not knowledge_base:
            return {"available": False, "error": "Knowledge base not initialized"}

        return {
            "available": True,
            "total_pages": len(knowledge_base.pages),
            "extraction_timestamp": knowledge_base.wiki_data.get("extracted_at"),
            "universal_system_enhanced": universal_bridge.is_available()
        }

    except Exception as e:
        return {"available": False, "error": str(e)}

# ============================================================================
# GITHUB INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/api/github/analyze")
async def analyze_github_repository(request: GitHubAnalysisRequest):
    """Analyze GitHub repository with optional Universal System enhancement"""

    try:
        # Standard GitHub analysis (you'll need to implement this based on your github_api.py)
        github_result = {
            "owner": request.owner,
            "repo": request.repo,
            "analysis_type": request.analysis_type,
            "timestamp": datetime.now().isoformat(),
            "standard_analysis": "GitHub analysis would go here"
        }

        # Enhanced analysis using Universal System if available
        if universal_bridge.is_available():
            try:
                enhanced_analysis = await universal_bridge.process_task(
                    domain="software_development",
                    task_type="code_analysis",
                    input_data={
                        "repository": f"{request.owner}/{request.repo}",
                        "commit_data": request.commit_data,
                        "analysis_type": request.analysis_type
                    },
                    metadata={"source": "github_api", "enhanced": True}
                )

                github_result["enhanced_analysis"] = enhanced_analysis
                github_result["processing_method"] = "universal_enhanced"

            except Exception as e:
                logger.warning(f"Enhanced GitHub analysis failed: {e}")
                github_result["processing_method"] = "standard_only"
        else:
            github_result["processing_method"] = "standard_only"

        return github_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/github/repositories")
async def list_github_repositories():
    """List configured GitHub repositories"""
    # This would integrate with your existing github_api.py
    return {
        "repositories": [],  # Populate from your GitHub service
        "message": "GitHub repository listing - integrate with existing github_api.py"
    }

# ============================================================================
# PRECHECK INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/api/precheck/analyze")
async def analyze_precheck_data(request: PrecheckRequest):
    """Enhanced precheck analysis using Universal System"""

    try:
        # Standard precheck processing
        standard_result = {
            "environment": request.environment,
            "timestamp": datetime.now().isoformat(),
            "standard_analysis": "Standard precheck analysis"
        }

        # Enhanced analysis using Universal System
        if request.enhanced_analysis and universal_bridge.is_available():
            try:
                enhanced_result = await universal_bridge.process_task(
                    domain="precheck_validation",
                    task_type="precheck_analysis",
                    input_data=request.precheck_data,
                    metadata={
                        "environment": request.environment,
                        "enhanced": True,
                        "source": "precheck_api"
                    }
                )

                return {
                    **standard_result,
                    "enhanced_analysis": enhanced_result,
                    "processing_method": "universal_enhanced"
                }

            except Exception as e:
                logger.warning(f"Enhanced precheck analysis failed: {e}")
                return {
                    **standard_result,
                    "processing_method": "standard_only",
                    "enhancement_error": str(e)
                }

        return {
            **standard_result,
            "processing_method": "standard_only"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/precheck/health")
async def precheck_health_check():
    """Check precheck service health"""
    return {
        "status": "healthy",
        "universal_system_available": universal_bridge.is_available(),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# UNIFIED SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def comprehensive_health_check():
    """Comprehensive health check for all services"""

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "services": {}
    }

    # Check Universal System
    health_status["services"]["universal_system"] = {
        "available": universal_bridge.is_available(),
        "status": "healthy" if universal_bridge.is_available() else "unavailable"
    }

    # Check Wiki Knowledge Base
    try:
        from .wiki_qa_api import knowledge_base
        health_status["services"]["wiki_kb"] = {
            "available": knowledge_base is not None,
            "pages_loaded": len(knowledge_base.pages) if knowledge_base else 0,
            "status": "healthy" if knowledge_base else "unavailable"
        }
    except:
        health_status["services"]["wiki_kb"] = {
            "available": False,
            "status": "error"
        }

    # Check GitHub integration
    health_status["services"]["github"] = {
        "available": True,  # Update based on your github_api.py status
        "status": "healthy"
    }

    # Check Precheck service
    health_status["services"]["precheck"] = {
        "available": True,
        "status": "healthy"
    }

    # Determine overall status
    service_statuses = [service["status"] for service in health_status["services"].values()]
    if "error" in service_statuses:
        health_status["overall_status"] = "degraded"
    elif "unavailable" in service_statuses:
        health_status["overall_status"] = "partial"

    return health_status

@app.get("/api/info")
async def get_api_info():
    """Get comprehensive API information"""
    return {
        "title": "Universal AI Gateway",
        "version": "2.0.0",
        "description": "Unified API gateway for all AI services",
        "universal_system_available": universal_bridge.is_available(),
        "endpoints": {
            "universal": [
                "POST /api/universal/process_task",
                "GET /api/universal/status",
                "GET /api/universal/capabilities"
            ],
            "wiki": [
                "POST /api/wiki/ask",
                "GET /api/wiki/info"
            ],
            "github": [
                "POST /api/github/analyze",
                "GET /api/github/repositories"
            ],
            "precheck": [
                "POST /api/precheck/analyze",
                "GET /api/precheck/health"
            ],
            "system": [
                "GET /api/health",
                "GET /api/info"
            ]
        },
        "configuration": {
            "config_available": CONFIG_AVAILABLE,
            "host": os.getenv('UNIVERSAL_GATEWAY_HOST', 'localhost'),
            "port": int(os.getenv('UNIVERSAL_GATEWAY_PORT', '8080'))
        }
    }

# ============================================================================
# STARTUP AND CONFIGURATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Universal AI Gateway starting up...")
    logger.info(f"🔧 Universal System available: {universal_bridge.is_available()}")

    # Initialize wiki knowledge base if not already done
    try:
        from .wiki_qa_api import knowledge_base
        if knowledge_base:
            logger.info(f"📚 Wiki KB loaded: {len(knowledge_base.pages)} pages")
    except:
        logger.warning("⚠️ Wiki knowledge base not available")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv('UNIVERSAL_GATEWAY_HOST', 'localhost')
    port = int(os.getenv('UNIVERSAL_GATEWAY_PORT', '8080'))

    logger.info(f"🚀 Starting Universal AI Gateway on {host}:{port}")
    logger.info(f"📚 API docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
