#!/usr/bin/env python3
"""
Wiki Knowledge Base Question-Answering Module
Integrated with Universal AI Engine and Neural Fallback System
"""

import json
import os
import requests
import asyncio
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

from config.settings import logger, get_env_int, security_config

class WikiKnowledgeBase:
    """Wiki Knowledge Base integrated with Universal Neural System"""

    def __init__(self, universal_system=None):
        """Initialize the knowledge base with optional Universal Neural System integration"""
        self.wiki_data = None
        self.pages = {}
        self.initialized = False

        # Universal Neural System integration for intelligent fallback
        self.universal_system = universal_system

        # Azure OpenAI configuration from environment
        self.api_base = os.getenv('AZURE_OPENAI_BASE_URL', '<<< Azure Base URL here >>>')
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY', '<<<< Update your token here >>>>')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', '<<Update here>>')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-05-01-preview')

        # Build the full endpoint URL
        self.endpoint_url = f"{self.api_base}openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"

        # Configuration
        self.max_tokens_default = get_env_int('WIKI_QA_MAX_TOKENS', 1000)
        self.timeout = get_env_int('WIKI_QA_TIMEOUT', 30)

        # Performance tracking
        self.response_stats = {
            'azure_responses': 0,
            'neural_responses': 0,
            'fallback_responses': 0,
            'total_questions': 0
        }

        # Initialize knowledge base
        self.initialize()

    def initialize(self):
        """Initialize the knowledge base from extracted wiki content with enhanced debugging"""
        try:
            logger.info("🔧 Starting wiki initialization...")

            # Look for wiki extraction files
            extraction_files = self._find_extraction_files()
            logger.info(f"📊 Found extraction files: {extraction_files}")

            if not extraction_files:
                logger.warning("❌ No wiki extraction files found")
                logger.info("📁 Searched directories:")
                search_dirs = [
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'),
                    os.path.dirname(os.path.dirname(__file__)),
                    os.getcwd()
                ]
                for search_dir in search_dirs:
                    logger.info(f"  - {search_dir} (exists: {os.path.exists(search_dir)})")
                    if os.path.exists(search_dir):
                        wiki_files = [f for f in os.listdir(search_dir) if 'wiki' in f.lower()]
                        logger.info(f"    Wiki files: {wiki_files}")
                return False

            # Use the most recent extraction file
            latest_file = sorted(extraction_files, key=os.path.getmtime)[-1]
            logger.info(f"🚀 Loading wiki knowledge base: {os.path.basename(latest_file)}")

            self.load_wiki_data(latest_file)
            logger.info(f"📊 After loading: {len(self.pages)} pages loaded")

            if self.pages:
                self.initialized = True
                logger.info(f"✅ Wiki knowledge base initialized with {len(self.pages)} pages")

                # Log integration status
                if self.universal_system:
                    logger.info("🧠 Universal Neural System integration enabled")
                else:
                    logger.info("📄 Running in standalone mode (no neural integration)")

                # Show sample pages
                for i, (page_id, page_data) in enumerate(list(self.pages.items())[:3]):
                    logger.info(f"  📄 {page_id}: {page_data['title']} ({page_data['content_length']} chars)")

                return True
            else:
                logger.warning("⚠️ Wiki knowledge base loaded but no pages found")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to initialize wiki knowledge base: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_extraction_files(self) -> List[str]:
        """Find wiki extraction files in multiple locations"""
        search_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'),
            os.path.dirname(os.path.dirname(__file__)),
            os.getcwd(),
            '/aiengine/src/aiengine/config',
            './config'
        ]

        extraction_files = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                try:
                    files = [f for f in os.listdir(search_dir) if f.startswith('wiki_extraction') and f.endswith('.json')]
                    extraction_files.extend([os.path.join(search_dir, f) for f in files])
                except PermissionError:
                    continue

        return extraction_files

    def load_wiki_data(self, extraction_file: str):
        """Load extracted wiki data from JSON file with adaptive parsing"""
        try:
            with open(extraction_file, 'r', encoding='utf-8') as f:
                self.wiki_data = json.load(f)

            logger.info(f"📊 Raw wiki data keys: {list(self.wiki_data.keys())}")

            # Build searchable pages dictionary with adaptive parsing
            pages_data = self.wiki_data.get('pages', {})

            if not pages_data:
                # Try alternative structures
                if 'data' in self.wiki_data:
                    pages_data = self.wiki_data['data']
                elif 'content' in self.wiki_data:
                    pages_data = self.wiki_data['content']
                elif 'wiki_pages' in self.wiki_data:
                    pages_data = self.wiki_data['wiki_pages']

            logger.info(f"📊 Found {len(pages_data)} pages in data")

            for page_id, page_data in pages_data.items():
                try:
                    # Adaptive parsing for different structures
                    if isinstance(page_data, dict):
                        # Check if it's the expected structure
                        if page_data.get('success', True) and 'page_info' in page_data:
                            # Standard structure
                            self.pages[page_id] = {
                                'title': page_data['page_info']['title'],
                                'content': page_data.get('content', ''),
                                'url': page_data['page_info'].get('web_url', ''),
                                'version': page_data['page_info'].get('version', '1.0'),
                                'content_length': page_data.get('content_length', len(page_data.get('content', '')))
                            }
                        elif 'title' in page_data and 'content' in page_data:
                            # Direct structure
                            self.pages[page_id] = {
                                'title': page_data['title'],
                                'content': page_data['content'],
                                'url': page_data.get('url', page_data.get('web_url', '')),
                                'version': page_data.get('version', '1.0'),
                                'content_length': page_data.get('content_length', len(page_data.get('content', '')))
                            }
                        elif isinstance(page_data, str):
                            # Content-only structure
                            self.pages[page_id] = {
                                'title': f"Page {page_id}",
                                'content': page_data,
                                'url': '',
                                'version': '1.0',
                                'content_length': len(page_data)
                            }
                        else:
                            logger.warning(f"⚠️ Unrecognized page structure for {page_id}: {list(page_data.keys())}")

                except Exception as e:
                    logger.warning(f"⚠️ Error parsing page {page_id}: {e}")
                    continue

            logger.info(f"✅ Successfully loaded {len(self.pages)} wiki pages")

            # Show sample pages for debugging
            for page_id, page in list(self.pages.items())[:3]:
                logger.info(f"  - {page_id}: {page['title']} ({page['content_length']} chars)")

            if len(self.pages) > 3:
                logger.info(f"  ... and {len(self.pages) - 3} more pages")

        except Exception as e:
            logger.error(f"❌ Error loading wiki data: {e}")
            self.wiki_data = {"pages": {}}
            self.pages = {}

    def search_relevant_content(self, question: str, max_pages: int = 3) -> List[Dict]:
        """Search for relevant wiki pages based on the question with enhanced scoring"""
        if not self.initialized:
            return []

        question_lower = question.lower()
        relevant_pages = []

        for page_id, page in self.pages.items():
            title_lower = page['title'].lower()
            content_lower = page['content'].lower()

            # Calculate relevance score
            score = 0

            # Title matches are highly relevant
            for word in question_lower.split():
                if len(word) > 2:  # Skip short words
                    if word in title_lower:
                        score += 10
                    if word in content_lower:
                        score += 1

            # Boost score for specific topics - FIXED: Use tuples as keys instead of lists
            topic_boosts = {
                ('precheck', 'prechecks'): 20,
                ('platform', 'ci', 'build'): 15,
                ('bkc', 'configuration'): 12,
                ('abi', 'automation'): 10,
                ('bronze', 'silver', 'gold'): 8,
                ('ingredient', 'ingredients'): 12,
                ('manifest', 'kit'): 10,
                ('milestone', 'gatekeeper'): 8,
                ('onebkc', 'wit'): 6,
                ('driver', 'firmware'): 7,
                ('test', 'testing', 'validation'): 9,
                ('build', 'compilation'): 8
            }

            for keywords, boost in topic_boosts.items():
                if any(keyword in question_lower for keyword in keywords):
                    if any(term in title_lower or term in content_lower for term in keywords):
                        score += boost

            # Additional scoring for question types
            question_type_boosts = {
                ('what', 'define', 'definition'): 5,
                ('how', 'process', 'workflow'): 8,
                ('why', 'reason', 'purpose'): 6,
                ('when', 'schedule', 'timing'): 4,
                ('where', 'location', 'place'): 3
            }

            for question_words, boost in question_type_boosts.items():
                if any(qword in question_lower for qword in question_words):
                    score += boost

            if score > 0:
                relevant_pages.append({
                    'page_id': page_id,
                    'title': page['title'],
                    'content': page['content'],
                    'url': page['url'],
                    'relevance_score': score
                })

        # Sort by relevance and return top results
        relevant_pages.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_pages[:max_pages]

    def create_context_prompt(self, question: str, relevant_pages: List[Dict]) -> str:
        """Create a context-rich prompt for the LLM or Neural System"""
        context_sections = []
        for page in relevant_pages:
            # Truncate content to fit within token limits
            content_preview = page['content'][:1500] + "..." if len(page['content']) > 1500 else page['content']
            context_sections.append(f"""
**Source: {page['title']}**
**URL: {page['url']}**
**Relevance Score: {page['relevance_score']}**
**Content:**
{content_preview}
""")

        context = "\n".join(context_sections)

        prompt = f"""You are a knowledgeable assistant helping users understand Intel Platform CI processes. Use the following wiki documentation to answer the user's question comprehensively and accurately.

**CONTEXT FROM INTEL WIKI PAGES:**
{context}

**USER QUESTION:**
{question}

**INSTRUCTIONS:**
1. Answer the question using ONLY the information provided in the context above
2. Be specific and detailed in your response
3. If the context doesn't contain enough information to fully answer the question, state what information is available and what is missing
4. Reference the relevant wiki page titles when citing information
5. Use clear, professional language appropriate for technical documentation
6. Keep the response concise but comprehensive
7. Structure your response with clear sections if the answer is complex

**ANSWER:**"""

        return prompt

    async def get_llm_response(self, prompt: str, max_tokens: int = None) -> str:
        """Get response from Azure OpenAI LLM with Universal Neural System fallback"""
        if max_tokens is None:
            max_tokens = self.max_tokens_default

        # First, try Azure OpenAI if properly configured
        if self.api_key and self.api_key != '<<<API Key here>>>':
            try:
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert assistant for Intel Platform CI processes. Provide accurate, detailed answers based on the provided wiki documentation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9
                }

                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                }

                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()
                response_data = response.json()

                if 'choices' in response_data and len(response_data['choices']) > 0:
                    self.response_stats['azure_responses'] += 1
                    logger.info("✅ Azure OpenAI response generated successfully")
                    return response_data['choices'][0]['message']['content'].strip()

            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Azure OpenAI connection failed: {str(e)[:100]}...")
            except Exception as e:
                logger.warning(f"⚠️ Azure OpenAI error: {str(e)[:100]}...")

        # Fallback to Universal Neural System
        if self.universal_system:
            logger.info("🧠 Using Universal Neural System for wiki response generation")
            self.response_stats['neural_responses'] += 1
            return await self._get_neural_response(prompt, max_tokens)
        else:
            logger.info("📄 Using basic fallback response")
            self.response_stats['fallback_responses'] += 1
            return self._generate_fallback_response(prompt)

    async def _get_neural_response(self, prompt: str, max_tokens: int) -> str:
        """Generate response using the Universal Neural System"""
        try:
            from core.universal_types import UniversalTask, DomainType, TaskType

            # Extract the question from the prompt
            question_match = re.search(r'\*\*USER QUESTION:\*\*\s*(.+?)\s*\*\*', prompt, re.DOTALL)
            question = question_match.group(1).strip() if question_match else "wiki question"

            # Extract context from prompt
            context_match = re.search(r'\*\*CONTEXT FROM INTEL WIKI PAGES:\*\*\s*(.+?)\s*\*\*USER QUESTION:', prompt, re.DOTALL)
            context = context_match.group(1).strip() if context_match else ""

            # Determine the best task type based on question
            task_type = self._determine_task_type(question)

            # Create a task for the Universal Neural System
            wiki_task = UniversalTask(
                task_id=f"wiki_qa_{int(time.time())}",
                domain=DomainType.NATURAL_LANGUAGE,
                task_type=task_type,
                input_data={
                    'question': question,
                    'context': context,
                    'wiki_pages': len(self.pages),
                    'task_type': 'wiki_question_answering',
                    'max_tokens': max_tokens,
                    'prompt_length': len(prompt),
                    'context_sources': len(re.findall(r'\*\*Source:', context))
                },
                metadata={
                    'source': 'wiki_knowledge_base',
                    'fallback_mode': True,
                    'context_length': len(context),
                    'question_type': self._classify_question_type(question),
                    'processing_mode': 'neural_fallback'
                }
            )

            # Process the task using the Universal Neural System
            solution = self.universal_system.process_universal_task(wiki_task)

            if solution.confidence > 0.3:
                # Format the neural response for wiki context
                neural_response = self._format_neural_wiki_response(
                    question, context, solution.solution, solution.confidence, solution.reasoning
                )
                logger.info(f"🧠 Neural response generated with confidence: {solution.confidence:.3f}")
                return neural_response
            else:
                logger.warning(f"⚠️ Neural response confidence too low: {solution.confidence:.3f}, using fallback")
                return self._generate_fallback_response(prompt)

        except Exception as e:
            logger.error(f"❌ Neural response generation failed: {e}")
            return self._generate_fallback_response(prompt)

    def _determine_task_type(self, question: str) -> 'TaskType':
        """Determine the best task type based on the question"""
        from core.universal_types import TaskType

        question_lower = question.lower()

        if any(word in question_lower for word in ['generate', 'create', 'write', 'compose']):
            return TaskType.TEXT_GENERATION
        elif any(word in question_lower for word in ['classify', 'categorize', 'type', 'kind']):
            return TaskType.CLASSIFICATION
        elif any(word in question_lower for word in ['analyze', 'sentiment', 'opinion']):
            return TaskType.SENTIMENT_ANALYSIS
        elif any(word in question_lower for word in ['recommend', 'suggest', 'advise']):
            return TaskType.RECOMMENDATION
        else:
            return TaskType.NATURAL_LANGUAGE_PROCESSING

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()

        if question_lower.startswith(('what', 'define')):
            return 'definition'
        elif question_lower.startswith(('how', 'explain')):
            return 'process_explanation'
        elif question_lower.startswith(('why', 'reason')):
            return 'reasoning'
        elif question_lower.startswith(('when', 'schedule')):
            return 'timing'
        elif question_lower.startswith(('where', 'location')):
            return 'location'
        elif question_lower.startswith(('who', 'responsible')):
            return 'responsibility'
        else:
            return 'general_inquiry'

    def _format_neural_wiki_response(self, question: str, context: str, neural_output: Any,
                                   confidence: float, reasoning: str = "") -> str:
        """Format the neural network output into a proper wiki response"""
        try:
            # Extract source information from context
            source_matches = re.findall(r'\*\*Source: (.+?)\*\*', context)
            sources = ", ".join(source_matches) if source_matches else "Intel Platform CI documentation"

            # Clean and format the context
            clean_context = re.sub(r'\*\*[^*]+\*\*', '', context)
            clean_context = re.sub(r'\n+', ' ', clean_context).strip()

            # Format the neural output
            if isinstance(neural_output, dict):
                if 'answer' in neural_output:
                    answer = neural_output['answer']
                elif 'response' in neural_output:
                    answer = neural_output['response']
                elif 'solution' in neural_output:
                    answer = neural_output['solution']
                else:
                    answer = str(neural_output)
            else:
                answer = str(neural_output)

            # Enhance the answer with neural reasoning
            enhanced_answer = self._enhance_neural_answer(answer, reasoning, confidence)

            # Create a comprehensive response
            response = f"""**Intel Platform CI Analysis - AI-Generated Response**

**Question:** {question}

**AI Analysis:**
{enhanced_answer}

**Supporting Documentation:**
{clean_context[:600]}{'...' if len(clean_context) > 600 else ''}

**Response Metadata:**
- **AI Engine:** Universal Neural Network
- **Confidence Level:** {confidence:.1%}
- **Sources:** {len(source_matches)} Intel wiki pages ({sources})
- **Processing Mode:** Neural fallback (Azure OpenAI unavailable)
- **Reasoning:** {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}

*This response combines the Universal AI Engine's neural network analysis with Intel Platform CI documentation content.*"""

            return response

        except Exception as e:
            logger.warning(f"Response formatting failed: {e}")
            return f"""**AI Engine Analysis**

{answer}

**Source:** Intel Platform CI documentation
**Confidence:** {confidence:.1%}
**Processing:** Universal Neural Network

*Note: Response formatting encountered an issue, showing simplified format.*"""

    def _enhance_neural_answer(self, answer: str, reasoning: str, confidence: float) -> str:
        """Enhance the neural answer with additional context and formatting"""
        try:
            # Add confidence indicators
            confidence_indicator = ""
            if confidence > 0.8:
                confidence_indicator = "🟢 High confidence analysis"
            elif confidence > 0.6:
                confidence_indicator = "🟡 Moderate confidence analysis"
            else:
                confidence_indicator = "🟠 Lower confidence analysis - consider additional verification"

            # Format the answer with structure
            enhanced = f"""{confidence_indicator}

{answer}

**Neural Network Reasoning:**
{reasoning}

**Recommendation:** This analysis is based on Intel Platform CI documentation processed through the Universal Neural Network. For critical decisions, please verify with official Intel documentation or consult with platform engineering teams."""

            return enhanced

        except Exception as e:
            logger.warning(f"Answer enhancement failed: {e}")
            return answer

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a basic fallback response when both Azure and Neural systems fail"""
        try:
            # Extract the question from the prompt
            question_match = re.search(r'\*\*USER QUESTION:\*\*\s*(.+?)\s*\*\*', prompt, re.DOTALL)
            question = question_match.group(1).strip() if question_match else "your question"

            # Extract context information
            context_match = re.search(r'\*\*CONTEXT FROM INTEL WIKI PAGES:\*\*\s*(.+?)\s*\*\*USER QUESTION:', prompt, re.DOTALL)
            context = context_match.group(1).strip() if context_match else ""

            if context:
                # Extract source titles
                source_matches = re.findall(r'\*\*Source: (.+?)\*\*', context)
                sources = ", ".join(source_matches) if source_matches else "Intel wiki documentation"

                # Extract and clean content
                content_preview = re.sub(r'\*\*[^*]+\*\*', '', context)
                content_preview = re.sub(r'\n+', ' ', content_preview).strip()

                # Limit content length
                if len(content_preview) > 800:
                    content_preview = content_preview[:800] + "..."

                return f"""**Intel Platform CI Documentation Extract**

**Question:** {question}

**Available Information:**
{content_preview}

**Sources:** {sources}

**Note:** This is a direct extraction from Intel Platform CI wiki pages. Both Azure OpenAI and the Universal Neural Network are currently unavailable for enhanced AI analysis. For more detailed analysis, please ensure system components are properly configured."""
            else:
                return f"""**Intel Platform CI Knowledge Base**

**Question:** {question}

Information about '{question}' was found in the Intel Platform CI knowledge base, but detailed content extraction failed.

**Suggestions:**
1. Try rephrasing your question with more specific terms
2. Check if the question relates to: precheck processes, BKC configuration, platform ingredients, build workflows, or CI automation
3. Verify system configuration for enhanced AI responses

**Available Topics:** Platform CI processes, precheck validation, BKC management, ingredient handling, build automation, and CI workflows."""

        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return f"Error processing question about '{question}'. Please try again or contact system administrator."

    async def answer_question(self, question: str, max_tokens: int = None, include_sources: bool = True) -> Dict:
        """Main method to answer a question using the wiki knowledge base with neural fallback"""
        if not self.initialized:
            return {
                "answer": "Wiki knowledge base is not initialized. Please check if wiki extraction files are available.",
                "sources": [],
                "confidence_score": 0.0,
                "response_time": 0.0,
                "error": "knowledge_base_not_initialized",
                "processing_mode": "error"
            }

        start_time = datetime.now()
        self.response_stats['total_questions'] += 1

        try:
            # Find relevant wiki content
            relevant_pages = self.search_relevant_content(question, max_pages=3)

            if not relevant_pages:
                return {
                    "answer": "I couldn't find relevant information in the Intel Platform CI wiki knowledge base to answer your question. Please try rephrasing your question or asking about Platform CI processes, prechecks, BKC configuration, ingredients, or build workflows.",
                    "sources": [],
                    "confidence_score": 0.0,
                    "response_time": (datetime.now() - start_time).total_seconds(),
                    "processing_mode": "no_relevant_content",
                    "suggestions": [
                        "Try asking about precheck processes",
                        "Ask about BKC configuration",
                        "Inquire about platform ingredients",
                        "Ask about build workflows"
                    ]
                }

            # Create context prompt
            prompt = self.create_context_prompt(question, relevant_pages)

            # Get response (Azure OpenAI -> Neural System -> Fallback)
            answer = await self.get_llm_response(prompt, max_tokens)

            # Prepare response
            sources = []
            if include_sources:
                sources = [
                    {
                        "title": page['title'],
                        "url": page['url'],
                        "relevance_score": page['relevance_score'],
                        "content_length": len(page['content'])
                    }
                    for page in relevant_pages
                ]

            response_time = (datetime.now() - start_time).total_seconds()

            # Calculate confidence based on relevance and processing mode
            base_confidence = min(relevant_pages[0]['relevance_score'] / 50.0, 1.0) if relevant_pages else 0.0

            # Adjust confidence based on processing mode
            if self.response_stats['azure_responses'] > self.response_stats['neural_responses']:
                processing_mode = "azure_openai"
                confidence_multiplier = 1.0
            elif self.response_stats['neural_responses'] > 0:
                processing_mode = "neural_system"
                confidence_multiplier = 0.85  # Slightly lower for neural
            else:
                processing_mode = "fallback_extraction"
                confidence_multiplier = 0.7   # Lower for basic extraction

            final_confidence = min(base_confidence * confidence_multiplier, 1.0)

            return {
                "answer": answer,
                "sources": sources,
                "confidence_score": final_confidence,
                "response_time": response_time,
                "processing_mode": processing_mode,
                "relevant_pages_found": len(relevant_pages),
                "total_wiki_pages": len(self.pages),
                "response_stats": self.response_stats.copy()
            }

        except Exception as e:
            logger.error(f"Error processing wiki question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "response_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
                "processing_mode": "error"
            }

    def get_knowledge_base_info(self) -> Dict:
        """Get comprehensive information about the loaded knowledge base"""
        if not self.initialized:
            return {
                "initialized": False,
                "total_pages": 0,
                "pages": [],
                "neural_integration": self.universal_system is not None,
                "response_stats": self.response_stats
            }

        pages_info = []
        for page_id, page in self.pages.items():
            pages_info.append({
                "page_id": page_id,
                "title": page["title"],
                "content_length": page["content_length"],
                "url": page["url"],
                "version": page.get("version", "unknown")
            })

        # Get neural system info if available
        neural_info = {}
        if self.universal_system:
            try:
                system_status = self.universal_system.get_system_status()
                neural_info = {
                    "available": True,
                    "total_tasks_processed": system_status['performance_metrics']['total_tasks_processed'],
                    "average_confidence": system_status['performance_metrics']['average_confidence'],
                    "domains_mastered": len(system_status['performance_metrics']['domains_mastered']),
                    "system_health": system_status['system_health']['overall_score']
                }
            except Exception as e:
                neural_info = {"available": True, "error": str(e)}
        else:
            neural_info = {"available": False}

        return {
            "initialized": True,
            "total_pages": len(self.pages),
            "extraction_timestamp": self.wiki_data.get("extracted_at"),
            "successful_extractions": self.wiki_data.get("successful_extractions", 0),
            "failed_extractions": self.wiki_data.get("failed_extractions", 0),
            "pages": pages_info,
            "neural_integration": neural_info,
            "response_stats": self.response_stats,
            "azure_openai_configured": bool(self.api_key and self.api_key != '<< key here >>'),
            "capabilities": {
                "search": True,
                "question_answering": True,
                "neural_fallback": self.universal_system is not None,
                "azure_openai": bool(self.api_key and self.api_key != '<< key here >>'),
                "basic_extraction": True
            }
        }

    def get_response_statistics(self) -> Dict:
        """Get detailed response statistics"""
        total = self.response_stats['total_questions']
        if total == 0:
            return {
                "total_questions": 0,
                "response_distribution": {},
                "success_rate": 0.0
            }

        return {
            "total_questions": total,
            "response_distribution": {
                "azure_openai": {
                    "count": self.response_stats['azure_responses'],
                    "percentage": (self.response_stats['azure_responses'] / total) * 100
                },
                "neural_system": {
                    "count": self.response_stats['neural_responses'],
                    "percentage": (self.response_stats['neural_responses'] / total) * 100
                },
                "basic_fallback": {
                    "count": self.response_stats['fallback_responses'],
                    "percentage": (self.response_stats['fallback_responses'] / total) * 100
                }
            },
            "success_rate": ((self.response_stats['azure_responses'] +
                            self.response_stats['neural_responses'] +
                            self.response_stats['fallback_responses']) / total) * 100,
            "neural_integration_active": self.universal_system is not None
        }

    def reset_statistics(self):
        """Reset response statistics"""
        self.response_stats = {
            'azure_responses': 0,
            'neural_responses': 0,
            'fallback_responses': 0,
            'total_questions': 0
        }
        logger.info("📊 Wiki response statistics reset")

    def test_all_systems(self) -> Dict:
        """Test all response systems (Azure, Neural, Fallback)"""
        test_results = {
            "azure_openai": {"available": False, "error": None},
            "neural_system": {"available": False, "error": None},
            "basic_fallback": {"available": True, "error": None},
            "wiki_pages": len(self.pages),
            "initialized": self.initialized
        }

        # Test Azure OpenAI
        if self.api_key and self.api_key != '<<< API Key Here >>>':
            try:
                # Simple test request
                test_payload = {
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 10
                }
                headers = {"Content-Type": "application/json", "api-key": self.api_key}

                response = requests.post(self.endpoint_url, headers=headers,
                                       json=test_payload, timeout=10)
                if response.status_code == 200:
                    test_results["azure_openai"]["available"] = True
                else:
                    test_results["azure_openai"]["error"] = f"HTTP {response.status_code}"
            except Exception as e:
                test_results["azure_openai"]["error"] = str(e)[:100]

        # Test Neural System
        if self.universal_system:
            try:
                # Check if neural system is responsive
                system_status = self.universal_system.get_system_status()
                if system_status.get('system_health', {}).get('overall_score', 0) > 0:
                    test_results["neural_system"]["available"] = True
                else:
                    test_results["neural_system"]["error"] = "Low system health"
            except Exception as e:
                test_results["neural_system"]["error"] = str(e)[:100]

        return test_results
