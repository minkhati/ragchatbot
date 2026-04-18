import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Outline / structure queries** (e.g. "what lessons does X have?", "show me the outline of X", "list the lessons in X"): use `get_course_outline` and return the course title, course link, and every lesson number + lesson title
- **Content / concept queries**: use `search_course_content` to find relevant material
- **Sequential tool calls**: You may make up to 2 tool calls in sequence. After seeing round-1 results, call another tool only if needed to complete the answer. After 2 tool calls you must synthesize a final answer.
- Use sequential calls for: cross-course comparisons, outline-then-content lookups, multi-part questions
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without any tool call
- **Outline queries**: Call `get_course_outline`, then present the full outline (course title, course link, each lesson number and title)
- **Course-specific content questions**: Call `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool output"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, messages, system_content, tools, tool_manager)

        return response.content[0].text

    def _run_tool_loop(self, first_response, messages: List, system_content: str,
                       tools: List, tool_manager) -> str:
        current_response = first_response

        for round_num in range(1, self.MAX_TOOL_ROUNDS + 1):
            messages.append({"role": "assistant", "content": current_response.content})

            tool_results = self._execute_tool_calls(current_response, tool_manager)
            if not tool_results:
                return current_response.content[0].text

            messages.append({"role": "user", "content": tool_results})

            is_last_round = (round_num >= self.MAX_TOOL_ROUNDS)
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if not is_last_round:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            next_response = self.client.messages.create(**next_params)

            if next_response.stop_reason != "tool_use" or is_last_round:
                return next_response.content[0].text

            current_response = next_response

        return next_response.content[0].text

    def _execute_tool_calls(self, response, tool_manager) -> List[Dict[str, Any]]:
        results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    content = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    content = f"Tool error: {e}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
        return results