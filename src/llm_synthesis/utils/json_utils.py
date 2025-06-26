"""
Centralized json-repair library for LLM-generated JSON content.
This module provides functions to safely parse JSON strings, repair them if necessary,
"""
import json
import logging
from typing import Any, Optional, Dict, Union

try:
    from json_repair import repair_json, loads as repair_loads
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    logging.warning("json-repair not installed. Install with: pip install json-repair")

logger = logging.getLogger(__name__)


def safe_json_loads(
    json_str: str,
    fallback_value: Optional[Any] = None,
    log_errors: bool = True
) -> Union[Dict[str, Any], Any]:
    """
    Parse JSON string with automatic repair for LLM-generated content.
    
    Args:
        json_str: JSON string to parse
        fallback_value: Value to return if parsing fails
        log_errors: Whether to log parsing attempts
    """
    if not json_str or not json_str.strip():
        if log_errors:
            logger.warning("Empty JSON string provided")
        return fallback_value
    
    # Standard JSON parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if log_errors:
            logger.debug(f"Standard JSON parsing failed: {e}")
        
        # json-repair fallback
        if HAS_JSON_REPAIR:
            try:
                # Using repair_loads which combines repair + loads in one step
                repaired = repair_loads(json_str)
                if log_errors:
                    logger.info("JSON successfully repaired via json-repair")
                return repaired
            except Exception as repair_error:
                if log_errors:
                    logger.debug(f"json-repair failed: {repair_error}")
        
        # Final fallback
        if log_errors:
            logger.warning(f"All JSON parsing failed: {json_str[:100]}...")
        return fallback_value


def extract_json_from_text(
    text: str, 
    fallback_value: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain additional content.
    Useful for LLM outputs that mix JSON with explanatory text.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return fallback_value
    
    # Find matching closing brace
    brace_count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return safe_json_loads(text[start_idx:i+1], fallback_value)
    
    # If no complete match, try from start to end
    return safe_json_loads(text[start_idx:], fallback_value)


def validate_required_fields(json_obj: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that JSON object contains all required fields.
    
    Args:
        json_obj: Parsed JSON object
        required_fields: List of required field names
        
    Returns:
        True if all required fields present
    """
    if not isinstance(json_obj, dict):
        return False
    return all(field in json_obj for field in required_fields)


# Clean aliases for imports
parse_json = safe_json_loads
extract_json = extract_json_from_text