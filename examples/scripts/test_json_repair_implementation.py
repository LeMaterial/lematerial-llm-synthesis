"""
Test the centralized JSON repair implementation.
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llm_synthesis.utils.json_utils import parse_json, extract_json, validate_required_fields

logging.basicConfig(level=logging.INFO)

def test_json_repair():
    """Test various JSON repair scenarios."""
    
    test_cases = [
        # Valid JSON
        ('Valid JSON', '{"target_compound": "TiO2", "materials": ["Ti", "O2"], "steps": ["heat"]}'),
        
        # Truncated JSON
        ('Truncated JSON', '{"target_compound": "TiO2", "materials": ["Ti", "O2"], "steps": ["heat"'),
        
        # JSON with trailing comma
        ('Trailing comma', '{"target_compound": "TiO2", "materials": ["Ti", "O2"], "steps": ["heat"],}'),
        
        # JSON embedded in text
        ('Embedded JSON', 'The synthesis is: {"target_compound": "TiO2", "materials": ["Ti"], "steps": ["heat"]} as shown.'),
        
        # Invalid JSON
        ('Invalid JSON', 'This is not JSON at all'),
        
        # Complex nested JSON with truncation
        ('Complex truncated', '{"target_compound": "ZnO", "materials": [{"name": "Zn", "amount": "10g"}], "steps": [{"step_number": 1, "description": "heat"'),
    ]
    
    print("=" * 60)
    print("TESTING CENTRALIZED JSON REPAIR IMPLEMENTATION")
    print("=" * 60)
    
    for test_name, test_input in test_cases:
        print(f"\n{test_name}:")
        print(f"Input: {test_input[:50]}{'...' if len(test_input) > 50 else ''}")
        
        # Test parse_json
        result = parse_json(test_input, fallback_value={"error": "parsing_failed"})
        print(f"parse_json result: {result}")
        
        # Test validation if it's a dict
        if isinstance(result, dict) and result != {"error": "parsing_failed"}:
            is_valid = validate_required_fields(result, ['target_compound', 'materials', 'steps'])
            print(f"Valid synthesis: {is_valid}")
        
        # Test extract_json for embedded cases
        if "embedded" in test_name.lower():
            extracted = extract_json(test_input, fallback_value=None)
            print(f"extract_json result: {extracted}")
        
        print("-" * 40)

def test_synthesis_validation():
    """Test synthesis JSON validation.""" 
    
    print("\n" + "=" * 60)
    print("SYNTHESIS JSON VALIDATION TEST")
    print("=" * 60)
    
    validation_cases = [
        # Valid synthesis JSON
        {
            "target_compound": "TiO2",
            "materials": ["Ti", "O2"],
            "steps": ["heat", "cool"]
        },
        
        # Missing required field
        {
            "target_compound": "TiO2",
            "materials": ["Ti", "O2"]
            # Missing 'steps'
        },
        
        # Empty but valid structure
        {
            "target_compound": "",
            "materials": [],
            "steps": []
        },
        
        # Invalid type
        "not a dictionary",
        
        # None value
        None,
    ]
    
    for i, case in enumerate(validation_cases, 1):
        print(f"\nValidation case {i}:")
        print(f"Input: {case}")
        is_valid = validate_required_fields(case, ['target_compound', 'materials', 'steps'])
        print(f"Valid synthesis JSON: {is_valid}")

if __name__ == "__main__":
    test_json_repair()
    test_synthesis_validation()