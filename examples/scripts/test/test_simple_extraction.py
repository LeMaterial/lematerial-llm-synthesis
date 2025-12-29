#!/usr/bin/env python3
"""
Simple test to debug material extraction issues.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from llm_synthesis.utils.dspy_utils import get_llm_from_name

def test_direct_llm():
    """Test LLM directly with a simple prompt."""
    
    # Create the language model
    lm = get_llm_from_name(
        "gemini-2.0-flash",
        {
            "temperature": 0.0,
            "max_tokens": 1000,
            "max_retries": 3
        }
    )
    
    # Test prompt
    test_text = """
    Structure cristalline Trimétaphosphate de Strontium Heptahydraté: Sr3(PaO9)2.7HzO
    
    Lithium Hydrogen Phosphite, LiH2PO3 crystallizes in the orthorhombic space group.
    """
    
    prompt = f"""
    You are a helpful assistant that extracts ONLY the final synthesized materials from scientific papers.

    Your task is to identify ONLY the materials that are the final products of synthesis procedures described in the paper.

    IMPORTANT GUIDELINES:
    - ONLY include materials that are the final synthesized products
    - DO NOT include starting materials, precursors, supports, gases, solvents, or other chemicals used in synthesis
    - DO NOT include materials that are just mentioned or characterized but not synthesized
    - Focus on the main target materials that are actually synthesized

    MATERIAL NAMING PRIORITY - CRITICAL RULES:
    - ALWAYS use ONLY chemical formulas, NEVER use common names or descriptive names
    - If a material has both a chemical formula and a common name, use ONLY the chemical formula
    - Use standard chemical notation with proper subscripts (e.g., "Cr2Te4O11", "Sr3(P3O9)2·7H2O")
    - For complex organic compounds, use the molecular formula (e.g., "C12H11O5N2Br")
    - Include crystal phase information when specified (e.g., "NaH2PO4·H2O")
    - DO NOT output both name and formula - use ONLY the formula

    CRITICAL OUTPUT REQUIREMENT:
    - You MUST return ONLY chemical formulas, never common names
    - If you see "Sr3(PaO9)2.7HzO" in the text, output "Sr3(P3O9)2·7H2O" 
    - If you see "LiH2PO3" in the text, output "LiH2PO3"
    - If you see "Lithium Hydrogen Phosphite" in the text, output "LiH2PO3"
    - ALWAYS convert common names to chemical formulas

    Return a simple comma-separated list of ONLY the final synthesized materials using chemical formulas.

    If no materials are synthesized in the paper, return "No materials synthesized".

    Keep the output simple and clean - just the final synthesized material chemical formulas separated by commas.

    Publication text: {test_text}
    """
    
    print("Testing direct LLM call...")
    print("Prompt:", prompt[:200] + "...")
    print("\n" + "="*80)
    
    try:
        response = lm(prompt)
        print("Raw LLM Response:", response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_direct_llm()
