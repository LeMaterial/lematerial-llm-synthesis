#!/usr/bin/env python3
"""
Test script for material extractor functionality.
Tests the material extraction on sample markdown files.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import required modules
from llm_synthesis.transformers.material_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_test_files(test_md_dir: str) -> Dict[str, str]:
    """Load all markdown test files from the test directory."""
    test_files = {}
    test_dir = Path(test_md_dir)
    
    if not test_dir.exists():
        logger.error(f"Test directory {test_md_dir} does not exist!")
        return test_files
    
    for md_file in test_dir.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                test_files[md_file.name] = content
                logger.info(f"Loaded test file: {md_file.name} ({len(content)} characters)")
        except Exception as e:
            logger.error(f"Failed to load {md_file.name}: {e}")
    
    return test_files

def create_material_extractor():
    """Create and configure the material extractor."""
    try:
        # Create the signature
        signature = make_dspy_text_extractor_signature(
            signature_name="TextToMaterials",
            instructions="Extract ONLY the final synthesized materials from the publication text.",
            input_description="The publication text to extract the final synthesized materials from.",
            output_name="materials",
            output_description="The final synthesized materials as a comma-separated list.",
        )
        
        # Create the language model
        lm = get_llm_from_name(
            "gemini-2.0-flash", 
            {
                "temperature": 0.0,
                "max_tokens": 1000,
                "max_retries": 3
            }
        )
        
        # Create the extractor
        extractor = DspyTextExtractor(signature, lm)
        
        logger.info("✅ Material extractor created successfully")
        return extractor
        
    except Exception as e:
        logger.error(f"❌ Failed to create material extractor: {e}")
        return None

def test_material_extraction(extractor: DspyTextExtractor, test_files: Dict[str, str]) -> Dict[str, Any]:
    """Test material extraction on all test files."""
    results = {}
    
    for filename, content in test_files.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 Testing file: {filename}")
        logger.info(f"{'='*80}")
        
        try:
            # Clean the text
            cleaned_text = clean_text(content)
            logger.info(f"📄 Text length: {len(cleaned_text)} characters")
            
            # Extract materials
            logger.info("🔍 Extracting materials...")
            materials_text = extractor.forward(cleaned_text)
            
            # Parse materials
            if materials_text and materials_text.strip().lower() != "no materials synthesized":
                materials = [
                    material.strip()
                    for material in materials_text.replace("\n", ",").split(",")
                    if material.strip()
                ]
            else:
                materials = []
            
            # Store results
            result = {
                "filename": filename,
                "raw_output": materials_text,
                "materials": materials,
                "material_count": len(materials),
                "success": True,
                "error": None
            }
            
            # Print results
            logger.info(f"📋 Raw LLM output: {materials_text}")
            logger.info(f"🎯 Extracted materials ({len(materials)}): {materials}")
            
            if materials:
                logger.info("✅ Extraction successful")
            else:
                logger.warning("⚠️  No materials extracted")
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            result = {
                "filename": filename,
                "raw_output": None,
                "materials": [],
                "material_count": 0,
                "success": False,
                "error": str(e)
            }
        
        results[filename] = result
    
    return results

def print_summary(results: Dict[str, Any]):
    """Print a summary of all test results."""
    logger.info(f"\n{'='*80}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    total_files = len(results)
    successful_files = sum(1 for r in results.values() if r["success"])
    total_materials = sum(r["material_count"] for r in results.values())
    
    logger.info(f"📁 Total files tested: {total_files}")
    logger.info(f"✅ Successful extractions: {successful_files}")
    logger.info(f"❌ Failed extractions: {total_files - successful_files}")
    logger.info(f"🎯 Total materials extracted: {total_materials}")
    logger.info(f"📈 Average materials per file: {total_materials/max(total_files, 1):.1f}")
    
    # Detailed results
    logger.info(f"\n{'='*80}")
    logger.info("📋 DETAILED RESULTS")
    logger.info(f"{'='*80}")
    
    for filename, result in results.items():
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {filename}: {result['material_count']} materials")
        if result["materials"]:
            logger.info(f"   Materials: {', '.join(result['materials'])}")
        if result["error"]:
            logger.info(f"   Error: {result['error']}")

def main():
    """Main test function."""
    logger.info("🚀 Starting Material Extractor Test")
    logger.info("="*80)
    
    # Set up paths
    test_md_dir = project_root / "test" / "test_md"
    
    # Load test files
    logger.info(f"📂 Loading test files from: {test_md_dir}")
    test_files = load_test_files(str(test_md_dir))
    
    if not test_files:
        logger.error("❌ No test files found!")
        return 1
    
    logger.info(f"📄 Loaded {len(test_files)} test files")
    
    # Create extractor
    logger.info("🔧 Creating material extractor...")
    extractor = create_material_extractor()
    
    if not extractor:
        logger.error("❌ Failed to create extractor!")
        return 1
    
    # Run tests
    logger.info("🧪 Running material extraction tests...")
    results = test_material_extraction(extractor, test_files)
    
    # Print summary
    print_summary(results)
    
    logger.info("\n🎉 Test completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
