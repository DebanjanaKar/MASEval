"""
Quick test script to verify dataset loading works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_mast_dataset

def test_load():
    """Test loading the dataset."""
    print("="*60)
    print("Testing MAST Dataset Loading")
    print("="*60)
    
    try:
        # Try loading the dataset
        df = load_mast_dataset()
        
        print(f"\n✅ Success! Loaded {len(df)} traces")
        print(f"\nDataFrame columns: {list(df.columns)}")
        print(f"\nFirst trace preview:")
        if len(df) > 0:
            first_trace = df.iloc[0]["trace"]
            print(f"  - Number of turns: {len(first_trace)}")
            if len(first_trace) > 0:
                print(f"  - First turn: {first_trace[0]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load()
    sys.exit(0 if success else 1)

# Made with Bob
