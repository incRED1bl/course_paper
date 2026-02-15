import sys
import time

# Show loading immediately before heavy imports
print("⏳ Loading modules...", end="", flush=True)

from app import *


if __name__ == "__main__":
    print("\r✅ Modules loaded!" + " " * 50)
    print("\n" + "=" * 70)
    print("  Respiratory Sound Analysis - VS Code")
    print("=" * 70)
    
    print("\n🧠 Analysis Functions Available:")
    print("   • compute_entropy_complexity() - Calculate entropy & complexity")
    print("   • extract_all_features() - Feature extraction")
    print("   • build_feature_summary() - Build feature table")
    
    print("\n📊 Visualization Functions:")
    print("   • All visualization code is in colab_notebook.ipynb")
    print("   • 6 plotting functions built into the notebook")
    print("   • No separate visual/ module needed")
    
    print("\n" + "─" * 70)
    print("\n🚀 WORKFLOW:")
    print("\n  1️⃣  Open colab/colab_notebook.ipynb in Google Colab")
    print("  2️⃣  Run analysis with 3.69GB dataset")
    print("  3️⃣  Download results (features.pkl, model.pkl)")
    print("  4️⃣  Use results here in VS Code")
    
    print("\n📝 Example - Load Colab Results:")
    print("   import pickle")
    print("   with open('features.pkl', 'rb') as f:")
    print("       data = pickle.load(f)")
    
    print("\n📚 Documentation:")
    print("   • README.md - Project overview")
    print("   • COLAB_GUIDE.md - Step-by-step Colab guide")
    print("   • colab/colab_notebook.ipynb - All-in-one Colab notebook")
    
    print("\n" + "=" * 70)
