import sys
import time

# Show loading immediately before heavy imports
print("⏳ Loading modules...", end="", flush=True)

from app import *


if __name__ == "__main__":
    stages = [
        "Loading modules...",
        "Initializing analysis...",
        "Generating signals...",
        "Processing Healthy samples...",
        "Processing Diseased samples...",
        "Processing COPD samples...",
        "Finalizing results..."
    ]
    
    total = len(stages)
    
    for i, stage in enumerate(stages):
        if i > 0:  # First stage already printed
            print(f"\r⏳ {stage}", end="", flush=True)
            
            if i == 1:
                # Already imported, just pause
                time.sleep(0.2)
            elif i == 2:
                signals, sample_rate = generate_sample_signals()
                time.sleep(0.3)
            elif i == 3:
                m, tau = 3, 1
                signal_data = signals['Healthy']
                entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
                time.sleep(0.3)
            elif i == 4:
                signal_data = signals['Diseased']
                entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
                time.sleep(0.3)
            elif i == 5:
                signal_data = signals['COPD']
                entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
                time.sleep(0.3)
            else:
                time.sleep(0.2)
    
    print("\r✅ Analysis complete!" + " " * 50)
    print("📊 Open visual.ipynb for detailed results")
