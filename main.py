from app import *



if __name__ == "__main__":
    signals, sample_rate = generate_sample_signals()
    
    m = 3
    tau = 1
    
    for disease_name, signal_data in signals.items():
        entropy, complexity = compute_entropy_complexity(signal_data, m=m, tau=tau)
        results(disease_name, entropy, complexity)
