from array import generate_sine_array, generate_gaussian_array, generate_lorenz_array
from func import compute_entropy_complexity


def results(name, entropy, complexity):
    print(f"\n{name}:")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Complexity: {complexity:.4f}")


def main():

    sine_array = generate_sine_array()
    gaussian_array = generate_gaussian_array()
    lorenz_array = generate_lorenz_array()
    
    entropy_sine, complexity_sine = compute_entropy_complexity(sine_array)
    results("Sine wave", entropy_sine, complexity_sine)
    
    entropy_gaussian, complexity_gaussian = compute_entropy_complexity(gaussian_array)
    results("Gaussian noise", entropy_gaussian, complexity_gaussian)
    
    entropy_lorenz, complexity_lorenz = compute_entropy_complexity(lorenz_array)
    results("Lorenz series", entropy_lorenz, complexity_lorenz)
    

if __name__ == "__main__":
    main()
