import math
from itertools import combinations


def binomial(n, k):
    # Calculate the number of combinations
    numerator = math.factorial(n)
    denominator = math.factorial(k) * math.factorial(n - k)
    num_combinations = numerator // denominator
    print(f"Number of combinations: {num_combinations}")

    # Generate and display all possible combinations
    elements = list(range(1, n + 1))  # Using numbers 1 to n for demonstration
    all_combinations = list(combinations(elements, k))
    print("All possible combinations:")
    for combo in all_combinations:
        print(combo)


binomial(3, 2)
