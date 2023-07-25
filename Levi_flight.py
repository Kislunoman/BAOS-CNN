import random
import math

def objective_function(x):
    # Define your objective function here
    # For demonstration purposes, we'll use a simple quadratic function
    return -((x - 2) ** 2) + 10

def levi_flight_search(iterations, step_size, alpha):
    current_position = random.uniform(-10, 10)  # Random initial position

    for _ in range(iterations):
        # Generate a random step size from a Levy distribution
        step = step_size / (math.sqrt(random.random()) ** (1 / alpha))

        # Generate a random direction for the step
        direction = random.choice([-1, 1])

        # Update the current position
        current_position += direction * step

        # Bound the position to the search space [-10, 10]
        current_position = max(min(current_position, 10), -10)

        # Calculate the objective function value at the current position
        current_value = objective_function(current_position)

        print(f"Iteration {_+1}: x = {current_position:.4f}, f(x) = {current_value:.4f}")

    return current_position, current_value

if __name__ == "__main__":
    iterations = 100  # Number of iterations
    step_size = 0.1   # Step size for the random walk
    alpha = 1.5       # Alpha parameter for the Levy distribution

    final_position, final_value = levi_flight_search(iterations, step_size, alpha)
    print("\nOptimization Result:")
    print(f"Final x = {final_position:.4f}, f(x) = {final_value:.4f}")
