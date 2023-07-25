import random
import numpy as np
from deap import base, creator, tools, algorithms
input_shape = (128, 128, 3)  # Change this to your image size
num_classes = 10             # Change this to the number of classes in your dataset
def evaluate_cnn(individual):
    # individual: List representing the CNN architecture (e.g., number of filters in each layer)
    # Implement code to create and train the CNN model using the individual's architecture
    # Evaluate the model's performance on the validation dataset (e.g., accuracy)
    # Return a tuple of the fitness value (accuracy) and any additional information you want to track
    return accuracy, additional_info
# Define the genetic algorithm parameters
POPULATION_SIZE = 50
CXPB = 0.6  # Crossover probability
MUTPB = 0.2  # Mutation probability
N_GEN = 10  # Number of generations

# Define the bounds for the search space (e.g., the range of values for the number of filters in each layer)
# Example bounds for three convolutional layers: [(16, 128), (16, 128), (16, 128)]
BOUNDS = []

# Create the DEAP Toolbox
toolbox = base.Toolbox()

# Define the individual and fitness types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Attribute generator
toolbox.register("filter_count", random.randint, 16, 128)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.filter_count,), n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
def main():
    pop = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the entire population
    fitnesses = list(map(evaluate_cnn, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(N_GEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(offspring)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(evaluate_cnn, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the offspring
        pop[:] = offspring

    best_individual = tools.selBest(pop, 1)[0]
    best_architecture = np.array(best_individual)
    print("Best CNN Architecture:", best_architecture)
    print("Best Accuracy:", best_individual.fitness.values[0])


if __name__ == "__main__":
    main()
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(individual):
    model = models.Sequential()

    # Add the input layer with the given input shape
    model.add(layers.Input(shape=input_shape))

    # Create the convolutional layers based on the individual's architecture
    for filters in individual:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output of the previous layer
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def evaluate_cnn(individual):
    model = create_cnn_model(individual)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Implement code to load your training and validation datasets and preprocess them
    # X_train, y_train, X_val, y_val = load_and_preprocess_data()

    # Train the model on the training dataset
    # history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

    # Evaluate the model on the validation dataset
    # accuracy = history.history['val_accuracy'][-1]

    # For the sake of demonstration, I'm using a random accuracy as a placeholder
    accuracy = random.random()

    return accuracy,

# The rest of the code remains the same...
