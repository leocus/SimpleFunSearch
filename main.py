# Use MPI for the communication
# Use timeout in communications and be careful to avoid deadlocks
import yaml
import numpy as np
from model_adapter import Model
from evaluators import Evaluator
from island import EvolutionaryLoop


# Read the config
config = yaml.load(open('config.yaml'), Loader=yaml.Loader)

# Init the model
model = Model(
    config['model']['url'],
    config['model']['name'],
    config['model']['key'],
    config['system_prompt'],
)

# Init the EA
ea = EvolutionaryLoop(
    model,
    config['ea']['k'],
    config['initial_solution'],
    config['intype'],
    config['outtype']
)

# Init the evaluator
evaluator = Evaluator.get_class(config['evaluator']['name'])(
    **config['evaluator']['kwargs']
)

best = None
best_fit = float("inf")

# Evolve the functions
for generation in range(config['ea']['generations']):
    solutions = ea.ask()
    fitnesses = []
    for idx, solution in enumerate(solutions):
        fitnesses.append(evaluator(solution))
    print(fitnesses)
    ea.tell(fitnesses)

    argmin = np.argmin(fitnesses) 
    if fitnesses[argmin] < best_fit:
        best_fit = fitnesses[argmin]
        best = solutions[argmin]

    print(f"Generation {generation}")
    for f in (np.min, np.mean, np.std, np.max):
        print(f"\t{f.__name__}: {f(fitnesses)}")
    print(f"Best so far: {best_fit}")
