import re
import numpy as np


class EvolutionaryLoop:
    def __init__(self, model, k, initial_solution, intype, outtype):
        """
        Initialize the evolutionary loop.
        This class tries to find agents that minimize a loss funciton.
        Params:
            model: an instance of Model
            k: the number of parameters to sample
            initial_solution: the initial solution
            intype: a string denoting the input type
            outtype: a string denoting the output type
        """
        self._model = model
        self._k = k
        self._pop = [initial_solution]
        self._fitnesses = []
        self._intype = intype
        self._outtype = outtype
        # Keep old populations to allow mu+lambda selection
        self._old_pop = []
        self._old_fit = []

    def ask(self):
        return self._pop

    def tell(self, fitnesses):
        # Order in ascending order
        fitnesses.extend(self._old_fit)
        self._pop.extend(self._old_pop)
        order = np.argsort(fitnesses)

        # Reorder population
        self._fitnesses = [fitnesses[i] for i in order]
        self._pop = [self._pop[i] for i in order]

        # Keep only the best ones in the old pop - reduce mem consumption
        self._old_pop = [x for x in self._pop[:2]]
        self._old_fit = [x for x in self._fitnesses[:2]]

        # Generate new pop - create the prompt
        prompt = ""
        for cnt, ind in enumerate(reversed(self._pop[:2])):
            prompt += f"```python\ndef solution_v{cnt}(x: {self._intype}) -> {self._outtype}:\n"
            for line in ind.split('\n')[1:]:
                prompt += line + '\n'
            prompt += "```\n"

        print("#"*80)
        print("#" + "prompt".center(78) + "#")
        print("#"*80)
        print("Prompt:", prompt)

        # Generate new pop - sample new individuals
        self._pop = [self._model(prompt) for i in range(self._k)]
        print("Uncleaned:", self._pop)
        self._pop = [self._clean_solution(x) for x in self._pop]
        print("Cleaned:", self._pop)


    def get_best(self):
        # Population is sorted - return the last individual
        return self._pop[-1]

    def _clean_solution(self, x):
        """
        Retrieves only the code from a reply and refactors the functions' name.
        """
        try:
            candidate_string = x.split('```')[1]
            candidate_string = candidate_string.replace('python', '', 1)
            candidate_string = candidate_string.replace('\\n', '\n')

            # Refactor the name in case we have both a single starting solution (first gen) and two starting solutions.
            candidate_string = re.sub('solution_v.', 'solution', candidate_string)

            while (candidate_string[0] == '\n'):
                candidate_string = candidate_string[1:]

            while (candidate_string[-2:] == '\n\n'):
                candidate_string = candidate_string[:-1]
        except:
            candidate_string = f"def solution(x: {self._intype}) -> {self._outtype}:\n\treturn 0"
        return candidate_string

