"""Stores configuration parameters for the CPPN."""
import random
import torch

from cppn_torch.activation_functions import *
from cppn_torch import CPPNConfig as Config

class AlgorithmConfig(Config):
    """Stores configuration parameters for the CPPN."""
    # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        # Initialize to CPPN default values:
        super().__init__()
        # overrides:
        self.seed = random.randint(0, 100000)
        self.activations=  [sin, sigmoid, gauss, identity, round_activation, abs_activation, pulse] # innovation engines
        
        # algorithm specific:
        self.target = "data/sunrise_tiny.png"
        self.population_size = 100
        self.num_generations = 1000
        self.population_elitism = 1
        
        self.do_crossover = True
        self.crossover_ratio = .75 # from original NEAT
        self.use_dynamic_mutation_rates = False
        self.dynamic_mutation_rate_end_modifier = 1.0
        self.init_connection_probability = 0.85
        
        self.novelty_adjusted_fitness_proportion = 0 # not currently using novelty
        
        self.fitness_function = 'mse' #     default: -mse (not used by MOVE)
        self.fitness_schedule_type = "alternating"
        self.fitness_schedule_period = 10
        self.fitness_schedule = None
        self.min_fitness = None
        self.max_fitness = None
        
        """DGNA: the probability of adding a node is 0.5 and the
        probability of adding a connection is 0.4.
        SGNA: probability of adding a node is 0.05 and the
         probability of adding a connection is 0.04.
        NEAT: probability of adding a node is 0.03 and the
          probability of adding a connection is 0.05."""
        self.prob_mutate_activation = .15
        self.prob_mutate_weight = .80 # .80 in the original NEAT
        self.prob_add_connection = .15 # 0.05 in the original NEAT
        self.prob_add_node = .15 # 0.03 in original NEAT
        self.prob_remove_node = 0.05
        self.prob_disable_connection = .05

        self.prob_random_restart = 0
        
        
        # EA
        self.run_id = 0
        self.output_dir = "output/default"
        self.experiment_condition = "_default"
        
        
        self._make_dirty()
        
