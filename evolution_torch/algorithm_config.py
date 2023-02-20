"""Stores configuration parameters for the CPPN."""
import random
import torch
import imageio as iio
from cppn_torch.activation_functions import *
import cppn_torch.activation_functions as af
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
        
        self.autoencoder_frequency = 0 # legacy, should be move to subclasses
        
        self.run_id = 0
        self.output_dir = "output/default"
        self.experiment_condition = "_default"
        
        
        self._make_dirty()
        
def apply_condition(config, controls, condition, name, name_to_function_map):
    config.name = name
    config.experiment_condition = name
    
    if len(controls) > 0:
        for k,v in controls.items():
            print("\t Control:", k, "->", v)
            config.apply_condition(k, v)
            if k == "num_runs":
                config.num_runs = v
            if k == "target":
                config.target = v
                config.target_name = config.target
                pilmode = "RGB" if len(config.color_mode) == 3 else "L"
                config.target = torch.tensor(iio.imread(config.target), dtype=torch.float32, device=config.device, pilmode=pilmode, as_gray=len(config.color_mode)==1)
                config.res_h, config.res_w = config.target.shape[:2]
        
        config.fitness_function = name_to_function_map.get( config.fitness_function)
        if config.fitness_function is None:
            raise Exception(f"Fitness function {config.fitness_function} not found")
        
    if len(condition) > 0:
        for k, v in condition.items():
            if k is not None:
                print(f"\t\tapply {k}->{v}")
                config.apply_condition(k, v)
            if k == "target":
                config.target = v
                if isinstance(config.target, str):
                    config.target_name = config.target
                    pilmode = "RGB" if len(config.color_mode) == 3 else "L"
                    config.target = torch.tensor(iio.imread(config.target), dtype=torch.float32, device=config.device, pilmode=pilmode, as_gray=len(config.color_mode)==1)

                config.res_h, config.res_w = config.target.shape[:2]
   
    if config.fitness_schedule is not None:
        for i,fn in enumerate(config.fitness_schedule):
            if isinstance(fn, str):
                config.fitness_schedule[i] = name_to_function_map.get(fn)
                if config.fitness_schedule[i] is None:
                    raise Exception(f"Fitness function {fn} not found")

    if config.color_mode=="L":
        config.target = config.target.mean(dim=2)

    if config.target.max() > 1.0:
        config.target = config.target.to(torch.float32) /255.0
    
    if len(config.color_mode) != config.num_outputs:
        print("WARNING: color_mode does not match num_outputs")
        print("setting num_outputs to len(color_mode)")
        config.num_outputs = len(config.color_mode)
    config.device = torch.device(config.device)
    config.target = config.target.to(config.device)     

    for i in range(len(config.activations)):
        if isinstance(config.activations[i], str):
            config.activations[i] = af.__dict__.get(config.activations[i])  
