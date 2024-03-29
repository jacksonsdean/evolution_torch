import math
import random
import time
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import __main__ as main
if not hasattr(main, '__file__'):
    try:
        from tqdm.notebook import trange
    except ImportError:
        from tqdm import trange
else:
    from tqdm import trange
import pandas as pd
import torch
import itertools
import os
from cppn_torch.graph_util import activate_population
from cppn_torch.fitness_functions import correct_dims
from cppn_torch.util import get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, visualize_network, get_max_number_of_hidden_nodes
from cppn_torch.image_cppn import ImageCPPN
import copy 
import logging

class CPPNEvolutionaryAlgorithm(object):
    def __init__(self, config, debug_output=False) -> None:
        self.config = config
        
        if self.config.with_grad:
            torch.autograd.set_grad_enabled(True)
        else:
            torch.autograd.set_grad_enabled(False)
        if self.config.autoencoder_frequency > 0 and self.gen % self.config.autoencoder_frequency == 0:
            from evolution_torch.autoencoder import initialize_encoders, AutoEncoder
        
        if not hasattr(self, "inputs"):
            self.inputs = None # default to coord inputs in CPPN class
        
        self.gen = 0
        self.debug_output = debug_output
        self.show_output = False
        
        self.results = pd.DataFrame(columns=['condition', 'target', 'run', 'gen', 'fitness', 'mean_fitness', 'diversity', 'population', 'avg_num_connections', 'avg_num_hidden_nodes', 'max_num_connections', 'max_num_hidden_nodes', 'time', 'total_offspring'])
                
        self.solutions_over_time = []
        self.time_elapsed = 0
        self.solution_generation = -1
        self.population = []
        self.solution = None
        self.this_gen_best = None
        self.novelty_archive = []
        self.device = config.device
        self.run_number = 0
        self.diversity = 0
        self.total_offspring = 0
        
        self.solution_fitness = -math.inf
        self.best_genome = None
        if self.config.genome_type is None:
            self.config.genome_type = ImageCPPN
        self.genome_type = config.genome_type

        self.fitness_function = config.fitness_function
        
        self.fitness_function_measures_genome = False
        
        if not hasattr(self,  "name_function_map"):
            import cppn_torch.fitness_functions as ff
            self.name_function_map = ff.__dict__
        
        if not isinstance(self.config.fitness_function, Callable):
            self.fitness_function = self.name_function_map.get(self.config.fitness_function)
            self.fitness_function_normed = self.fitness_function
            
        self.target = self.config.target.to(self.device)
        
        if len(self.target.shape) < 3:
            # grayscale image
            if self.config.color_mode != "L":
                logging.warning("Target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
                self.config.color_mode = "L"
                
        if self.config.res_w != self.target.shape[0]:
            self.config.res_w = self.target.shape[0]
            logging.warning("Target image width does not match config.res_w. Setting config.res_w to target image width")
        if self.config.res_h != self.target.shape[1]:
            self.config.res_h = self.target.shape[1]
            logging.warning("Target image height does not match config.res_h. Setting config.res_h to target image height")

        self.fitesses = {}
    
    def get_mutation_rates(self):
        """Get the mutate rates for the current generation 
        if using a mutation rate schedule, else use config values
            abandon all hope all ye who enter here

        Returns:
            float: prob_mutate_activation,
            float: prob_mutate_weight,
            float: prob_add_connection,
            float: prob_add_node,
            float: prob_remove_node,
            float: prob_disable_connection,
            float: weight_mutation_max, 
            float: prob_reenable_connection
        """
        
        if(self.config.use_dynamic_mutation_rates):
            # Behold:
            run_progress = self.gen / self.config.num_generations
            end_mod = self.config.dynamic_mutation_rate_end_modifier
            prob_mutate_activation   = self.config.prob_mutate_activation   - (self.config.prob_mutate_activation    - end_mod * self.config.prob_mutate_activation)   * run_progress
            prob_mutate_weight       = self.config.prob_mutate_weight       - (self.config.prob_mutate_weight        - end_mod * self.config.prob_mutate_weight)       * run_progress
            prob_add_connection      = self.config.prob_add_connection      - (self.config.prob_add_connection       - end_mod * self.config.prob_add_connection)      * run_progress
            prob_add_node            = self.config.prob_add_node            - (self.config.prob_add_node             - end_mod * self.config.prob_add_node)            * run_progress
            prob_remove_node         = self.config.prob_remove_node         - (self.config.prob_remove_node          - end_mod * self.config.prob_remove_node)         * run_progress
            prob_disable_connection  = self.config.prob_disable_connection  - (self.config.prob_disable_connection   - end_mod * self.config.prob_disable_connection)  * run_progress
            weight_mutation_max      = self.config.weight_mutation_max      - (self.config.weight_mutation_max       - end_mod * self.config.weight_mutation_max)      * run_progress
            prob_reenable_connection = self.config.prob_reenable_connection - (self.config.prob_reenable_connection  - end_mod * self.config.prob_reenable_connection) * run_progress
            return  prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection
        else:
            # just return the config values directly
            return  self.config.prob_mutate_activation, self.config.prob_mutate_weight, self.config.prob_add_connection, self.config.prob_add_node, self.config.prob_remove_node, self.config.prob_disable_connection, self.config.weight_mutation_max, self.config.prob_reenable_connection

    def evolve(self, run_number = 1, show_output=False, initial_population=True):
        self.start_time = time.time()
        self.run_number = run_number
        self.show_output = show_output or self.debug_output
        if initial_population:
            for i in range(self.config.population_size): 
                self.population.append(self.genome_type(self.config)) # generate new random individuals as parents
            
            # update novelty encoder 
            if self.config.novelty_mode == "encoder":  
                initialize_encoders(self.config, self.target)  
            if self.config.activation_mode == "population":
                activate_population(self.population, self.config, self.inputs)
            else:
                for g in self.population: g(inputs=self.inputs)
            self.update_fitnesses_and_novelty()
            self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
            self.solution = self.population[0].clone(cpu=True) 

        try:
            # Run algorithm
            pbar = trange(self.config.num_generations, desc=f"Run {self.run_number}")
        
            for self.gen in pbar:
                self.generation_start()
                self.run_one_generation()
                self.generation_end()
                b = self.get_best()
                if b is not None:
                    pbar.set_postfix_str(f"bf: {self.fitnesses[b.id]:.4f} (id:{b.id}) d:{self.diversity:.4f} af:{np.mean(list(self.fitnesses.values())):.4f} u:{self.n_unique}")
                else:
                    pbar.set_postfix_str(f"d:{self.diversity:.4f}")
            
        except KeyboardInterrupt:
            self.on_end()
            raise KeyboardInterrupt()  
        
        self.on_end()

    def on_end(self):
        self.end_time = time.time()     
        self.time_elapsed = self.end_time - self.start_time  
        print("\n\nEvolution completed with", self.gen+1, "generations", "in", self.time_elapsed, "seconds")
        print("Wrapping up, please wait...")

        # save results
        print("Saving data...")
        cond_dir = os.path.join(self.config.output_dir, "conditions", self.config.experiment_condition)
        os.makedirs(cond_dir, exist_ok=True)
        # self.config.run_id =len(os.listdir(cond_dir))
        self.run_number = self.config.run_id
        
        self.results.loc[self.run_number, "run_id"] = self.config.run_id
        
        run_dir = os.path.join(cond_dir, f"run_{self.config.run_id:04d}")
        if os.path.exists(run_dir):
            logging.warning(f"run dir already exists, overwriting: {run_dir}")
        else:
            os.makedirs(run_dir)
     
        with open(os.path.join(run_dir, f"target.txt"), 'w') as f:
            f.write(self.config.target_name)
        
        # save to run dir
        filename = os.path.join(run_dir, f"results.pkl")
        self.results.to_pickle(filename)
        
        # save to output dir
        filename = os.path.join(self.config.output_dir, f"results.pkl")
        if os.path.exists(filename):
            tries = 0
            while tries < 5:
                try:
                    with open(filename, 'rb') as f:
                        save_results = pd.read_pickle(f)
                        save_results = pd.concat([save_results, self.results]).reset_index(drop=True)
                        break
                except:
                    tries += 1
                    time.sleep(1)
            if tries == 5:
                logging.warning("Failed to read output_dir results file, overwriting")
                save_results = self.results
        else:
            save_results = self.results
        save_results.to_pickle(filename)
        
            
    def update_fitness_function(self):
        """Update normalize fitness function if using a schedule or normalized fitness function"""
        if self.config.fitness_schedule is not None:
            if self.config.fitness_schedule_type == 'alternating':
                if self.gen==0:
                    self.fitness_function = self.config.fitness_schedule[0]
                elif self.gen % self.config.fitness_schedule_period == 0:
                        self.fitness_function = self.config.fitness_schedule[self.gen // self.config.fitness_schedule_period % len(self.config.fitness_schedule)]
                if self.debug_output:
                    print('Fitness function:', self.fitness_function.__name__)
            else:
                raise Exception("Unrecognized fitness schedule")
            
        if self.config.min_fitness is not None and self.config.max_fitness is not None:
            self.fitness_function_normed = lambda x,y: (self.config.fitness_function(x,y) - self.config.min_fitness) / (self.config.max_fitness - self.config.min_fitness)
        else:
            self.fitness_function_normed = self.fitness_function # no normalization

    def generation_start(self):
        """Called at the start of each generation"""
        self.update_fitness_function()

        if self.show_output:
            self.print_fitnesses()
            
        # update the autoencoder used for novelty
        if self.config.autoencoder_frequency > 0 and self.gen % self.config.autoencoder_frequency == 0:
            AutoEncoder.instance.update_novelty_network(self.population) 
            
    def run_one_generation(self):
        """Run one generation of the algorithm"""
        raise NotImplementedError("run_one_generation() not implemented for base class")

    def generation_end(self):
        """Called at the end of each generation"""
        self.record_keeping()

    def update_fitnesses_and_novelty(self):
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))

        # fits = self.fitness_function(torch.stack([g.get_image() for g in self.population]), self.target).detach() # TODO maybe don't detach and experiment with autograd?
        if self.config.activation_mode == "population":
            imgs = activate_population(self.population, self.config, self.inputs)
        else:
            imgs = torch.stack([g(self.inputs) for g in self.population])
        imgs, target = correct_dims(imgs, self.target)
        
        if self.fitness_function_measures_genome:
            fits = self.fitness_function(self.population, target)
        else:
            fits = self.fitness_function(imgs, target)
        
        for i in pbar:
            if self.show_output:
                pbar.set_description_str("Evaluating gen " + str(self.gen) + ": ")
            
            self.population[i].fitness = fits[i]
        
        if self.show_output:
            pbar = trange(len(self.population))
        else:
            pbar = range(len(self.population))
            
        novelties = AutoEncoder.instance.get_ae_novelties(self.population).detach()
        for i, n in enumerate(novelties):
            self.population[i].novelty = n
            self.novelty_archive = self.update_solution_archive(self.novelty_archive, self.population[i], self.config.novelty_archive_len, self.config.novelty_k)
    
    def update_solution_archive(self, solution_archive, genome, max_archive_length, novelty_k):
        # genome should already have novelty score
        solution_archive = sorted(solution_archive, reverse=True, key = lambda s: s.novelty)

        if(len(solution_archive) >= max_archive_length):
            if(genome.novelty > solution_archive[-1].novelty):
                # has higher novelty than at least one genome in archive
                solution_archive[-1] = genome # replace least novel genome in archive
        else:
            solution_archive.append(genome)
        return solution_archive
    
    def record_keeping(self, skip_fitness=False):
        if len(self.fitnesses) > 0:
            if len(self.population) > 0:
                self.population = sorted(self.population, key=lambda x: self.fitnesses[x.id], reverse=True) # sort by fitness
                # if self.config.with_grad:
                    # self.population[0].discard_grads()
                self.this_gen_best = self.population[0].clone(self.config, cpu=True)  # still sorted by fitness
        
        div_mode = self.config.get('diversity_mode', None)
        if div_mode == 'full':
            std_distance, avg_distance, max_diff = calculate_diversity_full(self.population)
        elif div_mode == 'stochastic':
            std_distance, avg_distance, max_diff = calculate_diversity_stochastic(self.population)
        else:
            std_distance, avg_distance, max_diff = torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
        self.diversity = avg_distance
        n_nodes = get_avg_number_of_hidden_nodes(self.population)
        n_connections = get_avg_number_of_connections(self.population)
        max_connections = get_max_number_of_connections(self.population)
        max_nodes = get_max_number_of_hidden_nodes(self.population)

        self.n_unique = len(set([g.id for g in self.population]))

        if not skip_fitness:
            # fitness
            if self.fitnesses[self.population[0].id] > self.solution_fitness: # if the new parent is the best found so far
                self.solution = self.population[0]                 # update best solution records
                self.solution_fitness = self.fitnesses[self.population[0].id]
                self.solution_generation = self.gen
                self.best_genome = self.solution
            
            os.makedirs(os.path.join(self.config.output_dir, 'images'), exist_ok=True)
            self.save_best_img(os.path.join(self.config.output_dir, "images", f"current_best_output.png"))
        
        if self.solution is not None:
            self.results.loc[len(self.results.index)] = [self.config.experiment_condition, self.config.target_name, self.config.run_id, self.gen, self.solution_fitness, np.mean(list(self.fitnesses.values())),avg_distance.item(), float(len(self.population)), n_connections, n_nodes, max_connections, max_nodes, time.time() - self.start_time, self.total_offspring]
            plt.close()
            plt.plot(self.results['gen'], self.results['fitness'], label='best')
            plt.plot(self.results['gen'], self.results['mean_fitness'], label='mean')
            plt.legend()
            plt.savefig(os.path.join(self.config.output_dir, "current_fitness.png"))
            plt.close()
        else:
            self.results.loc[len(self.results.index)] = [self.config.experiment_condition, self.config.target_name, self.config.run_id, self.gen, 0.0,  np.mean(list(self.fitnesses.values())), avg_distance.item(), float(len(self.population)), n_connections, n_nodes, max_connections, max_nodes, time.time() - self.start_time, self.total_offspring]

    def mutate(self, child):
        rates = self.get_mutation_rates()
        child.fitness, child.adjusted_fitness = 0, 0 # new fitnesses after mutation
        child.mutate(rates)
    
    def get_best(self):
        if len(self.population) == 0:
            print("No individuals in population")
            return None
        max_fitness_individual = max(self.population, key=lambda x: self.fitnesses[x.id])
        return max_fitness_individual
    
    def print_best(self):
        best = self.get_best()
        print("Best:", best.id, best.fitness)
        
    def show_best(self):
        print()
        self.print_best()
        self.save_best_network_image()
        img = self.get_best()(self.inputs).cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()
        
    def save_best_img(self, fname):
        b = self.get_best()
        if b is None:
            return
        b.to(self.inputs.device)
        img = b(self.inputs, channel_first=False)
        if len(self.config.color_mode)<3:
            img = img.unsqueeze(-1).repeat(1,1,3)
        img = img.detach().cpu().numpy()
        plt.imsave(fname, img, cmap='gray')
        plt.close()
        # if hasattr(self, "this_gen_best") and self.this_gen_best is not None:
        #     img = self.this_gen_best.get_image(self.inputs)
        #     if len(self.config.color_mode)<3:
        #         img = img.repeat(1,1,3)
        #     img = img.detach().cpu().numpy()
        #     plt.imsave(fname.replace(".png","_final.png"), img, cmap='gray')

    def save_best_network_image(self):
        best = self.get_best()
        path = f"{self.config.output_dir}/genomes/best_{self.gen}.png"
        visualize_network(self.get_best(), sample=False, save_name=path, extra_text=f"Run {self.run_number} Generation: " + str(self.gen) + " fit: " + f"{best.fitness.item():.3f}" + " species: " + str(best.species_id))
     
    def print_fitnesses(self):
        div = calculate_diversity_stochastic(self.population)
        print("Generation", self.gen, "="*100)
        class Dummy:
            def __init__(self):
                
                self.fitness = 0; self.id = -1
        b = self.get_best()
        if b is None: b = Dummy()
        print(f" |-Best: {b.id} ({b.fitness:.4f})")
        if len(self.population) > 0:
            print(f" |  Average fitness: {torch.mean(torch.stack([i.fitness for i in self.population])):.7f} | adjusted: {torch.mean(torch.stack([i.adjusted_fitness for i in self.population])):.7f}")
            print(f" |  Diversity: std: {div[0]:.3f} | avg: {div[1]:.3f} | max: {div[2]:.3f}")
            print(f" |  Connections: avg. {get_avg_number_of_connections(self.population):.2f} max. {get_max_number_of_connections(self.population)}  | H. Nodes: avg. {get_avg_number_of_hidden_nodes(self.population):.2f} max: {get_max_number_of_hidden_nodes(self.population)}")
        for individual in self.population:
            print(f" |     Individual {individual.id} ({len(individual.hidden_nodes())}n, {len(list(individual.enabled_connections()))}c, s: {individual.species_id} fit: {individual.fitness:.4f}")
        
        print(f" Gen "+ str(self.gen), f"fitness: {b.fitness:.4f}")
        print()
        

def calculate_diversity_full(population):
    if len(population) == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    # very slow, compares every genome against every other
    diffs = []
    for i in population:
        for j in population:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff

def calculate_diversity_stochastic(population):
    if len(population) == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    # compare 10% of population
    diffs = torch.zeros(len(population)//10, device=population[0].config.device)
    pop = population
    num = len(pop)//10
    pairs = itertools.combinations(pop, 2)
    pairs = random.sample(list(pairs), num)
    for i, (g1, g2) in enumerate(pairs):
        diffs[i] = g1.genetic_difference(g2)
        assert not torch.isnan(diffs[i]).any(), f"nan in diffs {i} {g1.id} {g2.id}" 
    max_diff = torch.max(diffs) if(len(diffs)>0) else torch.tensor(0).to(population[0].config.device)
    if max_diff == 0:
        return torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
    std_distance = torch.std(diffs)
    avg_distance = torch.mean(diffs)
    return std_distance, avg_distance, max_diff
