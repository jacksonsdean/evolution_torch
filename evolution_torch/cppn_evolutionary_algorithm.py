import math
import random
import time
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pandas as pd
import torch
import os
from cppn_torch import Node
from cppn_torch.fitness_functions import correct_dims
from cppn_torch.graph_util import name_to_fn
from cppn_torch.util import get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, visualize_network, get_max_number_of_hidden_nodes
from evolution_torch.autoencoder import initialize_encoders, AutoEncoder

class CPPNEvolutionaryAlgorithm(object):
    def __init__(self, config, debug_output=False) -> None:
        if config.with_grad:
            torch.autograd.set_grad_enabled(True)
        else:
            torch.autograd.set_grad_enabled(False)
        self.gen = 0
        self.debug_output = debug_output
        self.show_output = False
        self.config = config
        
        self.results = pd.DataFrame(columns=['condition', 'run', 'gen', 'fitness', 'diversity', 'population', 'avg_num_connections', 'avg_num_hidden_nodes', 'max_num_connections', 'max_num_hidden_nodes', 'time'])
                
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
        
        self.solution_fitness = -math.inf
        self.best_genome = None
        self.genome_type = config.genome_type

        self.fitness_function = config.fitness_function
        
        if not isinstance(config.fitness_function, Callable):
            self.fitness_function = name_to_fn(config.fitness_function)
            self.fitness_function_normed = self.fitness_function
            
        self.target = self.config.target.to(self.device)
        
        if len(self.target.shape) < 3:
            # grayscale image
            if self.config.color_mode != "L":
                print("WARNING: target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
                self.config.color_mode = "L"
            
    
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
            initialize_encoders(self.config, self.target)  
            for g in self.population: g.get_image()
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
                    pbar.set_postfix_str(f"f: {b.fitness:.4f} d:{self.diversity:.4f}")
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
            print("WARN: run dir already exists, overwriting")
        else:
            os.makedirs(run_dir)
            
        # save to run dir
        filename = os.path.join(run_dir, f"results.pkl")
        self.results.to_pickle(filename)
        
        # save to output dir
        filename = os.path.join(self.config.output_dir, f"results.pkl")
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                save_results = pd.read_pickle(f)
                save_results = save_results.append(self.results, ignore_index=True)
        else:
            save_results = self.results
        save_results.to_pickle(filename)
        
        with open(os.path.join(run_dir, f"target.txt"), 'w') as f:
            f.write(self.config.target_name)
            
            
        
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
        imgs = torch.stack([g.get_image() for g in self.population])
        imgs, target = correct_dims(imgs, self.target)
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
        
        if len(self.population) > 0:
            self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
            if self.config.with_grad:
                self.population[0].discard_grads()
            self.this_gen_best = self.population[0].clone(cpu=True)  # still sorted by fitness
        
        # std_distance, avg_distance, max_diff = calculate_diversity_full(self.population)
        std_distance, avg_distance, max_diff = calculate_diversity_stochastic(self.population)
        self.diversity = avg_distance
        n_nodes = get_avg_number_of_hidden_nodes(self.population)
        n_connections = get_avg_number_of_connections(self.population)
        max_connections = get_max_number_of_connections(self.population)
        max_nodes = get_max_number_of_hidden_nodes(self.population)

        if not skip_fitness:
            # fitness
            if self.population[0].fitness > self.solution_fitness: # if the new parent is the best found so far
                self.solution = self.population[0]                 # update best solution records
                self.solution_fitness = self.solution.fitness
                self.solution_generation = self.gen
                self.best_genome = self.solution
            
            os.makedirs(os.path.join(self.config.output_dir, 'images'), exist_ok=True)
            self.save_best_img(os.path.join(self.config.output_dir, "images", f"current_best_output.png"))
        
        if self.solution is not None:
            self.results.loc[len(self.results.index)] = [self.config.experiment_condition, self.config.run_id, self.gen, self.solution_fitness.cpu().item(), avg_distance.item(), float(len(self.population)), n_connections, n_nodes, max_connections, max_nodes, time.time() - self.start_time]
        else:
            self.results.loc[len(self.results.index)] = [self.config.experiment_condition, self.config.run_id, self.gen, 0, avg_distance.item(), float(len(self.population)), n_connections, n_nodes, max_connections, max_nodes, time.time() - self.start_time]

    def mutate(self, child):
        rates = self.get_mutation_rates()
        child.fitness, child.adjusted_fitness = 0, 0 # new fitnesses after mutation
        child.mutate(rates)
    
    def get_best(self):
        if len(self.population) == 0:
            return None
        max_fitness_individual = max(self.population, key=lambda x: x.fitness.item())
        return max_fitness_individual
    
    def print_best(self):
        best = self.get_best()
        print("Best:", best.id, best.fitness)
        
    def show_best(self):
        print()
        self.print_best()
        self.save_best_network_image()
        img = self.get_best().get_image().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()
        
    def save_best_img(self, fname):
        b = self.get_best()
        if b is None:
            return
        img = b.get_image()
        if len(self.config.color_mode)<3:
            img = img.unsqueeze(-1).repeat(1,1,3)
        img = img.detach().cpu().numpy()
        plt.imsave(fname, img, cmap='gray')
        plt.close()
        # if hasattr(self, "this_gen_best") and self.this_gen_best is not None:
        #     img = self.this_gen_best.get_image()
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
        return 0, 0, 0
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
        return 0, 0, 0
    # compare 10% of population
    diffs = torch.zeros(len(population)//10, device=population[0].config.device)
    pop = population
    for i in range(int(len(population)//10)):
        g1 = random.choice(pop)
        g2 = random.choice(pop)
        diffs[i] = g1.genetic_difference(g2)

    std_distance = torch.std(diffs)
    avg_distance = torch.mean(diffs)
    max_diff = torch.max(diffs) if(len(diffs)>0) else torch.tensor(0).to(population[0].config.device)
    return std_distance, avg_distance, max_diff
