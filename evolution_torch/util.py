import math
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import torch
import warnings
import imageio as iio

import fitness_functions as ff
import cppn_torch.activation_functions as af
from cppn_torch.cppn import NodeType
from cppn_torch.graph_util import get_ids_from_individual, required_for_output

BASELINE_PATH = "baseline/"

warnings.filterwarnings("ignore") # uh oh
    
def _get_baseline_max_min_fits():
    """Get the baseline max and min fitnesses for each function."""
    if not os.path.exists(BASELINE_PATH):
        return {}, {}
    max_fits = {}
    min_fits = {}
    fns_dir = os.path.join(BASELINE_PATH, "fits")
    if not os.path.exists(fns_dir):
        return max_fits, min_fits
    for fn in os.listdir(fns_dir):
        fn_path = os.path.join(fns_dir, fn)
        min_file = os.path.join(fn_path, "min_fit.txt")
        max_file = os.path.join(fn_path, "max_fit.txt")
        with open(min_file, 'r') as f:
            min_fits[fn] = float(f.read())
        with open(max_file, 'r') as f:
            max_fits[fn] = float(f.read())
    return max_fits, min_fits   

def visualize_network(individual, sample_point=None, color_mode="L", visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None, curved=False, return_fig=False):
    # TODO: total mess
    c = individual.config
    if(sample):
        if sample_point is None:
            sample_point = [.25]*c.num_inputs
        individual.eval(sample_point)
            
        
    nodes = individual.node_genome
    connections = individual.connection_genome.items()

    if not visualize_disabled:
        req = required_for_output(*get_ids_from_individual(individual))
        nodes = {k: v for k, v in nodes.items() if v.id in req or v.type == NodeType.INPUT or v.type == NodeType.OUTPUT}

    max_weight = c.max_weight

    G = nx.DiGraph()
    function_colors = {}
    colors = ['lightsteelblue'] * len([node.activation for node in individual.node_genome.values()])
    node_labels = {}

    node_size = 2000
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    for i, fn in enumerate([node.activation for node in individual.node_genome.values()]):
        function_colors[fn.__name__] = colors[i]
    function_colors["identity"] = colors[0]

    fixed_positions={}
    inputs = {k:v for k,v in nodes.items() if v.type==NodeType.INPUT}
    for i, node in enumerate(inputs.values()):
        if node.type == NodeType.INPUT:
            if not visualize_disabled and node.layer == 999:
                continue
            G.add_node(node, color=function_colors[node.activation.__name__], shape='d', layer=(node.layer))
            if node.type == 0:
                node_labels[node] = f"input{i}\n{node.id}"
                
            fixed_positions[node] = (-4,((i+1)*2.)/len(inputs))

    for node in nodes.values():
        if node.type == NodeType.HIDDEN:
            if not visualize_disabled and node.layer == 999:
                continue
            G.add_node(node, color=function_colors[node.activation.__name__], shape='o', layer=(node.layer))
            node_labels[node] = f"{node.id}\n{node.activation.__name__}"

    for i, node in enumerate(nodes.values()):
        if node.type == NodeType.OUTPUT:
            if not visualize_disabled and node.layer == 999:
                continue
            title = i
            G.add_node(node, color=function_colors[node.activation.__name__], shape='s', layer=(node.layer))
            node_labels[node] = f"{node.id}\noutput{title}:\n{node.activation.__name__}"
            fixed_positions[node] = (4, ((i+1)*2)/len(individual.output_nodes()))
    pos = {}
    fixed_nodes = fixed_positions.keys()
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4,subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    plt.figure(figsize=(10, 10))
    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        this_nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in this_nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=this_nodes)

    edge_labels = {}
    for key, cx in connections:
        if key[0] not in nodes.keys() or key[1] not in nodes.keys():
            continue
        w = cx.weight.item()
        if not visualize_disabled and not cx.enabled:
            continue
        style = ('-', 'k',  .5+abs(w)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(w)/max_weight)
        if(cx.enabled and w<0): style  = ('-', 'r', .5+abs(w)/max_weight)
        from_node = nodes[key[0]]
        to_node = nodes[key[1]]
        if from_node in G.nodes and to_node in G.nodes:
            G.add_edge(from_node, to_node, weight=f"{w:.4f}", pos=pos, style=style)
        else:
            print("Connection not in graph:", from_node.id, "->", to_node.id)
        edge_labels[(from_node, to_node)] = f"{w:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))
    use_curved = curved
    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                connectionstyle= "arc3" if not use_curved else f"arc3,rad={0.2*random.random()}",
                            )
    
    if extra_text is not None:
        plt.text(0.5,0.05, extra_text, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        
    
    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    if return_fig:
        return plt.gcf()
    elif save_name is not None:
        plt.savefig(save_name, format="PNG")
    else:
        plt.show()

def visualize_hn_phenotype_network(individual, visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None):
    
    connections = individual.connections
    node_genome = individual.nodes
    c = individual.config
    input_nodes = [n for n in node_genome if n.type == 0]
    output_nodes = [n for n in node_genome if n.type == 1]
    hidden_nodes = [n for n in node_genome if n.type == 2]
    max_weight = c.max_weight

    G = nx.DiGraph()
    function_colors = {}
    colors = ['lightsteelblue'] * len([node.activation for node in node_genome])
    node_labels = {}

    node_size = 2000
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    for i, fn in enumerate([node.activation for node in node_genome]):
        function_colors[fn.__name__] = colors[i]
    function_colors["identity"] = colors[0]

    fixed_positions={}
    inputs = input_nodes
    
    for i, node in enumerate(inputs):
        G.add_node(node, color=function_colors[node.activation.__name__], shape='d', layer=(node.layer))
        if node.type == 0:
            node_labels[node] = f"S{i}:\n{node.activation.__name__}\n"+(f"{node.outputs:.3f}" if node.outputs!=None else "")
        else:
            node_labels[node] = f"CPG"
            
        fixed_positions[node] = (-4,((i+1)*2.)/len(inputs))

    for node in hidden_nodes:
        G.add_node(node, color=function_colors[node.activation.__name__], shape='o', layer=(node.layer))
        node_labels[node] = f"{node.id}\n{node.activation.__name__}\n"+(f"{node.outputs:.3f}" if node.outputs!=None else "" )

    for i, node in enumerate(output_nodes):
        title = i
        G.add_node(node, color=function_colors[node.activation.__name__], shape='s', layer=(node.layer))
        node_labels[node] = f"M{title}:\n{node.activation.__name__}\n"+(f"{node.outputs:.3f}")
        fixed_positions[node] = (4, ((i+1)*2)/len(output_nodes))
    pos = {}
    fixed_nodes = fixed_positions.keys()
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4,subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    plt.figure(figsize=(10, 10))
    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        this_nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in this_nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=this_nodes)

    edge_labels = {}
    for cx in connections:
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight, 0))): continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(cx.weight)/max_weight)
        if(cx.enabled and cx.weight<0): style  = ('-', 'r', .5+abs(cx.weight)/max_weight)
        if cx.from_node in G.nodes and cx.to_node in G.nodes:
            G.add_edge(cx.from_node, cx.to_node, weight=f"{cx.weight:.4f}", pos=pos, style=style)
        else:
            print("Connection not in graph:", cx.from_node.id, "->", cx.to_node.id)
        edge_labels[(cx.from_node, cx.to_node)] = f"{cx.weight:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))

    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                connectionstyle= "arc3"
                            )
    
    if extra_text is not None:
        plt.text(0.5,0.05, extra_text, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        
    
    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, format="PNG")
    else:
        plt.show()
    ""

def print_net(individual, show_weights=False, visualize_disabled=False):
    print(f"<CPPN {individual.id}")
    print(f"nodes:")
    for k, v in individual.node_genome.items():
        print("\t",k, "\t|\t",v.layer, "\t|\t",v.activation.__name__)
    print(f"connections:")
    for k, v in individual.connection_genome.items():
        print("\t",k, "\t|\t",v.enabled, "\t|\t",v.weight)
    print(">")
  
def get_network_images(networks):
    imgs = []
    for net in networks:
        fig = visualize_network(net, return_fig=True)
        ax = fig.gca()
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(img)  
    return imgs

def get_best_solution_from_all_runs(results):
    best_fit = -math.inf
    best = None
    run_index = -1
    for i, run in enumerate(results):
        sorted_run = sorted(run, key = lambda x: x.fitness, reverse=True)
        run_best = sorted_run[0]
        if(run_best.fitness > best_fit):
            best_fit = run_best.fitness
            best = run_best
            run_index = i
    return best, run_index


def get_max_number_of_hidden_nodes(population):
    max = 0
    for g in population:
        if len(list(g.hidden_nodes()))> max:
            max = len(list(g.hidden_nodes()))
    return max

def get_avg_number_of_hidden_nodes(population):
    count = 0
    for g in population:
        count+=len(g.node_genome) - g.n_inputs - g.n_outputs
    return count/len(population)

def get_max_number_of_connections(population):
    max_count = 0
    for g in population:
        count = len(list(g.enabled_connections()))
        if(count > max_count):
            max_count = count
    return max_count

def get_min_number_of_connections(population):
    min_count = math.inf
    for g in population:
        count = len(list(g.enabled_connections())) 
        if(count < min_count):
            min_count = count
    return min_count

def get_avg_number_of_connections(population):
    count = 0
    for g in population:
        count+=len(list(g.enabled_connections()))
    return count/len(population)

