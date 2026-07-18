## ENN - Evolving Neural Network

NEAT (NeuroEvolution of Augmenting Topologies) is an implementation of ENN, a genetic algorithm that evolves artificial neural networks. While traditional deep learning only optimizes connection weights, this tool dynamically evolves both network weights and structural topology over time, allowing the AI to discover the most efficient network shape for itself.

—

## How It Works

The tool automates the evolution of neural networks through an iterative, generation-based cycle:

1. **Initialization:** The process begins with a population of basic neural networks featuring minimal structural complexity. 

2. **Evaluation:** Each neural network in the current population is tested within the target environment to calculate its **fitness score** (how well it performed the task). 

3. **Selection & Reproduction:** Networks with the highest fitness scores are selected as "parents." Their genetic structures are crossed over and mutated to generate a new, potentially better population of neural networks. 

4. **Termination:** This evolutionary loop repeats automatically generation after generation until a network reaches the predefined target fitness score. 

