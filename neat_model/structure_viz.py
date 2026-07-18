import pickle
import neat
import graphviz


def draw_net(config, genome, filename='best_snake_1.pkl', node_names=None):
    """
    Parses a NEAT genome and generates a Graphviz diagram.
    """
    if node_names is None:
        node_names = {}

    # Initialize Graphviz Digraph
    dot = graphviz.Digraph(format='png',
                           node_attr={'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'})

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    # 1. Draw Input Nodes
    for n in inputs:
        name = node_names.get(n, str(n))
        dot.node(str(n), name, color='lightblue', style='filled', shape='box')

    # 2. Draw Output Nodes
    for n in outputs:
        name = node_names.get(n, str(n))
        dot.node(str(n), name, color='lightgreen', style='filled')

    # 3. Draw Hidden Nodes
    for n in genome.nodes.keys():
        if n not in inputs and n not in outputs:
            name = node_names.get(n, str(n))
            dot.node(str(n), name, color='white', style='filled')

    # 4. Draw Connections
    for (in_node, out_node), cg in genome.connections.items():
        if cg.enabled:
            # Green for positive weights, red for negative
            color = 'green' if cg.weight > 0 else 'red'
            # Line thickness based on weight magnitude
            width = str(0.1 + abs(cg.weight / 5.0))

            dot.edge(
                str(in_node),
                str(out_node),
                _attributes={'color': color, 'penwidth': width}
            )

    # Render and save the file
    dot.render(filename, view=True)
    print(f"Network graph saved to {filename}.png")


if __name__ == '__main__':
    # Load the NEAT config
    # Replace 'config-feedforward' with your actual .ini file path
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-snake.txt')

    # Load the pickled genome
    # Replace 'winner.pkl' with your actual .pkl file path
    with open('best_snake_3.pkl', 'rb') as f:
        winner_genome = pickle.load(f)

    # Map the node IDs to readable labels.
        # Map the 7 inputs and 4 outputs based on the config
        labels = {
            # Inputs (IDs -1 to -7)
            -1: 'Sensor 1',
            -2: 'Sensor 2',
            -3: 'Sensor 3',
            -4: 'Sensor 4',
            -5: 'Sensor 5',
            -6: 'Sensor 6',
            -7: 'Sensor 7',

            # Outputs (IDs 0 to 3)
            0: 'Action 1',
            1: 'Action 2',
            2: 'Action 3',
            3: 'Action 4'
        }

    draw_net(config, winner_genome, filename='winner_network', node_names=labels)