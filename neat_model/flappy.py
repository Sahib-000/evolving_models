import pygame
import random
import sys
import neat
import visualize
import os


# --- Configuration & Classes ---
# (KEEP YOUR Pygame init, WIDTH, HEIGHT, Bird, and Pipe classes exactly as they were here)
pygame.init()
WIDTH, HEIGHT = 500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Flappy Bird Training")
clock = pygame.time.Clock()


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.gravity = 1.2
        self.jump_power = -10.5
        self.radius = 15

    def jump(self):
        self.velocity = self.jump_power

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 200, 0), (int(self.x), int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 70
        self.gap = 200
        self.height = random.randint(100, 450)
        self.passed = False
        self.speed = 5

    def move(self):
        self.x -= self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 180, 0), (self.x, 0, self.width, self.height))
        bottom_y = self.height + self.gap
        pygame.draw.rect(surface, (0, 180, 0), (self.x, bottom_y, self.width, HEIGHT - bottom_y))

    def collide(self, bird_rect):
        top_rect = pygame.Rect(self.x, 0, self.width, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.gap, self.width, HEIGHT - (self.height + self.gap))
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


# --- NEAT Fitness Function ---
def eval_genomes(genomes, config):
    nets = []
    ge = []
    birds = []

    # Set up the population for this generation
    for genome_id, genome in genomes:
        genome.fitness = 0  # Start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(150, 350))
        ge.append(genome)

    pipes = [Pipe(600)]
    score = 0
    font = pygame.font.SysFont(None, 60)

    running = True
    while running and len(birds) > 0:  # Run until all birds in this generation die
        clock.tick(60)  # Sped up to 60fps for faster training

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Determine which pipe the birds should look at
        pipe_ind = 0
        if len(birds) > 0:
            # If the birds passed the first pipe, look at the second one
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].width:
                pipe_ind = 1

        # 1. AI Decision Making
        for x, bird in enumerate(birds):
            bird.move()
            # Reward birds slightly for every frame they stay alive
            ge[x].fitness += 0.1

            # Give the AI its inputs: (Bird Y, Distance to Pipe X, Top Pipe Height)
            output = nets[x].activate((bird.y, abs(bird.x - pipes[pipe_ind].x), pipes[pipe_ind].height))

            # Output is a list. If the first output neuron is > 0.5, jump!
            if output[0] > 0.5:
                bird.jump()

        # 2. Environment Logic
        screen.fill((135, 206, 235))
        rem_pipes = []
        add_pipe = False

        for pipe in pipes:
            pipe.move()

            # Check collisions
            for x, bird in enumerate(birds):
                if pipe.collide(bird.get_rect()):
                    ge[x].fitness -= 1  # Penalize hitting a pipe
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.width < 0:
                rem_pipes.append(pipe)

            pipe.draw(screen)

        if add_pipe:
            score += 1
            # Reward birds heavily for passing a pipe
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(600))

        for r in rem_pipes:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            # Check ground / ceiling collision
            if bird.y + bird.radius >= HEIGHT or bird.y - bird.radius <= 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
            else:
                bird.draw(screen)

        # 3. Drawing UI
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        alive_text = font.render(f"Alive: {len(birds)}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))
        screen.blit(alive_text, (20, 70))
        pygame.display.flip()


# --- NEAT Setup ---
# --- NEAT Setup ---
def run(config_path):
    # CORRECTED ORDER: Genome, Reproduction, SpeciesSet, Stagnation
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # Create the population
    p = neat.Population(config)

    # Add terminal output so we can see progress in the console
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run the fitness function for up to 50 generations
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))
    # Run the fitness function for up to 50 generations
    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

    # # --- NEW VISUALIZATION CODE ---
    # # Draw the neural network
    # node_names = {-1: 'Bird Y', -2: 'Dist to Pipe', -3: 'Pipe Height', 0: 'Jump'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    #
    # # Plot fitness and species stats (optional but cool)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)