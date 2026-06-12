import pygame
import pickle
import math
import random
import sys
import neat
import os

# --- Configuration ---
pygame.init()
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Snake Training")
clock = pygame.time.Clock()

# --- Classes ---
class Snake:

    def __init__(self):
        start_x = (WIDTH // 2) // GRID_SIZE * GRID_SIZE
        start_y = (HEIGHT // 2) // GRID_SIZE * GRID_SIZE

        self.body = [(start_x, start_y), (start_x - GRID_SIZE, start_y), (start_x - 2 * GRID_SIZE, start_y)]
        self.direction = (GRID_SIZE, 0)

        self.alive = True
        self.score = 0

        # --- NEW 5-SECOND TIMER LOGIC ---
        # 60 FPS * 5 seconds = 300 frames
        self.frames_since_last_meal = 0
        self.max_frames_without_food = 500

        self.color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))
        self.food = self.spawn_food()

    # (Keep your spawn_food method exactly the same)

    def move(self):
        if not self.alive:
            return

        # --- 5-SECOND STARVATION CHECK ---
        self.frames_since_last_meal += 1
        if self.frames_since_last_meal >= self.max_frames_without_food:
            self.alive = False  # Kill the snake if 5 seconds pass without eating
            return

        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # 1. Check Wall Collision
        if new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT:
            self.alive = False
            return

        # 2. Check Self Collision
        if new_head in self.body:
            self.alive = False
            return

        self.body.insert(0, new_head)

        # 3. Check Food Collision
        if new_head == self.food:
            self.score += 1
            self.frames_since_last_meal = 0  # Reset the timer back to 0!
            self.food = self.spawn_food()
        else:
            self.body.pop()

    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, GRID_SIZE)
            y = random.randrange(0, HEIGHT, GRID_SIZE)
            if (x, y) not in self.body:
                return (x, y)

    def get_vision(self):
        # Only 4 directions: Up, Right, Down, Left
        directions = [
            (0, -1), (1, 0), (0, 1), (-1, 0)
        ]

        vision_inputs = []
        head_x, head_y = self.body[0]

        for dx, dy in directions:
            dist_to_wall = 0
            dist_to_food = 0
            dist_to_body = 0

            curr_x, curr_y = head_x, head_y

            food_found = False
            body_found = False

            # Shoot the ray out one grid step at a time
            while True:
                curr_x += dx * GRID_SIZE
                curr_y += dy * GRID_SIZE

                # Calculate true Euclidean distance, scaled down to grid units
                euclidean_dist = math.hypot((curr_x - head_x) / GRID_SIZE, (curr_y - head_y) / GRID_SIZE)

                # 1. Check if the ray hit a wall
                if curr_x < 0 or curr_x >= WIDTH or curr_y < 0 or curr_y >= HEIGHT:
                    dist_to_wall = 1.0 / euclidean_dist
                    break

                    # 2. Check if the ray hit the personal food target
                if not food_found and (curr_x, curr_y) == self.food:
                    dist_to_food = 1.0 / euclidean_dist
                    food_found = True

                    # 3. Check if the ray hit the snake's own body
                if not body_found and (curr_x, curr_y) in self.body:
                    dist_to_body = 1.0 / euclidean_dist
                    body_found = True

            vision_inputs.extend([dist_to_wall, dist_to_food, dist_to_body])

        return vision_inputs

    # def get_vision(self):
    #     directions = [
    #         (0, -1), (1, -1), (1, 0), (1, 1),
    #         (0, 1), (-1, 1), (-1, 0), (-1, -1)
    #     ]
    #
    #     vision_inputs = []
    #     head_x, head_y = self.body[0]
    #
    #     for dx, dy in directions:
    #         dist_to_wall = 0
    #         dist_to_food = 0
    #         dist_to_body = 0
    #
    #         curr_x, curr_y = head_x, head_y
    #         distance = 0
    #
    #         food_found = False
    #         body_found = False
    #
    #         while True:
    #             curr_x += dx * GRID_SIZE
    #             curr_y += dy * GRID_SIZE
    #             distance += 1
    #
    #             if curr_x < 0 or curr_x >= WIDTH or curr_y < 0 or curr_y >= HEIGHT:
    #                 dist_to_wall = 1.0 / distance
    #                 break
    #
    #             if not food_found and (curr_x, curr_y) == self.food:
    #                 dist_to_food = 1.0 / distance
    #                 food_found = True
    #
    #             if not body_found and (curr_x, curr_y) in self.body:
    #                 dist_to_body = 1.0 / distance
    #                 body_found = True
    #
    #         vision_inputs.extend([dist_to_wall, dist_to_food, dist_to_body])
    #
    #     return vision_inputs

    def draw(self, surface):
        if not self.alive: return
        for segment in self.body:
            pygame.draw.rect(surface, self.color, (segment[0], segment[1], GRID_SIZE - 1, GRID_SIZE - 1))
        pygame.draw.rect(surface, (255, 50, 50), (self.food[0], self.food[1], GRID_SIZE - 1, GRID_SIZE - 1))


# --- NEAT Fitness Function ---
def eval_genomes(genomes, config):
    nets = []
    ge = []
    snakes = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(Snake())
        ge.append(genome)

    font = pygame.font.SysFont(None, 40)

    running = True
    while running and len(snakes) > 0:
        # Run at 60 FPS for training speed
        clock.tick(120)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((20, 20, 20))

        for x, snake in enumerate(snakes):
            # Give the AI its 24 vision inputs
            inputs = snake.get_vision()
            output = nets[x].activate(inputs)

            # Determine the highest output node (0: Up, 1: Down, 2: Left, 3: Right)
            decision = output.index(max(output))

            # Apply movement, preventing it from reversing into itself
            if decision == 0 and snake.direction != (0, GRID_SIZE):
                snake.direction = (0, -GRID_SIZE)
            elif decision == 1 and snake.direction != (0, -GRID_SIZE):
                snake.direction = (0, GRID_SIZE)
            elif decision == 2 and snake.direction != (GRID_SIZE, 0):
                snake.direction = (-GRID_SIZE, 0)
            elif decision == 3 and snake.direction != (-GRID_SIZE, 0):
                snake.direction = (GRID_SIZE, 0)

            # Move and evaluate fitness
            snake.move()

            if snake.alive:
                ge[x].fitness += 0.01  # Tiny reward for surviving a frame
                snake.draw(screen)
            else:
                ge[x].fitness -= 2  # Penalty for dying
                # Heavy reward based on how much food it ate
                ge[x].fitness += (snake.score * 10) + 2

                # Remove dead snakes
                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)

        # Draw UI
        alive_text = font.render(f"Alive: {len(snakes)}", True, (255, 255, 255))
        screen.blit(alive_text, (10, 10))
        pygame.display.flip()


# --- NEAT Setup ---
# --- NEAT Setup ---
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    total_generations = 300
    log_filename = "fitness_log.txt"

    # Create or clear the log file before starting
    with open(log_filename, "w") as f:
        f.write("--- NEAT Snake Fitness Evolution ---\n")

    print("\n--- Starting Evolution Track ---")
    for generation in range(total_generations):
        # Run exactly 1 generation
        p.run(eval_genomes, 1)

        # Get the absolute best snake from this specific generation
        best_of_gen = stats.best_genome()

        # Estimate apples eaten (based on our fitness logic: fitness roughly equals score * 10)
        estimated_score = max(0, int((best_of_gen.fitness + 1) / 10))

        # Format a clean log entry
        log_entry = f"Generation {generation}: Best Fitness = {best_of_gen.fitness:.2f} | Approx Apples = {estimated_score}\n"
        print(f"Logging progress: {log_entry.strip()}")

        # Append the result to your local text file list
        with open(log_filename, "a") as f:
            f.write(log_entry)

    # After all 100 generations, save the ultimate champion
    with open("best_snake.pkl", "wb") as f:
        pickle.dump(stats.best_genome(), f)
    print("Saved the absolute best snake to 'best_snake.pkl'!")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-snake.txt")
    run(config_path)