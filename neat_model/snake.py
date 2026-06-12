import pygame
import random
import sys
import neat
import os
import math
import pickle


# --- Configuration ---
pygame.init()
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Snake - Clean Slate")
clock = pygame.time.Clock()

generation = 0


# --- Classes ---
class Snake:
    def __init__(self):
        start_x = (WIDTH // 2) // GRID_SIZE * GRID_SIZE
        start_y = (HEIGHT // 2) // GRID_SIZE * GRID_SIZE

        self.body = [(start_x, start_y), (start_x - GRID_SIZE, start_y), (start_x - 2 * GRID_SIZE, start_y)]
        self.direction = (GRID_SIZE, 0)

        self.alive = True
        self.score = 0
        self.frames_without_food = 0

        self.color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))
        self.food = self.spawn_food()

    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, GRID_SIZE)
            y = random.randrange(0, HEIGHT, GRID_SIZE)
            if (x, y) not in self.body:
                return (x, y)

    def move(self):
        if not self.alive: return

        self.frames_without_food += 1
        # Strict 5-second starvation timer (60 FPS * 5 = 300 frames)
        if self.frames_without_food >= 480:
            self.alive = False
            return

        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # Wall Collision
        if new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT:
            self.alive = False
            return

        # Self Collision
        if new_head in self.body:
            self.alive = False
            return

        self.body.insert(0, new_head)

        # Food Collision
        if new_head == self.food:
            self.score += 1
            self.frames_without_food = 0
            self.food = self.spawn_food()
        else:
            self.body.pop()

    def get_vision(self):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction

        # 1. Normalize the current heading
        nx = dir_x // GRID_SIZE
        ny = dir_y // GRID_SIZE

        # 2. Calculate the 5 Relative Hazard Directions
        front = (nx, ny)
        left = (ny, -nx)
        right = (-ny, nx)
        front_left = (nx + ny, ny - nx)  # Combines Front and Left
        front_right = (nx - ny, ny + nx)  # Combines Front and Right

        vision_inputs = []

        # 3. Raycast for Hazards in all 5 directions
        # Order: [Front, Front-Left, Left, Front-Right, Right]
        for dx, dy in [front, front_left, left, front_right, right]:
            curr_x, curr_y = head_x, head_y
            dist = 0
            while True:
                curr_x += dx * GRID_SIZE
                curr_y += dy * GRID_SIZE
                dist += 1

                hit_wall = (curr_x < 0 or curr_x >= WIDTH or curr_y < 0 or curr_y >= HEIGHT)
                hit_body = (curr_x, curr_y) in self.body

                if hit_wall or hit_body:
                    vision_inputs.append(1.0 / dist)
                    break

        # 4. Distance to Food
        food_x, food_y = self.food
        dist_to_food = math.hypot((food_x - head_x) / GRID_SIZE, (food_y - head_y) / GRID_SIZE)
        vision_inputs.append(1.0 / dist_to_food)

        # 5. Angle to Food
        snake_angle = math.atan2(dir_y, dir_x)
        food_angle = math.atan2(food_y - head_y, food_x - head_x)

        angle_diff = food_angle - snake_angle

        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi

        vision_inputs.append(angle_diff / math.pi)

        # Returns exactly 7 values:
        # [Front, Front-Left, Left, Front-Right, Right, Food_Dist, Food_Angle]
        return vision_inputs

    def draw(self, surface):
        if not self.alive: return
        for segment in self.body:
            pygame.draw.rect(surface, self.color, (segment[0], segment[1], GRID_SIZE - 1, GRID_SIZE - 1))
        pygame.draw.rect(surface, (255, 50, 50), (self.food[0], self.food[1], GRID_SIZE - 1, GRID_SIZE - 1))


# --- NEAT Fitness Function ---
def eval_genomes(genomes, config):
    global generation
    generation += 1

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
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((20, 20, 20))

        for x, snake in enumerate(snakes):
            inputs = snake.get_vision()
            output = nets[x].activate(inputs)
            decision = output.index(max(output))

            # --- THE RELATIVE STEERING BUG FIX ---
            dir_x, dir_y = snake.direction

            if decision == 0:
                pass  # Go Straight (Do nothing to the direction)
            elif decision == 1:
                # Turn Right (90-degree vector rotation)
                snake.direction = (-dir_y, dir_x)
            elif decision == 2:
                # Turn Left (90-degree vector rotation)
                snake.direction = (dir_y, -dir_x)


            # --- MEASURE DISTANCE BEFORE MOVING ---
            dist_before = math.hypot(snake.body[0][0] - snake.food[0], snake.body[0][1] - snake.food[1])

            snake.move()

            if snake.alive:
                ge[x].fitness += 0.02

                # --- MEASURE DISTANCE AFTER MOVING ---
                dist_after = math.hypot(snake.body[0][0] - snake.food[0], snake.body[0][1] - snake.food[1])

                # The Breadcrumb Reward
                if dist_after < dist_before:
                    ge[x].fitness += 0.1  # Reward for moving toward the apple
                else:
                    ge[x].fitness -= 0.09  # Penalty for moving away or driving parallel

                snake.draw(screen)
            else:
                ge[x].fitness -= 1  # Penalty for hitting a wall or starving
                ge[x].fitness += (snake.score * 10)  # Big bonus for actually eating!

                snakes.pop(x)
                nets.pop(x)
                ge.pop(x)

        # Minimal UI
        alive_text = font.render(f"Alive: {len(snakes)}", True, (255, 255, 255))
        gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
        screen.blit(alive_text, (10, 10))
        screen.blit(gen_text, (10, 50))

        pygame.display.flip()


# --- NEAT Setup ---
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 1. Capture the winner after 100 generations (or if it hits the fitness threshold)
    print("\n--- Starting Evolution ---")
    winner = p.run(eval_genomes, 100)

    # 2. Save the champion's brain to a file
    with open("best_snake.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nTraining Complete!")
    print("Saved the absolute best snake to 'best_snake.pkl'")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-snake.txt")
    run(config_path)