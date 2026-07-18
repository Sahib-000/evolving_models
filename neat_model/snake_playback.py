import pygame
import random
import pickle
import neat
import os
import sys
import math

# --- Configuration ---
pygame.init()
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Snake - Champion Playback")
clock = pygame.time.Clock()


# --- Snake Class (Identical to your trained environment) ---
class Snake:
    def __init__(self):
        start_x = (WIDTH // 2) // GRID_SIZE * GRID_SIZE
        start_y = (HEIGHT // 2) // GRID_SIZE * GRID_SIZE

        self.body = [(start_x, start_y), (start_x - GRID_SIZE, start_y), (start_x - 2 * GRID_SIZE, start_y)]
        self.direction = (GRID_SIZE, 0)

        self.alive = True
        self.score = 0
        self.frames_without_food = 0

        self.color = (50, 200, 50)  # Make the champion green!
        self.food = self.spawn_food()

    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, GRID_SIZE)
            y = random.randrange(0, HEIGHT, GRID_SIZE)
            if (x, y) not in self.body:
                return (x, y)

    import random  # localized import for spawn_food

    def move(self):
        if not self.alive: return

        self.frames_without_food += 1
        if self.frames_without_food >= 300:  # 5 second starvation
            self.alive = False
            return

        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        # Wall & Body Collisions
        if new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT:
            self.alive = False
            return
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
        # The 7-Input Relative Vision
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction

        nx = dir_x // GRID_SIZE
        ny = dir_y // GRID_SIZE

        front = (nx, ny)
        left = (ny, -nx)
        right = (-ny, nx)
        front_left = (nx + ny, ny - nx)
        front_right = (nx - ny, ny + nx)

        vision_inputs = []

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

        food_x, food_y = self.food
        dist_to_food = math.hypot((food_x - head_x) / GRID_SIZE, (food_y - head_y) / GRID_SIZE)
        vision_inputs.append(1.0 / dist_to_food)

        snake_angle = math.atan2(dir_y, dir_x)
        food_angle = math.atan2(food_y - head_y, food_x - head_x)
        angle_diff = food_angle - snake_angle

        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi

        vision_inputs.append(angle_diff / math.pi)

        return vision_inputs

    def draw(self, surface):
        for segment in self.body:
            pygame.draw.rect(surface, self.color, (segment[0], segment[1], GRID_SIZE - 1, GRID_SIZE - 1))
        pygame.draw.rect(surface, (255, 50, 50), (self.food[0], self.food[1], GRID_SIZE - 1, GRID_SIZE - 1))


# --- Playback Engine ---
def replay_genome(config_path, genome_path):
    # Load the Config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # Load the Champion Brain
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Create the network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    snake = Snake()
    font = pygame.font.SysFont(None, 40)

    print(f"Loaded Champion! Previous Fitness: {genome.fitness}")

    running = True
    while running:
        # Watchable speed (15 FPS)
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get vision and activate brain
        inputs = snake.get_vision()
        output = net.activate(inputs)
        decision = output.index(max(output))

        # 3-Output Relative Steering
        dir_x, dir_y = snake.direction
        if decision == 0:
            pass  # Straight
        elif decision == 1:
            snake.direction = (-dir_y, dir_x)  # Right
        elif decision == 2:
            snake.direction = (dir_y, -dir_x)  # Left

        snake.move()

        # If the snake dies, instantly resurrect it for endless viewing
        if not snake.alive:
            print(f"Snake died with a score of: {snake.score}. Restarting...")
            snake = Snake()

        # Draw the screen
        screen.fill((20, 20, 20))
        snake.draw(screen)

        score_text = font.render(f"Apples Eaten: {snake.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-snake.txt")
    # Change this filename if you want to test snake 2 or 3!
    genome_path = os.path.join(local_dir, "best_snake_1.pkl")

    replay_genome(config_path, genome_path)