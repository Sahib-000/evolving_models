import pygame
import random
import sys

# --- Configuration ---
pygame.init()
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Snake Environment")
clock = pygame.time.Clock()


# --- Classes ---
class Snake:
    def __init__(self):
        # Start in the middle of the screen
        start_x = (WIDTH // 2) // GRID_SIZE * GRID_SIZE
        start_y = (HEIGHT // 2) // GRID_SIZE * GRID_SIZE

        # Body is a list of (x, y) tuples. Head is index 0.
        self.body = [(start_x, start_y), (start_x - GRID_SIZE, start_y), (start_x - 2 * GRID_SIZE, start_y)]
        self.direction = (GRID_SIZE, 0)  # Moving right

        self.alive = True
        self.score = 0

        # Give each snake a random color so we can tell them apart when there are 50!
        self.color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))

        # Each snake tracks its own personal food target
        self.food = self.spawn_food()

    def get_vision(self):
        """Provide vision input for the snake"""
        # The 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        vision_inputs = []
        head_x, head_y = self.body[0]

        for dx, dy in directions:
            dist_to_wall = 0
            dist_to_food = 0
            dist_to_body = 0

            curr_x, curr_y = head_x, head_y
            distance = 0

            food_found = False
            body_found = False

            # Shoot the ray out one grid step at a time
            while True:
                curr_x += dx * GRID_SIZE
                curr_y += dy * GRID_SIZE
                distance += 1

                # 1. Check if the ray hit a wall
                if curr_x < 0 or curr_x >= WIDTH or curr_y < 0 or curr_y >= HEIGHT:
                    # Invert the distance so 1.0 is touching, and lower means further away
                    dist_to_wall = 1.0 / distance
                    break  # Stop the ray, it can't see past a wall

                # 2. Check if the ray hit the personal food target
                if not food_found and (curr_x, curr_y) == self.food:
                    dist_to_food = 1.0 / distance
                    food_found = True  # We found it, but keep the ray going to find walls/body

                # 3. Check if the ray hit the snake's own body
                if not body_found and (curr_x, curr_y) in self.body:
                    dist_to_body = 1.0 / distance
                    body_found = True

            # Add the 3 values for this specific direction to our flat list of 24 inputs
            vision_inputs.extend([dist_to_wall, dist_to_food, dist_to_body])

        return vision_inputs
    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, GRID_SIZE)
            y = random.randrange(0, HEIGHT, GRID_SIZE)
            # Make sure food doesn't spawn inside the snake's body
            if (x, y) not in self.body:
                return (x, y)

    def move(self):
        if not self.alive:
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

        # Move the snake forward
        self.body.insert(0, new_head)

        # 3. Check Food Collision
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            # If we didn't eat, remove the tail piece so we don't grow infinitely
            self.body.pop()

    def draw(self, surface):
        if not self.alive:
            return

        # Draw the snake body
        for segment in self.body:
            pygame.draw.rect(surface, self.color, (segment[0], segment[1], GRID_SIZE - 1, GRID_SIZE - 1))

        # Draw this specific snake's personal food (Red)
        pygame.draw.rect(surface, (255, 50, 50), (self.food[0], self.food[1], GRID_SIZE - 1, GRID_SIZE - 1))


# --- Main Game Loop ---
def main():
    # Start with just one snake for testing
    snakes = [Snake()]
    font = pygame.font.SysFont(None, 40)

    running = True
    while running:
        # FPS locked to 12 so a human can reasonably control it during testing
        clock.tick(12)

        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Human controls for the first snake in the list
            if event.type == pygame.KEYDOWN and len(snakes) > 0:
                # Prevent the snake from reversing directly into itself
                if event.key == pygame.K_UP and snakes[0].direction != (0, GRID_SIZE):
                    snakes[0].direction = (0, -GRID_SIZE)
                elif event.key == pygame.K_DOWN and snakes[0].direction != (0, -GRID_SIZE):
                    snakes[0].direction = (0, GRID_SIZE)
                elif event.key == pygame.K_LEFT and snakes[0].direction != (GRID_SIZE, 0):
                    snakes[0].direction = (-GRID_SIZE, 0)
                elif event.key == pygame.K_RIGHT and snakes[0].direction != (-GRID_SIZE, 0):
                    snakes[0].direction = (GRID_SIZE, 0)

        # 2. Logic & Movement
        screen.fill((20, 20, 20))  # Dark gray background

        # Move and clean up snakes
        for snake in snakes[:]:
            snake.move()
            if not snake.alive:
                snakes.remove(snake)
            else:
                snake.draw(screen)

        # Reset if all snakes die (useful for human testing)
        if len(snakes) == 0:
            snakes = [Snake()]

        # 3. Drawing UI
        if len(snakes) > 0:
            score_text = font.render(f"Score: {snakes[0].score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))

        pygame.display.flip()


if __name__ == "__main__":
    main()