import pygame
import random
import sys

# --- Configuration ---
pygame.init()
WIDTH, HEIGHT = 500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Flappy Bird Environment")
clock = pygame.time.Clock()


# --- Classes ---
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
        # Used for collision detection
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 70
        self.gap = 200  # Space between top and bottom pipe
        self.height = random.randint(100, 450)  # Height of the top pipe
        self.passed = False
        self.speed = 5

    def move(self):
        self.x -= self.speed

    def draw(self, surface):
        # Top pipe
        pygame.draw.rect(surface, (0, 180, 0), (self.x, 0, self.width, self.height))
        # Bottom pipe
        bottom_y = self.height + self.gap
        pygame.draw.rect(surface, (0, 180, 0), (self.x, bottom_y, self.width, HEIGHT - bottom_y))

    def collide(self, bird_rect):
        top_rect = pygame.Rect(self.x, 0, self.width, self.height)
        bottom_rect = pygame.Rect(self.x, self.height + self.gap, self.width, HEIGHT - (self.height + self.gap))
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


# --- Main Game Loop ---
def main():
    # We use a list so we can easily swap this to 100 AI birds later
    birds = [Bird(150, 350)]
    pipes = [Pipe(600)]
    score = 0
    font = pygame.font.SysFont(None, 60)

    running = True
    while running:
        clock.tick(30)

        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Human control for testing the physics
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                for bird in birds:
                    bird.jump()

        # 2. Logic & Movement
        screen.fill((135, 206, 235))  # Sky blue background

        # Handle Pipes
        rem_pipes = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            # Check for collisions with all surviving birds
            for bird in birds[:]:  # Iterate over a copy of the list
                if pipe.collide(bird.get_rect()):
                    birds.remove(bird)  # Bird hit a pipe, remove it

                # Check if bird passed the pipe
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            # Mark pipe for removal if it goes off screen
            if pipe.x + pipe.width < 0:
                rem_pipes.append(pipe)

            pipe.draw(screen)

        # Add new pipes and update score
        if add_pipe:
            score += 1
            pipes.append(Pipe(600))
        for r in rem_pipes:
            pipes.remove(r)

        # Handle Birds
        for bird in birds[:]:
            bird.move()
            bird.draw(screen)

            # Check ground / ceiling collision
            if bird.y + bird.radius >= HEIGHT or bird.y - bird.radius <= 0:
                birds.remove(bird)

        # Reset if all birds die (useful for human testing)
        if len(birds) == 0:
            birds = [Bird(150, 350)]
            pipes = [Pipe(600)]
            score = 0

        # 3. Drawing UI
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))

        pygame.display.flip()


if __name__ == "__main__":
    main()