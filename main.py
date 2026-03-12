import torch
import torch.nn as nn
import numpy as np
import random
import pygame

VERSION = 1.0
SPEED = 3.0  

def save_best_individuals(preys, filename="best_individuals.pt"):
    sorted_preys = sorted(preys, key=lambda x: x.energy, reverse=True)
    best_preys = sorted_preys[:10] 
    states = [prey.brain.state_dict() for prey in best_preys]
    torch.save(states, filename)

def load_best_individuals(filename="best_individuals.pt"):
    import os
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            states = torch.load(filename)
            return states
        except Exception as e:
            print(f"Erreur de chargement : {e}. Le fichier sera ignoré.")
            return []
    else:
        return []

class Individu(nn.Module):
    def __init__(self, is_prey=True):
        super().__init__()
        self.is_prey = is_prey
        self.x = random.uniform(0, 800)
        self.y = random.uniform(0, 600)
        self.energy = 100
        self.speed = 2 if is_prey else 1.5
        self.brain = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.brain(inputs)

    def update(self, env):
        if self.is_prey:
            if env.sharks:
                min_dist_to_sharks = min(
                    (((self.x - shark.x)**2 + (self.y - shark.y)**2)**0.5 for shark in env.sharks),
                    default=800
                )
            else:
                min_dist_to_sharks = 800

            min_dist_to_border = min(self.x, self.y, 800 - self.x, 600 - self.y)
            inputs = torch.tensor([
                self.x / 800, self.y / 600,
                min_dist_to_sharks / 800,
                min_dist_to_border / 400
            ], dtype=torch.float32)
        else:
            if env.preys:
                min_dist_to_preys = min(
                    (((self.x - prey.x)**2 + (self.y - prey.y)**2)**0.5 for prey in env.preys),
                    default=800
                )
            else:
                min_dist_to_preys = 800 

            min_dist_to_border = min(self.x, self.y, 800 - self.x, 600 - self.y)
            inputs = torch.tensor([
                self.x / 800, self.y / 600,
                min_dist_to_preys / 800,
                min_dist_to_border / 400
            ], dtype=torch.float32)

        with torch.no_grad():
            dx, dy = self.forward(inputs)
        self.x += dx * self.speed * SPEED
        self.y += dy * self.speed * SPEED
        self.x = max(0, min(800, self.x))
        self.y = max(0, min(600, self.y))
        self.energy -= 0.1 * SPEED

class Environnement:
    def __init__(self, n_preys=50, n_sharks=5):
        self.preys = []
        best_states = load_best_individuals()
        if best_states:
            for state in best_states[:n_preys//2]:
                prey = Individu(is_prey=True)
                prey.brain.load_state_dict(state)
                self.preys.append(prey)
            while len(self.preys) < n_preys:
                self.preys.append(Individu(is_prey=True))
        else:
            self.preys = [Individu(is_prey=True) for _ in range(n_preys)]
        self.sharks = [Individu(is_prey=False) for _ in range(n_sharks)]

    def update(self):
        for prey in self.preys:
            prey.update(self)
        for shark in self.sharks:
            shark.update(self)
        self.check_collisions()
        self.reproduce()

    def check_collisions(self):
        for prey in self.preys[:]:
            for shark in self.sharks:
                if ((prey.x - shark.x)**2 + (prey.y - shark.y)**2)**0.5 < 20:
                    if prey in self.preys:
                        self.preys.remove(prey)
                    shark.energy += 50
                    break

    def reproduce(self):
        if len(self.preys) < 50 and len(self.preys) > 0:
            best_states = load_best_individuals()
            if best_states:
                parent_state = random.choice(best_states)
                child = Individu(is_prey=True)
                child.brain.load_state_dict(parent_state)
                with torch.no_grad():
                    for param in child.brain.parameters():
                        param.data += 0.1 * torch.randn_like(param.data)
                self.preys.append(child)
            else:
                parent = random.choice(self.preys)
                child = Individu(is_prey=True)
                with torch.no_grad():
                    for param_parent, param_child in zip(parent.brain.parameters(), child.brain.parameters()):
                        param_child.data = param_parent.data + 0.1 * torch.randn_like(param_parent.data)
                self.preys.append(child)

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"AquaTorch {VERSION}")
    clock = pygame.time.Clock()
    env = Environnement()

    font = pygame.font.SysFont("Arial", 16)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.update()
        screen.fill((0, 0, 50))

        speed_text = font.render(f"Vitesse: {SPEED:.1f}x", True, (255, 255, 255))
        screen.blit(speed_text, (10, 10))

        for prey in env.preys:
            pygame.draw.circle(screen, (0, 200, 0), (int(prey.x), int(prey.y)), 5)
        for shark in env.sharks:
            pygame.draw.circle(screen, (200, 0, 0), (int(shark.x), int(shark.y)), 10)

        pygame.display.flip()
        clock.tick(30)

    save_best_individuals(env.preys)
    pygame.quit()

if __name__ == "__main__":
    main()
