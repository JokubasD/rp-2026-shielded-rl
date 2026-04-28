import pygame
import sys
from simulator import Simulator, MapConfig
from agent import AgentAction

# --- STYLING ---
COLORS = {
    "wall": (40, 44, 52),
    "floor": (200, 200, 200),
    "agent": (0, 120, 255, 200),   # RGBA
    "victim": (255, 50, 50, 255),
    "confidence": (255, 255, 0, 100), # Yellow tint for high confidence
    "ui_bg": (30, 30, 35),
    "text": (255, 255, 255),
    "accent": (0, 200, 100)
}

class Checkbox:
    def __init__(self, x, y, label, default=True):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.active = default
        self.font = pygame.font.SysFont("Arial", 16)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        if self.active:
            pygame.draw.rect(screen, COLORS["accent"], self.rect.inflate(-6, -6))
        
        label_surf = self.font.render(self.label, True, COLORS["text"])
        screen.blit(label_surf, (self.rect.right + 10, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = not self.active
            return True
        return False

class Visualizer:
    def __init__(self, history, cell_size=25):
        pygame.init()
        self.history = history
        self.cell_size = cell_size
        self.current_step = 0
        
        self.rows = history[0].height
        self.cols = history[0].width
        
        # Layout Layout
        self.grid_w = self.cols * cell_size
        self.grid_h = self.rows * cell_size
        self.sidebar_w = 200
        self.bottom_h = 80
        
        self.screen = pygame.display.set_mode((self.grid_w + self.sidebar_w, self.grid_h + self.bottom_h))
        
        # Initialize UI Components
        self.checkboxes = [
            Checkbox(self.grid_w + 20, 50, "Traversability", True),
            Checkbox(self.grid_w + 20, 80, "Victims", True),
            Checkbox(self.grid_w + 20, 110, "Agents", True),
            Checkbox(self.grid_w + 20, 140, "Confidence Map", False),
        ]
        
        self.slider_rect = pygame.Rect(50, self.grid_h + 40, self.grid_w - 100, 10)
        self.is_dragging_slider = False

    def draw_grid_layer(self, surface, matrix, color_mapping_func):
        """Generic layer drawer for performance"""
        for y in range(self.rows):
            for x in range(self.cols):
                val = matrix[y][x]
                color = color_mapping_func(val)
                if color: # Only draw if there's a color (alpha support)
                    rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if len(color) == 4: # Handle transparency
                        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        s.fill(color)
                        surface.blit(s, rect)
                    else:
                        pygame.draw.rect(surface, color, rect)

    def render(self):
        self.screen.fill(COLORS["ui_bg"])
        state = self.history[self.current_step]
        
        # 1. DRAW LAYERS
        # Base Floor/Walls
        if self.checkboxes[0].active:
            self.draw_grid_layer(self.screen, state.traversability.matrix, 
                                 lambda v: COLORS["wall"] if v == 1 else COLORS["floor"])
        
        # Victims
        if self.checkboxes[1].active:
            self.draw_grid_layer(self.screen, state.victims.matrix, 
                                 lambda v: COLORS["victim"] if v == 1 else None)
        
        # Agents
        if self.checkboxes[2].active:
            self.draw_grid_layer(self.screen, state.agents.matrix, 
                                 lambda v: COLORS["agent"] if v == 1 else None)

        # Confidence (Example of Heatmap/Alpha layer)
        if self.checkboxes[3].active:
            self.draw_grid_layer(self.screen, state.confidence.matrix, 
                                 lambda v: (255, 255, 0, int(v * 100)) if v > 0 else None)

        # 2. DRAW SIDEBAR
        pygame.draw.rect(self.screen, (20, 20, 25), (self.grid_w, 0, self.sidebar_w, self.grid_h))
        for cb in self.checkboxes:
            cb.draw(self.screen)

        # 3. DRAW SLIDER
        pygame.draw.rect(self.screen, (100, 100, 100), self.slider_rect) # Slider bar
        # Slider handle position
        progress = self.current_step / (len(self.history) - 1)
        handle_x = self.slider_rect.x + (progress * self.slider_rect.width)
        pygame.draw.circle(self.screen, COLORS["accent"], (int(handle_x), self.slider_rect.centery), 10)
        
        # Label
        font = pygame.font.SysFont("Arial", 14)
        ts_text = font.render(f"Step: {self.current_step}", True, COLORS["text"])
        self.screen.blit(ts_text, (self.slider_rect.x, self.slider_rect.y - 20))

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                # Checkbox events
                for cb in self.checkboxes:
                    cb.handle_event(event)
                
                # Slider events
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.slider_rect.collidepoint(event.pos):
                        self.is_dragging_slider = True
                if event.type == pygame.MOUSEBUTTONUP:
                    self.is_dragging_slider = False
                
                if self.is_dragging_slider and event.type == pygame.MOUSEMOTION:
                    # Update current step based on mouse X
                    rel_x = max(0, min(event.pos[0] - self.slider_rect.x, self.slider_rect.width))
                    self.current_step = int((rel_x / self.slider_rect.width) * (len(self.history) - 1))

            self.render()
            pygame.display.flip()
            clock.tick(60)

# Implementation inside main.py or bottom of visualization.py:
if __name__ == "__main__":
    sim = Simulator(30, 20)
    sim.generate_ground_truth(MapConfig(num_agents=1, num_victims=3))
    
    # Simple logic to make the agent move right for the demo
    from agent import Agent
    test_agent = Agent("Robot1", 2, 2, 30, 20)
    sim.add_agent(test_agent)
    
    history = sim.run(50)
    viz = Visualizer(history)
    viz.run()