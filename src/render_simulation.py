import sys

import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import HTML
from flocking import WIDTH, direction
import numpy as np
import pygame
import time


def render_text(text, ax):
    return ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left")

def make_animation(states, interval_ms=60, N_frames=50):
    N = len(states)
    N_frames = min(N, N_frames)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = render_image(states[0], ax)
    text = render_text("", ax)
    
    def update(_frame):
        f = min((_frame*N)//N_frames, N-1)
        im.set_data(states[f])
        # text.set_text(f"gen: {_frame*(N//N_frames)}")
        return (im, text)

    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True, frames=N_frames)
    plt.close()
    return HTML(ani.to_jshtml(default_mode="once", fps=15))

def render_image(image, ax, title=None):
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax.imshow(image, cmap="binary", interpolation="nearest", vmin=0, vmax=1)

  
# Rendering
def get_dims(boids):
    if len(boids.shape) == 2:
        boids = boids[None,:]
    width = max(WIDTH, (boids[...,0].min(1) - boids[...,0].max(1)).max())
    height = max(WIDTH, (boids[...,1].min(1) - boids[...,1].max(1)).max())
    return [width, height]

def _draw_boids(boids, screen):
    # Use draws the boids on a given screen
    # Includes all the drawing code
    BOID_RADIUS = 5
    screen.fill((255,255,255))
    c = [(125,125,125)] * boids.shape[0]
    
    # center = np.mean(boids[:,:2], axis=0) - np.array(get_dims(boids)) / 2
    for i, boid in enumerate(boids):
        # boid = (float(boid[0]) - center[0], float(boid[1]) - center[1], float(boid[2]))
        pygame.draw.circle(screen, c[i], boid[:2], BOID_RADIUS)

        end_line = boid[:2] + direction(boid[2]) * 20
        pygame.draw.line(screen, c[i], boid[:2], end_line, 2)

def draw_boids(boids, screen=None):
    # Returns the image assiciated with the given boids
    need_quit = False
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode(get_dims(boids))
        need_quit = True
    pygame.display.flip()
    _draw_boids(boids, screen)
    im = pygame.surfarray.pixels3d(screen).transpose(1,0,2).copy()
    if need_quit:
        pygame.quit()
    return im

def plot_boids(boids, ax=None, savepath=None, show=False):
    # Plot the image of the boids on a plt axis
    if ax is None:
        ax = plt.gca()
    im = draw_boids(boids)
    render_image(im, ax)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
        
def draw_simulation(history, direct=True, fps=12):
  # Returns a list of images corresponding to each boids positions in the history
  states = []
  pygame.init()
  if direct:
      clock = pygame.time.Clock()
  screen = pygame.display.set_mode(get_dims(history))
  for boids in history:
      if direct: 
          clock.tick(fps)
      image = draw_boids(boids, screen)
      states.append(image)
      if direct:
          for e in pygame.event.get():
              if e.type == pygame.QUIT:
                  return
          pygame.display.update()
  pygame.quit()
  return states

def plot_simulation(history, direct=True, fps=12, interval_ms=60, N_frames=50):
    video = draw_simulation(history, direct, fps)
    if not direct:
        return make_animation(video, interval_ms, N_frames)
