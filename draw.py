import os

import pygame
from PIL import Image

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('Arial', 15)
text = font.render('Draw a single digit number and I will recognize it:', True, (255, 255, 255))


def main():

    # Set the window size
    WIDTH, HEIGHT = 280, 280

    # Create the window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Set the title
    pygame.display.set_caption('Drawing App')

    # Initialize some variables
    drawing = False  # Keep track of whether we are drawing or not
    last_pos = (0, 0)  # Keep track of the previous mouse position

    # Run the game loop
    running = True
    while running:
        screen.blit(text, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                os.remove("Playerdrawing.jpg")
                pygame.image.save(screen, "Playerdrawing.jpg")
                with Image.open("Playerdrawing.jpg") as im:
                    # Resize the image
                    im_resized = im.resize((28, 28))
                    # Save the resized image
                    im_resized.save('Playerdrawing.jpg')
                im = Image.open('Playerdrawing.jpg')
                im_gray = im.convert('L')
                im_gray.save('Playerdrawinggray.jpg')
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start drawing
                drawing = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                # Stop drawing
                if drawing:
                    os.remove("Playerdrawing.jpg")
                    pygame.image.save(screen, "Playerdrawing.jpg")
                    with Image.open("Playerdrawing.jpg") as im:
                        # Resize the image
                        im_resized = im.resize((28, 28))
                        # Save the resized image
                        im_resized.save('Playerdrawing.jpg')
                    im = Image.open('Playerdrawing.jpg')
                    im_gray = im.convert('L')
                    im_gray.save('Playerdrawinggray.jpg')
                    running = False
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                # If we are drawing, draw a line from the previous mouse position to the current mouse position
                if drawing:
                    pygame.draw.line(screen, (255, 255, 255), last_pos, event.pos, 28)
                    last_pos = event.pos
        pygame.display.update()

    # Quit Pygame
    pygame.quit()
