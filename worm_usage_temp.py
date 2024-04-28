import numpy as np
import worm as wrm

IMAGE_DIR = 'example_temp/images'
IMAGE_NAME='original'
MASK = [320, 560, 160, 880]  #! ymin ymax xmin xmax


# Example of cropping and preparing an image
image = wrm.prep_image(IMAGE_DIR, IMAGE_NAME, MASK)

# Example of using the Camo_Worm class
worm = wrm.Camo_Worm(200, 100, 50, np.pi/6, 70, np.pi/3, 10, 0.8)

# Example of using the Drawing class
drawing = wrm.Drawing(image)
drawing.add_worms(worm)
drawing.show(save='outputs/bezier.png')

# Example of initializing a random clew of worms
clew = wrm.initialise_clew(40, image.shape, (40, 30, 1))
drawing = wrm.Drawing(image)
drawing.add_worms(clew)
drawing.show()
