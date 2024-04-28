import numpy as np
import imageio.v3 as iio
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

rng = np.random.default_rng()
Path = mpath.Path

class Camo_Worm:
    """
    Class representing a camouflage worm.
    """

    def __init__(self, x, y, r, theta, deviation_r, deviation_gamma, width, colour):
        """
        Initialize a Camo_Worm object.

        Parameters:
        - x, y: Coordinates of the center point of the worm.
        - r: Radius of the worm.
        - theta: Angle of the center line of the worm from the x-axis.
        - deviation_r: Radius of deviation of the mid control point from the center.
        - deviation_gamma: Angle of the line segment joining the center and mid control points from the center line.
        - width: Width of the worm.
        - colour: Colour of the worm.

        """
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.colour = colour
        p0 = [self.x - self.r * np.cos(self.theta), self.y - self.r * np.sin(self.theta)]
        p2 = [self.x + self.r * np.cos(self.theta), self.y + self.r * np.sin(self.theta)]
        p1 = [self.x + self.dr * np.cos(self.theta+self.dgamma), self.y + self.dr * np.sin(self.theta+self.dgamma)]
        self.bezier = mbezier.BezierSegment(np.array([p0, p1,p2]))

    def control_points(self):
        """
        Get the control points of the worm.

        Returns:
        - List of control points.
        """
        return self.bezier.control_points

    def path(self):
        """
        Get the path of the worm.

        Returns:
        - Matplotlib path.
        """
        return mpath.Path(self.control_points(), [Path.MOVETO, Path.CURVE3, Path.CURVE3])

    def patch(self):
        """
        Get the patch representing the worm.

        Returns:
        - Matplotlib patch.
        """
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.colour), lw=self.width, capstyle='round')

    def intermediate_points(self, intervals=None):
        """
        Get intermediate points of the worm.

        Parameters:
        - intervals: Number of intervals to divide the worm's path. Default is None.

        Returns:
        - Array of intermediate points.
        """
        if intervals is None:
            intervals = max(3, int(np.ceil(self.r/8)))
        return self.bezier.point_at_t(np.linspace(0,1,intervals))

    def approx_length(self):
        """
        Get an approximate length of the worm.

        Returns:
        - Approximate length of the worm.
        """
        intermediates = intermediate_points(self)
        eds = euclidean_distances(intermediates, intermediates)
        return np.sum(np.diag(eds, 1))

    def colour_at_t(self, t, image):
        """
        Get the colour of the worm at a given parameter t.

        Parameters:
        - t: Parameter value.
        - image: Image to sample colours from.

        Returns:
        - Array of colours.
        """
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))
        colours = [image[point[0],point[1]] for point in intermediates]
        return(np.array(colours)/255)


class Drawing:
    """
    Class for drawing images and worms.
    """

    def __init__(self, image):
        """
        Initialize a Drawing object.

        Parameters:
        - image: Image to be displayed.
        """
        self.fig, self.ax = plt.subplots()
        self.image = image
        self.im = self.ax.imshow(self.image, cmap='gray', origin='lower')

    def add_patches(self, patches):
        """
        Add patches to the drawing.

        Parameters:
        - patches: List of patches to be added.
        """
        try:
            for patch in patches:
                self.ax.add_patch(patch)
        except TypeError:
            self.ax.add_patch(patches)

    def add_dots(self, points, radius=4, **kwargs):
        """
        Add dots to the drawing.

        Parameters:
        - points: List of points to be represented as dots.
        - radius: Radius of the dots. Default is 4.
        - **kwargs: Additional keyword arguments for dot appearance.
        """
        try:
            for point in points:
                self.ax.add_patch(mpatches.Circle((point[0],point[1]), radius, **kwargs))
        except TypeError:
            self.ax.add_patch(mpatches.Circle((points[0],points[1]), radius, **kwargs))

    def add_worms(self, worms):
        """
        Add worms to the drawing.

        Parameters:
        - worms: List of worms to be added.
        """
        try:
            self.add_patches([w.patch() for w in worms])
        except TypeError:
            self.add_patches([worms.patch()])

    def show(self, save=None):
        """
        Display the drawing.

        Parameters:
        - save: File path to save the drawing. Default is None.
        """
        if save is not None:
            plt.savefig(save)
        plt.show()


def crop(image, mask):
    """
    Crop an image.

    Parameters:
    - image: Image to be cropped.
    - mask: Mask defining the cropping area.

    Returns:
    - Cropped image.
    """
    h, w = np.shape(image)
    return image[max(mask[0],0):min(mask[1],h), max(mask[2],0):min(mask[3],w)]

def prep_image(imdir, imname, mask):
    """
    Prepare an image for display.

    Parameters:
    - imdir: Directory containing the image.
    - imname: Name of the image file.
    - mask: Mask defining the cropping area.

    Returns:
    - Cropped and flipped image.
    """
    print("Image name (shape) (intensity max, min, mean, std)\n")
    image = np.flipud(crop(iio.imread(imdir+'/'+imname+".png"), mask))
    print("{} {} ({}, {}, {}, {})".format(imname, np.shape(image), np.max(image), np.min(image), round(np.mean(image),1), round(np.std(image),1)))
    plt.imshow(image, vmin=0, vmax=255, cmap='gray', origin='lower') # use vmin and vmax to stop imshow from scaling
    plt.show()
    return image

def random_worm(imshape, init_params):
    """
    Generate a random worm.

    Parameters:
    - imshape: Shape of the image.
    - init_params: Initial parameters for generating the worm.

    Returns:
    - Randomly generated Camo_Worm object.
    """
    (radius_std, deviation_std, width_theta) = init_params
    (ylim, xlim) = imshape
    midx = xlim * rng.random()
    midy = ylim * rng.random()
    r = radius_std * np.abs(rng.standard_normal())
    theta = rng.random() * np.pi
    dr = deviation_std * np.abs(rng.standard_normal())
    dgamma = rng.random() * np.pi
    colour = rng.random()
    width = width_theta * rng.standard_gamma(3)
    return Camo_Worm(midx, midy, r, theta, dr, dgamma, width, colour)

def initialise_clew(size, imshape, init_params):
    """
    Initialize a clew of worms.

    Parameters:
    - size: Number of worms in the clew.
    - imshape: Shape of the image.
    - init_params: Initial parameters for generating worms.

    Returns:
    - List of initialized Camo_Worm objects.
    """
    clew = []
    for i in range(size):
        clew.append(random_worm(imshape, init_params))
    return clew
