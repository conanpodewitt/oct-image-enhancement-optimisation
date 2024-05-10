import numpy as np
import imageio.v3 as iio
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.bezier as mbezier
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

rng = np.random.default_rng()
Path = mpath.Path

class CamoWorm:
    """
    Class representing a camouflage worm.
    """

    def __init__(self, x, y, r, theta, deviation_r, deviation_gamma, width, color):
        """
        Initialize a Camo_Worm object.

        Parameters:
        - x, y: Coordinates of the center point of the worm.
        - r: Radius of the worm.
        - theta: Angle of the center line of the worm from the x-axis.
        - deviation_r: Radius of deviation of the mid control point from the center.
        - deviation_gamma: Angle of the line segment joining the center and mid control points from the center line.
        - width: Width of the worm.
        - color: color of the worm.

        """
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta
        self.dr = deviation_r
        self.dgamma = deviation_gamma
        self.width = width
        self.color = color
        
        p0 = (self.x, self.y)
        p1 = (self.x + self.dr * np.cos(self.theta + self.dgamma), 
              self.y + self.dr * np.sin(self.theta + self.dgamma))
        p2 = (self.x + self.r * np.cos(self.theta), 
              self.y + self.r * np.sin(self.theta))

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
        return mpatches.PathPatch(self.path(), fc='None', ec=str(self.color), lw=self.width, capstyle='round')

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

    def color_at_t(self, t, image):
        """
        Get the color of the worm at a given parameter t.

        Parameters:
        - t: Parameter value.
        - image: Image to sample colors from.

        Returns:
        - Array of colors.
        """
        intermediates = np.int64(np.round(np.array(self.bezier.point_at_t(t)).reshape(-1,2)))
        colors = [image[point[0],point[1]] for point in intermediates]
        return(np.array(colors)/255)
    
    def bezier_points(self, num_points=100):
        """
        Generate points along the Bézier curve.

        Parameters:
        - num_points: Number of points to generate along the curve.

        Returns:
        - Arrays of x and y coordinates of the generated points.
        """
        t_values = np.linspace(0, 1, num_points)
        points = [self.bezier.point_at_t(t) for t in t_values]
        x_points, y_points = zip(*points)
        return np.array(x_points, dtype=int), np.array(y_points, dtype=int)

    def grow(self, growth_amount=0.5, new_width=3):
        """
        Increase the radius of the worm by the specified growth amount 
        and set its width to the specified new width.
        
        Parameters:
        - growth_amount: Amount by which the radius should increase.
        - new_width: New width of the worm.
        """
        self.r += growth_amount
        self.width = new_width

    def move(self, image_shape):
        """
        Move the worm to a random position within the image shape.

        Parameters:
        - image_shape: Shape of the image.
        """
        self.x = np.random.randint(0, image_shape[1])  # random x-coordinate
        self.y = np.random.randint(0, image_shape[0])  # random y-coordinate
        
    def adapt_color(self, image):
        """
        Adapt the color of the worm based on the color of the pixel at its position in the image.

        Parameters:
        - image: Image to sample colors from.
        """
        color = image[int(self.y), int(self.x)] / 255  # get color of the pixel and normalize
        self.color = color

    def adapt_curvature(self, dgamma_range=(0, np.pi/10), theta_range=(-np.pi/10, np.pi/10), dr_range=(-50, 50)):
        """
        Adjust the curvature of the worm based on specified ranges for each parameter.

        Parameters:
            - dgamma_range: Range for random selection of dgamma.
            - theta_range: Range for random selection of theta.
            - dr_range: Range for random selection of dr.
        """
        self.dgamma = np.random.uniform(*dgamma_range)
        self.theta = np.random.uniform(*theta_range)
        self.dr = np.random.uniform(*dr_range)
        
        # Recalculate the control points for the Bezier curve
        p0 = [self.x - self.r * np.cos(self.theta), self.y - self.r * np.sin(self.theta)]
        p2 = [self.x + self.r * np.cos(self.theta), self.y + self.r * np.sin(self.theta)]
        p1 = [self.x + self.dr * np.cos(self.theta + self.dgamma), self.y + self.dr * np.sin(self.theta + self.dgamma)]
        self.bezier = mbezier.BezierSegment(np.array([p0, p1, p2]))

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