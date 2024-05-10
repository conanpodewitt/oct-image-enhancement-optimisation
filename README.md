# oct-image-enhancement-optimisation
### CITS4404 Assignment 2 @ The University of Western Australia

Dennis Gunadi (**22374535**)\
Conan Dewitt (**22877792**)\
Alian Haidar (**22900426**)\
Lili Liu (**23212326**)

### Camo Worm Optimization for OCT Image Enhancement

This Jupyter notebook presents a project aimed at enhancing OCT (Optical Coherence Tomography) images using camo worms and optimizing the enhancement process through a genetic algorithm.

### Summary of Contents:

1. **Initialization and Constants:**
   - Constants and initialization code for setting up the environment, including image directory, image name, mask, population size, and population parameters.

2. **Image Preprocessing and Worm Creation:**
   - Functions for cropping images and preparing images for processing. Also, includes code for creating random worms with random attributes.

3. **Fitness Evaluation and Genetic Algorithm:**
   - Implementation of the genetic algorithm for optimizing the population of worms within the image. Includes functions for evaluating fitness, selecting fittest worms, performing crossover, and mating worms.

4. **Visualization and Result Analysis:**
   - Functions for visualizing the population of worms and analyzing the results of the genetic algorithm optimization process.

5. **Genetic Algorithm Execution:**
   - Execution of the genetic algorithm to optimize the population of worms for OCT image enhancement. Includes visualization of the initial population and the best generation obtained through optimization.

The genetic algorithm iteratively improves the distribution of worms within the OCT images by evaluating their fitness, selecting the fittest ones, and generating new generations through crossover and mutation. The process aims to enhance image features while ensuring color uniformity and minimizing deviation from the target intensity.