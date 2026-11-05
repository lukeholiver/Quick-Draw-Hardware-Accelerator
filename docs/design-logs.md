Oct 29th
Finally starting this project. Setting up git environment and outlining needed steps.
Installed: venv, pytorch, numpy, matplotlib, serial
Downloaded Quick Draw "cats" class

Oct 30th
Choose all 10 classes: cat, sword, sailboat, flower, face, umbrella, eyeglasses, tree, boomerang, house.
Possible issues: cat and face might have issues in the neural net due to their similarity. Tree and flower as well. Will depend on training

Oct 31st
Wrote plot_samples.py to pick 5 random samples of each specified class and display them in a window.
Decided to remove tree and umbrella classes and replace them with spider and hat.
Umbrella, tree, and flower showed visual overlap; removals seek to prevent ambiguity

Nov 5th
Generate foward.py to run the model. ReLU passes through positive values and sets negative values to zero. Forward single runs a single image through the net. Forward batch runs 50 images through the net. The structure is 784 -> 128 -> 10. Script gives 16% accuracy before training with 5 images of each class (50 total)