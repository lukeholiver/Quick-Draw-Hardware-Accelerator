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
foward.py to run the model. ReLU passes through positive values and sets negative values to zero. Forward single runs a single image through the net. Forward batch runs 50 images through the net. The structure is 784 -> 128 -> 10. Script gives 16% accuracy before training with 5 images of each class (50 total)

train.py to train the mode. 87.89% accuracy with 100% match compared with PyTorch over 20,000 test images. 
top confusions (true -> predicted):
  cat          -> spider       119 images
  spider       -> cat          109 images
  flower       -> spider       82 images
  sailboat     -> hat          74 images
  spider       -> flower       74 images

I prediced that cat and face would be confused the most, but spider and cat had the most confusion. I'd assume wiskers on a cat could get confused for a spider's legs.

Cheap experiment: run model with 40,000 images per class instead of 10,000:
Results: no accuracy gains -> bottleneck is model structure not ammount of data

Aiming for >=85% test accuracy

Output from quantize.py:

accuracy on test set:
  float  87.89%
  int8   87.93%   (change +0.04%)
  int8 and float agree on 99.59% of images

accumulator widths (signed bits):
  layer 1: observed 21, worst case 26
  layer 2: observed 17, worst case 23
  requant product acc1*M0: observed max needs 35 bits
  hidden activations clipped at 255: 0.000% of values