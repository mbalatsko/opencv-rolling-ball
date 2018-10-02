# Rolling ball and sliding paraboloid background subtraction algorithms 

[![PyPI version](https://badge.fury.io/py/opencv-rolling-ball.svg)](https://badge.fury.io/py/opencv-rolling-ball)
[![Downloads](https://pepy.tech/badge/opencv-rolling-ball)](https://pepy.tech/project/pypylon-opencv-viewer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Fully Ported to Python from ImageJ's Background Subtractor.
Only works for 8-bit greyscale images currently.
Based on the concept of the rolling ball algorithm described
in Stanley Sternberg's article,
"Biomedical Image Processing", IEEE Computer, January 1983.
Imagine that the 2D grayscale image has a third (height) dimension by the image
value at every point in the image, creating a surface. A ball of given radius
is rolled over the bottom side of this surface; the hull of the volume
reachable by the ball is the background.
http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html

**This algorithms are perfect for microscope images, to distinguish particles
from background.**

## Installation

```bash
pip install opencv-rolling-ball
```

## Usage

```python
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball

img = cv2.imread(f'path/to/img.tif', 0)
img, background = subtract_background_rolling_ball(img, 30, light_background=True,
                                     use_paraboloid=False, do_presmooth=True)
```

## Example outputs

#### Input

![Input](https://raw.githubusercontent.com/mbalatsko/opencv-rolling-ball/master/outputs/example.png)

#### Subtracted background

![Background](https://raw.githubusercontent.com/mbalatsko/opencv-rolling-ball/master/outputs/example_bg.png)

#### Without background

![Without background](https://raw.githubusercontent.com/mbalatsko/opencv-rolling-ball/master/outputs/example_roll.png)


