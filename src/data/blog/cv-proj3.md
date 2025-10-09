---
title: "CV Project 3: Image Warping and Mosaicing"
author: Helena Su
pubDatetime: 2025-10-08
slug: cv-proj3
featured: false
draft: false
tags:
  - Computer Vision
description:
  perspective transforms, warping
---


## Pictures
I took some screenshots by rotating camera view from my favorite space game ten years ago——*Galaxy on Fire 2*. Since the game isn’t rendered from a true first-person perspective (the camera is offset from the spaceship’s center of rotation), rotating the view causes small translational shifts as well. To minimize distortions that can’t be captured by a simple perspective transform, I drove around my ship and took pictures that were large and distant enough.
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/void2.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Void Planet</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/void3.jpg" alt="original mushroom"/>
    <figcaption class="text-center">Void Space Station</figcaption>
  </figure>
</div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/vossktop.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Vossk Space Station (top)</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/vosskbottom.jpg" alt="original mushroom"/>
    <figcaption class="text-center">Vossk Space Station (bottom)</figcaption>
  </figure>
</div>


# Recover Homographies
Perspective transform in 2D pictures can be captured by a 3x3 matrix with 8 DoF (scale is fixed to be 1). 

$$
\begin{bmatrix}
wx' \\[4pt]
wy' \\[4pt]
w'
\end{bmatrix}
=
\begin{bmatrix}
a & b & c \\[4pt]
d & e & f \\[4pt]
g & h & 1
\end{bmatrix}
\begin{bmatrix}
x \\[4pt]
y \\[4pt]
1
\end{bmatrix}

$$
Since there are eight unknowns, we need at least eight linear equations to solve for the homography. Each pair of corresponding points — where x0, y0 in image 0 maps to x1, y1 in image 1 — provides two equations. This means a minimum of four pairs of correspondences is required to compute the transformation. To account for noise and distortion, we typically use more than four pairs (I selected around eight) and solve for the best-fit homography matrix using least squares. After some algebra, we arrive at the following code to recover matrix H.
```bash
def computeH(target, other):
  A = []
  b = []
  im1_pts = other
  im2_pts = target
  for i in range(len(im1_pts)):
    x, y = im1_pts[i][0], im1_pts[i][1]
    u, v = im2_pts[i][0], im2_pts[i][1]
    A.append([-x, -y, -1, 0, 0, 0, x*u, y*u])
    A.append([0, 0, 0, -x, -y, -1, x*v, y*v])
    b.append(-u)
    b.append(-v)
  h, residuals, rank, s = np.linalg.lstsq(A, b)
  h = np.append(h, 1)
  H = h.reshape(3,3)
  return H
```
In this part of the project, we manually pick out the points of correspondance as displayed below:
<figure style="margin-bottom: 0.5rem">
<img style="width:100%" src="/images/proj3/void_corr.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">correspondance of Void</figcaption>
</figure>

<figure style="margin-bottom: 0.5rem">
<img style="width:100%" src="/images/proj3/vossk_corr.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">correspondance of Vossk</figcaption>
</figure>

# Warping the Images
Given the transformation matrix H, we can warp the image to the desired target space. Since the image could be resized to a larger scale, we want to impelement inverse warping (sampling colors from the original image to fill the pixels of the final canvas) as opposed to forward warping (directly projecting original image pixels onto new image space).

Transformed coordinates may be decimals, meaning we need to sample in between discrete pixels. The most straightforward approach is sampling the **nearest neighbor** pixel by rounding the coordinate. This may cause some artifacts and blockiness. A smoother version is **bilinear interpolation**, which computes the pixel value as a weighted average of the four neighboring pixels, with weights determined by their distances from the sampled point.

## Nearest Neighbor Sampling
Nearest neighbor sampling work by rounding each inverse warped coordinate to the nearest integer, then paint the canvas using only those coordinates that fall within the original image’s width and height.

```bash
# for each pixel
for i in range(warped_height):
  for j in range(warped_width):
    # apply an offset so warped image starts at 0,0
    warpped_x = (j + top_left[0])
    warpped_y = (i + top_left[1])
    # find where this pixel came from by inverse warping
    original_coord = round(inv_H @ np.array([warpped_x, warpped_y, 1]))
    original_coord /= original_coord[2]
    # masking and checking bounds
    if 0 <= original_coord[0] < w and 0 <= original_coord[1] < h:
    # paint
      out[i, j] = im[int(original_coord[1]), int(original_coord[0])]
```
A vectorized version so rounding is done on the matrix level instead of individual pixel level:

```bash
min_x = min(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
min_y = min(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
x_normalized = (np.arange(warped_width) + min_x)
y_normalized = (np.arange(warped_height) + min_y)
all_x = np.tile(x_normalized, (len(y_normalized), 1))
all_y = np.tile(y_normalized, (len(x_normalized), 1)).T
ones = np.ones_like(all_x)
coords = np.stack([all_x, all_y, ones]).reshape(3, -1)  # (3, n)
inv_H = np.linalg.inv(H)
original_coords = inv_H @ coords
original_coords /= original_coords[2, :]  
x_original = original_coords[0, :].reshape(warped_height, warped_width)
y_original = original_coords[1, :].reshape(warped_height, warped_width)
x_int = np.round(x_original).astype(int)
y_int = np.round(y_original).astype(int)
mask = ((x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h))
out = np.zeros((warped_height, warped_width, 3), dtype=im.dtype)
out[mask] = im[y_int[mask], x_int[mask]]
```

## Bilinear Interpolation Sampling

<figure style="margin-bottom: 0.5rem">
<img style="width:100%" src="/images/proj3/bilinear_interpolation.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">bilinear interpolation (taken from lecture slides)</figcaption>
</figure>
We take the four neighboring pixels and calculate their weighted average based on how close they are to each of the four pixels. A vectorized bilinear interpolation is shown below.

```bash
  partial_x = x_original - x0_int
  partial_y = y_original - y0_int
  partial_x = partial_x[..., np.newaxis]
  partial_y = partial_y[..., np.newaxis]
  topleft = im[y0_int, x0_int]
  topright = im[y0_int, x1_int]
  bottomleft = im[y1_int, x0_int]
  bottomright = im[y1_int, x1_int]
  out = ((1 - partial_x) * (1 - partial_y) * topleft +
      partial_x * (1 - partial_y) * topright +
      (1 - partial_x) * partial_y * bottomleft +
      partial_x * partial_y * bottomright)
```

## Comparison and Rectifying Rectangle

To verify our warping and H matrix, let's restore a rectangular laptop.
<figure style="margin-bottom: 1rem">
<img style="width:50%" src="/images/proj3/laptop.jpg" alt="cat yawning on musrhoom"/>

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/rectificationnearest.webp" alt="original cat yawning"/>
    <figcaption class="text-center">rectified by nearest neighbor</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/rectificationbilinear.webp" alt="original mushroom"/>
    <figcaption class="text-center">rectified by bilinear interpolation</figcaption>
  </figure>
</div>

We can examine the results in detail to compare the two sampling methods. Nearest-neighbor sampling produces a blocky, pixelated texture, while bilinear interpolation yields smoother and more continuous edges. However, there’s a trade-off where smoother, higher-quality interpolation requires more computation time.

<div style="display: flex; justify-content: center;">
  <figure style="width: 75%;margin-bottom: 0.5rem">
    <img src="/images/proj3/nearest.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">nearest neighbor sampling</figcaption>
  </figure>
</div>
<div style="display: flex; justify-content: center;">
  <figure style="width: 75%; margin-top: 0rem; padding-top 0rem">
    <img src="/images/proj3/bilinear.jpg" alt="original mushroom"/>
    <figcaption class="text-center">bilinear interpolation sampling</figcaption>
  </figure>
</div>

<figure style="margin-bottom: 1rem">
<img style="width:50%" src="/images/proj3/squares.jpg" alt="cat yawning on musrhoom"/>

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/squaresnearest.webp" alt="original cat yawning"/>
    <figcaption class="text-center">rectified by nearest neighbor</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/squaresbilinear.webp" alt="original mushroom"/>
    <figcaption class="text-center">rectified by bilinear interpolation</figcaption>
  </figure>
</div>

# Creating a Mosaic
We can stitch together the images now that they are in the same perspective. To reduce edge artifact, I used a fall off mask where weights gradually decrease from the center to the edge. To make sure the final weight add up to one, I use the following formula.
$$
out = \frac{w_1 in_1 + w_2 in_2}{w_1 + w_2}
$$
Intereseting to note, the game camera apply heavy distortion at the edge of the view, which cause drastic perspective transform.

  <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_void23.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Void Mothership and Planet</figcaption>
  </figure>
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/void2.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Void Planet</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/void3.jpg" alt="original mushroom"/>
    <figcaption class="text-center">Void Space Station</figcaption>
  </figure>
</div>
<div style="display: flex; justify-content: center;">
  <figure style="width: 70%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_vosskhead.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Big Head Vossk Space Station</figcaption>
  </figure>
  <figure style="width: 70%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_vosskleg.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Long Leg Vossk Space Station</figcaption>
  </figure>
</div>
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/vossktop.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Vossk Space Station (top)</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/vosskbottom.jpg" alt="original mushroom"/>
    <figcaption class="text-center">Vossk Space Station (bottom)</figcaption>
  </figure>
</div>
<div style="display: flex; justify-content: center;">
<figure style="width: 70%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_night.webp" alt="original cat yawning"/>
    <figcaption class="text-center">Night Street</figcaption>
  </figure>
</div>
