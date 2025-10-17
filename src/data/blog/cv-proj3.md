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

# Manual Stitching 
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


## Recover Homographies
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

## Warping the Images
Given the transformation matrix H, we can warp the image to the desired target space. Since the image could be resized to a larger scale, we want to impelement inverse warping (sampling colors from the original image to fill the pixels of the final canvas) as opposed to forward warping (directly projecting original image pixels onto new image space).

Transformed coordinates may be decimals, meaning we need to sample in between discrete pixels. The most straightforward approach is sampling the **nearest neighbor** pixel by rounding the coordinate. This may cause some artifacts and blockiness. A smoother version is **bilinear interpolation**, which computes the pixel value as a weighted average of the four neighboring pixels, with weights determined by their distances from the sampled point.

### Nearest Neighbor Sampling
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

### Bilinear Interpolation Sampling

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

### Comparison and Rectifying Rectangle

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

## Creating a Mosaic
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

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_vosskhead.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Big Head Vossk Space Station</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_vosskleg.jpg" alt="original mushroom"/>
    <figcaption class="text-center">Long Leg Vossk Space Station (bottom)</figcaption>
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
<figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj3/mosaic_night.webp" alt="original cat yawning"/>
    <figcaption class="text-center">Night Street</figcaption>
  </figure>
</div>

# Automatic Stitching

In the previous section, the correspondance points are manually chosen. In this section, we will use corner detection and feature matching to automatically find pairs of corrspondance points, thus achieving automatic stitching. In the following parts, we explore how to detect corresponding features with harris corner detection, ANMS, feature descriptor and RANSAC.

## Harris Corner Detection
A corner can be identified by observing a significant change in intensity when moving in both the x and y directions within a small window, unlike edges or flat regions, where movement in one or both directions results in little or no change.

  <figure style="width: 75%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/harris_corner.jpg" alt="harris corner"/>
    <figcaption class="text-center">what is a corner (picture from lecture slides)</figcaption>
  </figure>

Using ```corner_harris()``` from library ```skimage```, we detect the following corners on a landscape image.
  <figure style="width: 50%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/lake0_harris.webp" alt="harris corner"/>
    <figcaption class="text-center">detected harris corners</figcaption>
  </figure>
The Harris corner detector identifies potential feature points, then we used adaptive non-maximal suppression (ANMS) to find the most prominent corners.

## Adaptive Non-Maximal Supression (ANMS)
Adaptive Non-Maximal Suppression (ANMS) is a more dynamic and adaptive approach than traditional Non-Maximal Suppression (NMS). While NMS relies purely on corner strength, iteratively suppressing weaker corners near the strongest one (or those within overlapping regions) until a desired number of corners is obtained, ANMS takes spatial distribution into account. It preserves both weak corners in flat regions and strong corners in textured or edge-dense areas.

```bash
  min_r_dic = {}
  threshold = 0.9
  # for each harris corner
  for y, x in zip(harris_corner_coords[0], harris_corner_coords[1]):
      strength = h[y, x]
      # find all corners above threshold
      mask = h > threshold * strength
      stronger_ys, stronger_xs = np.nonzero(mask)
      matrix = np.vstack((stronger_ys, stronger_xs)).T
      other_matrix = np.array([y, x]).reshape(1, 2)
      # calculate distance to all corners above threshold
      distances = self.dist2(matrix, other_matrix)
      min_r_dic[(y, x)] = np.min(distances)
  # sort and take the top 50 corners
  anms_corners = sorted(min_r_dic.items(), key=lambda kv: kv[1], reverse=True)[:50]
```
For each corner candidate, ANMS computes the minimum distance to another corner whose strength exceeds a certain threshold fraction of its own. Corners with smaller such distances (similar to a local max) are shortlisted as the final set of sharp corners.


We can see the anms selected corners are much more prominent and more suitable to be correspondance points.

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3b/lake0_harris.webp" alt="original cat yawning"/>
    <figcaption class="text-center">before anms</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3b/lake0_anms.webp" alt="original mushroom"/>
    <figcaption class="text-center">after anms</figcaption>
  </figure>
</div>

## Feature Descriptor Extraction
After selecting feature points, the next step is to match them across the two overlapping images. To do this, we need a descriptor that can uniquely identify each corner. For simplicity, the descriptor is defined to be a 40×40 pixel patch centered at each corner and downsampled to 5×5. To make the descriptors robust to brightness and intensity variations, each patch is normalized to a range between 0 and 1. 

  <figure style="width: 50%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/patches.jpg" alt="harris corner"/>
    <figcaption class="text-center">example 4 patches</figcaption>
  </figure>
Note that these descriptors are rotation variant and scale variant. Since my images are without rotatoin and without change in zoom, these descriptors can still work well.

## Feature Matching
We use the Lowe ratio test to match feature descriptors. Instead of relying solely on raw pixel differences, the Lowe ratio measures how much better the best match is compared to the second-best match. The intuition is that correct matches should have significantly lower matching errors than incorrect ones.

$$
\text{Lowe Ratio} = \frac{Error_{\text{lowest}}}{Error_{\text{second-lowest}}}
$$

Each feature is matched to the corresponding feature with the best Lowe ratio in the other image. When this ratio is below a predefined threshold (0.65, as suggested by Brown et al.), the feature is excluded.
  <figure style="width: 100%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/matchings.webp" alt="harris corner"/>
    <figcaption class="text-center">matchings according to Lowe Ratio</figcaption>
  </figure>

  ## RANSAC (Random Sample Consensus)
 We observed that the previous results included a few incorrect matches. To eliminate these outliers, we implement Random Sample Consensus (RANSAC). The key intuition is that correct matches (inliers) should produce a consistent homography H with little error, whereas incorrect matches (outliers) would cause significant deviations in H. We randomly select four points to compute a candidate homography matrix. Points whose projections fall within a predefined error threshold are classified as correct matches. This process is repeated iteratively until a sufficient number of inliers is obtained. The error threshold is tricky because the ideal value varies across images from images

  ```bash
inliers = set() # correct matches
for _ in range(k):
    valid_matchings = [i for i, v in enumerate(self.matchings) if v != -1]
    selected_indices = random.sample(valid_matchings, 4)
    im0_pts = [self.anms0[i] for i in selected_indices]
    im1_pts = [self.anms1[self.matchings[i]] for i in selected_indices]
    H = self.computeH(im0_pts, im1_pts)
    for index in range(4):
        projected = np.dot(H, np.array([im1_pts[index][1], im1_pts[index][0], 1]))
        error = projected - np.array([im0_pts[index][1], im0_pts[index][0], 0])
        # if the projected point falls under an error threshold, it is considered as a correct match
        if np.linalg.norm(error) < e:
            inliers.add(selected_indices[index])
  ```
  <figure style="width: 100%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/matchings.webp" alt="harris corner"/>
    <figcaption class="text-center">matchings before RANSAC</figcaption>
  </figure>
  <figure style="width: 100%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj3b/new_matchings.webp" alt="harris corner"/>
    <figcaption class="text-center">matchings after RANSAC</figcaption>
  </figure>
  We can finally proceed to stitch the image with code from manual stitching with the correct matches. 

# Gallery

<div style="display: flex; justify-content: center;">
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj3b/auto_void_mosaic.jpg" alt="original mushroom"/>
    <figcaption class="text-center">automatic stitching</figcaption>
  </figure>
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj3b/manual_void.webp" alt="original cat yawning"/>
    <figcaption class="text-center">manual stitching</figcaption>
  </figure>
</div>
<!-- <div style="display: flex; gap: 1rem; justify-content: center;"> -->
Notice manual stitching has high quality, probably due to a higher number and a better selection of feature points.
  <figure  style="margin-top: 0.5rem">
    <img style="width: 75%;" src="/images/proj3b/auto_corre.jpg" alt="original mushroom"/>
    <figcaption class="text-center">auto correspondance of Void</figcaption>
  </figure>
<figure style="margin-bottom: 0.5rem">
<img style="width:75%" src="/images/proj3/void_corr.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">manual correspondance of Void</figcaption>
</figure>
<!-- </div> -->
  <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj3b/lake_mosaic.webp" alt="original cat yawning"/>
    <figcaption class="text-center">Lake Mosaic</figcaption>
  </figure>
  <div style="display: flex; justify-content: center;">
  <figure style="width: 75%;margin-top: 0.5rem">
    <img src="/images/proj3b/spiral_mosaic.webp" alt="original cat yawning"/>
    <figcaption class="text-center">Spiral Mosaic</figcaption>
  </figure>
</div>
  <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj3b/castle_mosaic.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">Castle Mosaic</figcaption>
  </figure>

Notice there are some artifacts in the stitched images. These pictures are taken in the game *Wuthering Waves*, whose camera rotation introduces slight translational motion. This can result in movements that cannot be fully modeled by a homography matrix.