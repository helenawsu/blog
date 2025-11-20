---
title: "CV Project 4: Neural Radiance Field"
author: Helena Su
pubDatetime: 2025-11-17
slug: cv-proj4
featured: false
draft: false
tags:
  - Computer Vision
description:
  perspective transforms, warping
---

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/final_final_nerf_render_test_15k_23-1.gif" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">Keys of the Void</figcaption>
  </figure>
  </div>

# Overview
In this project, we build and train a nerf from sratch with custom dataset. The process is divided into three parts: camera calibration and image capture, 2d nerf, and 3d nerf.

<iframe
  src="/plotly/volrend_trisurf_step0-5.html"
  style="width:100%;height:600px;border:none;"
  loading="lazy"
  title="Volrend trisurf (step 0.5)"
></iframe>

# Setup

## Camera Calibration 

Camera intrinsics is needed to calculate camera extrinsics, which projects screen coordinates to world coordinate. Intrinsics matrix (K matrix) contains focal length, image center and scale. To achieve this, 30 pictures of 6 aruco markers with various distance and angles are taken with an iPhone 12. Here are some examples of the captured images:

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 25%;margin-top: 0.5rem">
    <img src="/images/proj4/calibration/IMG_6931.JPG" alt="calibration example 1"/>
    <figcaption class="text-center">calibration example 1</figcaption>
  </figure>
  
  <figure style="width: 25%; margin-top: 0.5rem">
    <img src="/images/proj4/calibration/IMG_6957.JPG" alt="calibration example 2"/>
    <figcaption class="text-center">calibration example 2</figcaption>
  </figure>
</div>


The top left corner of the top right aruco marker is defined as world origin $(0, 0, 0)$. Using `cv2.aruco.detectMarkers` screen coordinate of each corner of each marker is extracted. The world coordinates and their corresponding screen coordinates across all images are fed into `cv2.calibrateCamera` that returns the reprojection error, camera matrix, and distortion coefficients. The iPhone 12 camera intrinsics are as follows, with a reprojection error of $1.27$:
$$
K = \begin{bmatrix}
591.17 & 0.0 & 394.55 \\
0.0 & 566.64 & 307.32 \\
0.0 & 0.0 & 1.0
\end{bmatrix}, 
\text{dist} = [ 0.29, -0.93,  0.012,  0.001,  0.10]
$$

 <br>

## Image Capture

This process involves placing my desired object—Keys of the Void from Honkai Impact 3rd—next to some aruco markers and take 46 pictures from multiple angle. Having six aruco markers spread around the object allows allows for a greater variety of shooting angles. The grid board is used to determine world coordinate of aruco markers and center of rotation axis for later test pose generation.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/gun/IMG_6853.JPG" alt="calibration example 1"/>
    <figcaption class="text-center">calibration example 1</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj4/gun/IMG_6891.JPG" alt="calibration example 2"/>
    <figcaption class="text-center">calibration example 2</figcaption>
  </figure>
</div>

The camera-to-world matrix for each image's camera extrinsics is then calculated. The calculation use `cv2.aruco.detectMarkers` again to detect screen coordinates of aruco markers and feed them into `cv2.solvePnP` to find the rotation and translation vector of camera in each image. It is used to build the c2w matrix. R matrix is converted from rvec with `cv2.Rodrigues`.

$$
\mathbf{T}_{cw} = \left[ \mathbf{R} | \mathbf{t} \right] = \begin{pmatrix}
r_{11} & r_{12} & r_{13} & t_1 \\
r_{21} & r_{22} & r_{23} & t_2 \\
r_{31} & r_{32} & r_{33} & t_3
\end{pmatrix}
$$ 

Viser visualizes the camera extrinsics in 3d space.
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/gun/viser_cloud0.jpg" alt="Keys of the Void example 1"/>
    <figcaption class="text-center">viser cloud example 1</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj4/gun/viser_cloud1.jpg" alt="Keys of the Void example 2"/>
    <figcaption class="text-center">viser cloud example 2</figcaption>
  </figure>
</div>

To minimize background noises and focus training on the object itself, I used ViT-H `Segment Anything Model` (SAM) to mask the background black. After undistorting the images with `cv2.undistort`, focal, images and their corresponding c2ws are packed into .npz file.

# 2D Image Neural Field

As a warm up to nerf, a 2D neural MLP field is build instead with inputs screen coordinate $(u, v)$ and output $(r, g, b)$. The model has the following architecture with a learning rate of 0.01, batch size of 10K, number of layers of 3, channel width of 256.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/image.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">2D Neural Field Architecture</figcaption>
  </figure>
  </div>

Since the neural network can only learn low frequency trend with raw 2d screen coordinate, sinusoidal positional encoding (PE) is applied to augment dimensionality, which allows the network to learn high frequency detail.
$$
\text{PE}(x) = \left\{ x, \sin(2^0\pi x), \cos(2^0\pi x), \sin(2^1\pi x), \cos(2^1\pi x), \dots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x) \right\}
$$
This comparison of without and with positional encoding explains its benefit. 
<div style="display: flex; gap: 1rem; justify-content: center; margin-bottom: 0rem">

  <figure style="width: 50%;margin-top: 0.5rem;  margin-bottom: 0rem">
    <img src="/images/proj4/twodnerf/pe.jpg" alt="comparison of pe"/>
    <figcaption class="text-center">image taken from cs180 slides</figcaption>
  </figure>
  </div>

Some training progressions:
<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_500.png" alt="fox 500"/>
    <figcaption class="text-center">fox 500 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_1000.png" alt="fox 500"/>
    <figcaption class="text-center">fox 1000 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_1500.png" alt="fox 500"/>
    <figcaption class="text-center">fox 1500 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_2000.png" alt="fox 500"/>
    <figcaption class="text-center">fox 2000 epochs</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/twodnerf/sunset_500.png" alt="sunset 500"/>
    <figcaption class="text-center">sunset 500 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/twodnerf/sunset_1000.png" alt="sunset 500"/>
    <figcaption class="text-center">sunset 1000 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/twodnerf/sunset_1500.png" alt="sunset 500"/>
    <figcaption class="text-center">sunset 1500 epochs</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/proj4/twodnerf/sunset_2000.png" alt="sunset 500"/>
    <figcaption class="text-center">sunset 2000 epochs</figcaption>
  </figure>
  </div>

  Observe how channel width and dimension of PE affects the quality of rendered image.
    <div style=" display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 30%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_3000_L10_W256.png" alt="sunset 500"/>
    <figcaption class="text-center">L10 W256</figcaption>
  </figure>
    <figure style="width: 30%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_3000_L10_W512.png" alt="sunset 500"/>
    <figcaption class="text-center">L10 W512</figcaption>
  </figure>
    <figure style="width: 30%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_3000_L20_W256.png" alt="sunset 500"/>
    <figcaption class="text-center">L20 W256</figcaption>
  </figure>
    <figure style="width: 30%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/twodnerf/fox_3000_L20_W512.png" alt="sunset 500"/>
    <figcaption class="text-center">L20 W512</figcaption>
  </figure>

  </div>
  <div style="display: flex; gap: 1rem; justify-content: center; margin-bottom: 0rem">

  <figure style="width: 35%;margin-top: 0.5rem;  margin-bottom: 0rem">
    <img src="/images/proj4/twodnerf/psnr.png" alt="train psnr vs spoch"/>
    <figcaption class="text-center">progression of psnr vs epoch</figcaption>
  </figure>
  </div>

# 3D Neural Radiance Field

Our training dataset contain a list of images and their corresponding camera-to-world (c2w) transformation matrices. We need to transform pixel data, screen coordinate $(u, v)$ and color, to ray origin and ray direction. Ray origin is the world coordinate of camera of each image, which is c2w matrices with homogenous dimension dropped. Ray direction is the normalized (pixel_world_coordinate - camera_world_coordinate) vector. We then batch pairs of ray origin and ray direction into the model. The model's architecture is as follows. 

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 100%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/arch.png" alt="3D Neural Field Architecture"/>
    <figcaption class="text-center">3D nerf structure</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/sample_ray_segment.jpg" alt="ray sampling"/>
    <figcaption class="text-center">example of ray sampling</figcaption>
  </figure>
  </div>

## Volumetric Rendering
Nerf learns the density and rgb at each continous point in the space, as oppose to explicit surface representation. This volumetric representation means that to render the final color at one position at one view direction, we need to integrate all the points on that ray (clipped by a bounding box) using alpha blending. For practicality, each ray is divided into many small segments to approximate the integral. This means in addition to sampling camera rays, we also need to segment each ray into many samples. The dimension of training input is thus (batch size, sample size, 3).

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 25%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/ray_segment.jpg" alt="ray segments"/>
    <figcaption class="text-center">ray segments</figcaption>
  </figure>
  </div>

The final color can be seen as the sum of color contribution from each ray segment $i$. The color of each segment is then calculated as transmitance (how much light is left) times opacity times raw rgb color.
$$
C = \sum_{j=near}^{j=far} T_i\cdot \alpha_i \cdot rgb_{raw}
$$

`torch.cumprod` is used to efficiently calculate transmitance T of each segment. The final weight (opacity) is calculated as follows. 
$$
T_i =  \prod_{j=near}^{j=i} (1 - \alpha_j)
$$
$$
\alpha_i =   (1 - e^{-\sigma_i \cdot \delta_i})
$$

As a small note: usually step size is disturbed so that the model learns continously instead of fixed grid points. However, a fixed step size in the volume rendering equation is sufficient to approximate the color accurately.

## Results

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/lego_render.gif" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">Lego Novel View</figcaption>
  </figure>
  </div>


  <div style=" display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 19%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/single_rendered_frame_10.png" alt="sunset 500"/>
    <figcaption class="text-center">10 iterations</figcaption>
  </figure>
    <figure style="width: 19%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/single_rendered_frame_100.png" alt="sunset 500"/>
    <figcaption class="text-center">100 iterations</figcaption>
  </figure>
    <figure style="width: 19%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/single_rendered_frame_500.png" alt="sunset 500"/>
    <figcaption class="text-center">500 iterations</figcaption>
  </figure>
    <figure style="width: 19%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/single_rendered_frame_1200.png" alt="sunset 500"/>
    <figcaption class="text-center">1200 iterations</figcaption>
  </figure>
  <figure style="width: 19%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/single_rendered_frame_2000.png" alt="sunset 500"/>
    <figcaption class="text-center">2000 iterations</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 40%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/lego/lego_psnr.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">lego validation psnr</figcaption>
  </figure>
  </div>

# Custom Data 

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/gun/gun_viser.jpg" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">3d visualizatoin of train and test pose</figcaption>
  </figure>
  </div>
<!-- <iframe
  src="/plotly/final_density_cloud.html"
  style="width:100%;height:600px;border:none;"
  loading="lazy"
  title="Volrend trisurf (step 0.5)"
></iframe> -->

## Optimization
In comparison to the Lego dataset, there are a few issues in the my dataset that makes trianing much slower. First, the majority of the image is background but the object should be focused. Second, my object is long on one end. This means my bounding box and loose and naive uniform step size sampling in ray will most likely hit empty space. To address these issues, two improvements are made in the sampling methods.

### 1. priotize object pixels

My train images are preprocessed with SAM to segment the object and mask the background black. This allows easy identification of valid pixels (those aren't pure black). When sampling image pixels, object pixels have a higher probability of being sampled. During training, I set the object ratio to be 0.9. If object ratio is too high, floats start to appear since the model learn too little about background.

### 2. prioritize high density ray segments
As opposed to sample uniformly along ray, points with higher density are more likely to be picked. This avoids wasting samples points on blank space.

To implement this optimization, two models are being trained at the same time. One is a coarse model with 64 number of samples per ray, and the other is the actual fine model with 256+64 samples per ray. Every epoch, the output of the coarse model is used to calculate final contribution of each ray segment with volrend. The calculated weights is then passed into a new `sample_along_rays_priority` method .

```bash
...
# in train loop
sigmas_coarse, rgbs_coarse = model_coarse(segment_world_coord_coarse, view_dir) 
outputs_coarse, segmented_weights_raw = volrend(sigmas_coarse, rgbs_coarse, None, dataset.B, t=t_coarse)
...
def sample_along_rays_priority(self, segmented_weights_raw, ro, rd, t_coarse, n_samples=256, near = 0.0, far = 100.0):
  segmented_weights = segmented_weights_raw.squeeze(-1).detach() + 1e-5
  bin_indices = torch.multinomial(segmented_weights.squeeze(-1), num_samples=n_samples, replacement=True) # Nx256
  step_coarse = (self.far-self.near) / self.num_samples_coarse
  t_far_edge = t_coarse[..., -1:] + step_coarse
  bin = torch.cat([t_coarse, t_far_edge], dim=-1)
  batch_indices = torch.arange(self.B, device=ro.device).unsqueeze(-1)
  bin_left_edge = bin[batch_indices, bin_indices]
  bin_right_edge = bin[batch_indices, bin_indices + 1]
  rand = torch.rand_like(bin_left_edge)
  t_fine = (bin_left_edge + (bin_right_edge - bin_left_edge) * rand).to(device)
  pts = ro[:, None] + rd[:, None, :] * t_fine[..., None]
  return pts, t_fine
```
## Results
The model is trained for 4 hours on A100 with Google Colab, with 15k epochs, batch size 10k, learning rate 5e-4, number of samples is 320 for the fine model and 64 for coarse model. The final psnr is around 25.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 100%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/final_final_nerf_render_test_15k_23-1.gif" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">Keys of the Void Novel View Synthesis</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/gun/train_progression.gif" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">training progression</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/gun/psnr15k.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">Training PSNR</figcaption>
  </figure>
  </div>

Note the dip in psnr at epoch 5000. I increased the probability of sampling object pixels from 0.75 to 0.9 at that epoch. The real PNSR should be 1-2 db higher than the graph since the psnr is calculated mostly on the object pixels.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/gun/coarse_fine_psnr.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">PSNR of coarse and fine model</figcaption>
  </figure>
  </div>

Comparing the performance of coarse and fine model gives some insights into the benefits of this two model method. Both the optimize sample and higher sampling rate contribute to the increase in psnr performance, of around 1.5 db. In addition to the offset, We can also observe the trend that the gap of psnr between to models is increasing, the psnr fine is leanring faster than the coarse one.

# Cube Marching

Another area to explore is converting implicit representation to explicit representation. Cube marching is such an algorithm that takes in a level set (input position, output density), such as a voxel and generates a mesh. 

[This link](https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html) provides an explanation of the algorithm, but breifly given 3d grid of density and a isovalue threshold, for each grid point, if the density at a the 8 corners of the cube is higher than the threshold, the space is occupied by volume. However if not all 8 corners are occupied, a surface is generated with the following configuration. The configuration is stored in lookup table, thus the algorithm is fast.


<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/marching_cube.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">marching cube configurations (from wikipedia)</figcaption>
  </figure>
  </div>

A step size of 0.5 is used to generate the 3d grid coordinate. Since there is no view direction in the mesh world, view direction is approximated as grid coordinate - object center. The 3d grid coordinates is then queried into the trained model. A visualization of the 3d grid density output is as follows (note that a square cube has been quried, but since there is a filter of > 1.25 in the visualizaion), only the model part is shown.

<iframe
  src="/plotly/final_density_cloud.html"
  style="width:100%;height:600px;border:none;"
  loading="lazy"
  title="Volrend trisurf (step 0.5)"
></iframe>


## Mesh
`skimage.measure.marching_cubes` is used to generate the mesh, with a empircal isovalue of 1.25. Low step size outputs finer but also noisier mesh.

| Step Size | Vertices (Verts) | Faces (Triangles) |
| :--- | ---: | ---: |
| 0.25 | 65,220 | 130,240 |
| 0.5 | 13,440 | 26,810 |
| 1.0 | 2,758 | 5,550 |

  <div style=" display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 40%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/stepquarter.jpg" alt="sunset 500"/>
    <figcaption class="text-center">step size 0.25cm</figcaption>
  </figure>
    <figure style="width: 40%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/stephalf.jpg" alt="sunset 500"/>
    <figcaption class="text-center">step size 0.5cm</figcaption>
  </figure>
    <figure style="width: 40%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/step1.jpg" alt="sunset 500"/>
    <figcaption class="text-center">step size 1cm</figcaption>

  </div>

## Coloring
Two methods of coloring is explored. One is direct vertex look up, the second is volrend. Given the mesh data and the 3d color grid from the previous pass, the color of the vertexcan be picked by looking up the nearest neighbor's color in the color grid. Volrend involes querying the model again with the full volrend pass with vertex as the ray origin towards the object center. The second method is much slower. Volrend is more accurate but much slower than direct look up. Neither is accurate because the view direction is approximated. 

  <div style=" display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 50%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/lookup.jpg" alt="sunset 500"/>
    <figcaption class="text-center">color by nearest neighbor lookup</figcaption>
  </figure>
    <figure style="width: 50%;margin-top: 0.5rem;margin-bottom:0rem">
    <img src="/images/proj4/volrend.jpg" alt="sunset 500"/>
    <figcaption class="text-center">color by volrend</figcaption>
  </figure>


  </div>




<iframe
  src="/plotly/volrend_trisurf_step0-5.html"
  style="width:100%;height:600px;border:none;"
  loading="lazy"
  title="Volrend trisurf (step 0.5)"
></iframe>
