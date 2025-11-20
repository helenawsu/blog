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
  nerf, volumetric rendering, cube marching
---

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/final_final_nerf_render_test_15k_23-1.gif" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">Keys of the Void</figcaption>
  </figure>
  </div>

# Overview
This projects and train a nerf from sratch with custom dataset. The process includes camera calibration using ArUco markers, training a warm-up 2D image field, and implementing a full 3D NeRF with optimization techniques like prioritizing object pixels and high-density ray segments. Addtionally, we explore converting the implicit NeRF representation into an explicit mesh with cube marching.

<iframe
  src="/plotly/volrend_trisurf_step0-5.html"
  style="width:100%;height:600px;border:none;"
  loading="lazy"
  title="Volrend trisurf (step 0.5)"
></iframe>

# Setup
The transformation from world coordinates to screen (pixel) coordinates is defined as:
$$
x_{screen} = K\ [R | t]\ X_{world} 
$$

For nerf training, we need to calculate world coordinate of given screen coordinate and camera extrinsics rotation matrix and translation vector. We begin by estimating the camera intrinsics K. Using these intrinsics, we then compute the extrinsics—R and t—for each image. Combining the intrinsics with the extrinsics yields the world-to-camera matrix, and its inverse gives the camera-to-world matrix used by NeRF.

## Camera Calibration 
 In this section, we are calculating camera intrinsics (K matrix) which contains focal length, image center and scale. To achieve this, 30 pictures of 6 aruco markers with various distance and angles are taken with an iPhone 12. Here are some examples of the captured images:

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

Next, we compute the camera-to-world (c2w) matrix for each image. These 2D points, together with the known camera intrinsics, are passed to `cv2.solvePnP`, which estimates the camera’s rotation and translation for each image. The rotation vector returned by solvePnP is then converted to a rotation matrix using `cv2.Rodrigues`. Together, the rotation matrix and translation vector form the extrinsics, which we combine to construct the final c2w matrix.

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

Our training dataset consists of images and their corresponding camera-to-world (c2w) transformation matrices. To train the model, we convert each pixel—defined by its screen coordinates $(u,v)$ and color—into a ray origin and a ray direction.

The ray origin is simply the camera position in world coordinates, obtained from the c2w matrix (after removing the homogeneous coordinate).
The ray direction is computed by transforming the pixel into world coordinates and taking the normalized vector from the camera origin to this 3D point.

These ray origins and directions are then batched and fed into the model. The model architecture is shown below.

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
NeRF learns the density and RGB color at every continuous point in space, rather than modeling explicit surfaces. Because of this volumetric representation, rendering a pixel’s final color from a given view direction requires integrating the contributions of all points along the corresponding camera ray (within a bounded region), using alpha blending.

In practice, this integral is approximated by dividing each ray into many small segments. Thus, for every sampled camera ray, we also generate a set of sample points along that ray. As a result, the training input has the shape: (batch size, sample size, 3) representing the 3D coordinates of each sampled point.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 25%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/nerf/ray_segment.jpg" alt="ray segments"/>
    <figcaption class="text-center">ray segments</figcaption>
  </figure>
  </div>

The final pixel color can be interpreted as the sum of the color contributions from each ray segment $i$. The contribution of a single segment is computed as the transmittance (the remaining light after passing through previous segments) multiplied by the segment’s opacity and its predicted RGB color.
$$
C = \sum_{j=near}^{j=far} T_i\cdot \alpha_i \cdot rgb_{raw}
$$

`torch.cumprod` is used to efficiently calculate transmitance T of each segment. The final weight (or effective opacity) is calculated as follows. 
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
Compared to the Lego dataset, my dataset presents a few challenges that make training slower. First, most of each image consists of background, while only a small region contains the object of interest. Second, the object itself is long, which forces the bounding box to be large; with a uniform step size along each ray, many sampled points end up in empty space. To address these issues, I introduced two improvements to the sampling method.

### 1. priotize object pixels

My training images are preprocessed with SAM to segment the object and mask the background in black. This makes it easy to identify valid pixels (those that are not pure black). When sampling pixels during training, object pixels are given a higher probability of being selected. I use an object sampling ratio of 0.9. If this ratio is set too high, floating artifacts begin to appear because the model learns too little about the background.

### 2. prioritize high density ray segments

Instead of sampling points uniformly along a ray, points with higher predicted density are given a higher probability of being selected. This helps avoid wasting samples on empty space.

To implement this optimization, two models are trained simultaneously. The first is a coarse model with 64 samples per ray, and the second is the fine model, which uses 256 + 64 samples per ray. In each epoch, the coarse model’s output is used to compute the contribution of each ray segment via volrend. The resulting weights are then fed into a new method, `sample_along_rays_priority`, to guide the fine sampling process.

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

Note the dip in PSNR at epoch 5000. At that point, I increased the probability of sampling object pixels from 0.75 to 0.9. The actual PSNR is likely 1–2 dB higher than shown in the graph, since the PSNR is calculated mostly over object pixels.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/gun/coarse_fine_psnr.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">PSNR of coarse and fine model</figcaption>
  </figure>
  </div>

Comparing the performance of the coarse and fine models provides insight into the benefits of the two-model approach. Both optimized sampling and the higher sampling rate contribute to an increase in PSNR of roughly 1.5 dB. In addition, we can observe that the gap in PSNR between the two models is widening, indicating that the fine model is learning faster than the coarse model.

# Cube Marching

Another area worth exploring is converting an implicit representation into an explicit one. Marching Cubes is an algorithm that does this: given a level set (input position, output density), such as a voxel grid, it generates a mesh.

[This link](https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html) provides a good explanation, but briefly: given a 3D grid of densities and an isovalue threshold, each cube defined by eight neighboring grid points is evaluated. If all eight corners have densities above the threshold, the cube is considered fully occupied. If not all corners are occupied, the algorithm generates a surface within the cube according to a predefined configuration. These configurations are stored in a lookup table, which makes the algorithm fast and efficient.


<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 75%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/proj4/marching_cube.png" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">marching cube configurations (from wikipedia)</figcaption>
  </figure>
  </div>

A step size of 0.5 is used to generate the 3D grid coordinates. Since there is no explicit view direction in the mesh space, the view direction is approximated as the vector from the object center to each grid point. The 3D grid coordinates are then fed into the trained model.

A visualization of the resulting 3D grid densities is shown below. Note that although a cubic grid was queried, only regions with density greater than 1.25 are displayed, so the visualization highlights the object itself.

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
Two methods for coloring the mesh were explored. The first is direct vertex lookup, and the second is volumetric rendering (volrend). The color of each face is calculated by averaging the colors of its three vertices.

In the direct lookup method, given the mesh and the 3D color grid generated in the previous step, the color of each vertex is assigned by finding the nearest neighbor in the color grid.

The volrend method, on the other hand, involves querying the model again: each vertex is treated as a ray origin pointing toward the object center, and the full volumetric rendering pass is performed. This method is more accurate but significantly slower than direct lookup.

Note that neither method is perfectly accurate, since the view direction is only approximated.

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
