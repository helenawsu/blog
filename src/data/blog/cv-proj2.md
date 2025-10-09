---
title: "CV Project 2: Fun with Filters and Frequencies"
author: Helena Su
pubDatetime: 2025-09-26
slug: cv-proj2
featured: false
draft: false
tags:
  - Computer Vision
description:
  blurring and sharpening with filters, multi-resolution blending
---
# Overview
<figure style="margin-bottom: 0.5rem">
<img style="width:78%" src="/images/proj2/yawn_mushroom/cat_yawns_on_mushroom.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">cat yawning on mushroom</figcaption>
</figure>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 25%;margin-top: 0.5rem">
    <img src="/images/proj2/yawn_mushroom/cat_original.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">cat yawning</figcaption>
  </figure>
  
  <figure style="width: 50%; margin-top: 0.5rem">
    <img src="/images/proj2/yawn_mushroom/mushroom_original.jpg" alt="original mushroom"/>
    <figcaption class="text-center">mushroom</figcaption>
  </figure>
</div>

This project explores fundamental image processing techniques with filters and frequency-domain methods, such as convolution and its application on blurring, sharpening, and edge detection. Several creative technique such as hybrid images multi-resolution blending (make an oraple) are explored.

# Convolution

We start by building a convolution from scratch. Convolution is a math operation applied on two signals that has versatile effects in image processing. A convolution works by sliding a kernel across an image, and the kernel determines how local neighboring pixels interfere with each other.

```bash
def conv(img, filter):
  padded_img = pad(img, filter)
  # filter always have 2 dimensions
  filter_w = filter.shape[1]
  filter_h = filter.shape[0]
  out = np.full(padded_img.shape, 0, dtype=float)
  start_i = filter_h // 2
  start_j = filter_w // 2
  end_i = padded_img.shape[0] - start_i
  end_j = padded_img.shape[1] - start_j
  for i in range(start_i, end_i): # row
    for j in range(start_j, end_j): # col
      cur_sum = 0.0
      for m in range(-start_i,start_i+1):
        for n in range(-start_j, start_j+1):
          cur_sum += padded_img[i+m][j+n] * filter[m+start_i][n+start_j]
      avg = cur_sum
      out[i][j] = avg
  return out

```
Observe the effect of different filters on the same image.

```bash
box_filter = np.full((3,3), 1/9)
dx = np.array([1,0,-1]).reshape(1, -1)
dy = np.array([1,0,-1]).reshape(-1, 1)
```
<div style="display: flex;  justify-content: center; flex-wrap: wrap;">
  <figure style="width: 35%;margin: 0.5rem">
    <img src="/images/proj2/me/me_original.webp" alt="me"/>
    <figcaption class="text-center">original me</figcaption>
  </figure>
  
  <figure style="width: 35%; margin: 0.5rem">
    <img src="/images/proj2/me/me_boxed.webp" alt="me boxed"/>
    <figcaption class="text-center">me boxed</figcaption>
  </figure>

  <figure style="width:35%;margin: 0.5rem">
    <img src="/images/proj2/me/me_hgrad.webp" alt="me dxed"/>
    <figcaption class="text-center">me dxed</figcaption>
  </figure>
  
  <figure style="width: 35%; margin: 0.5rem">
    <img src="/images/proj2/me/me_vgrad.webp" alt="me dyed"/>
    <figcaption class="text-center">me dyed</figcaption>
  </figure>
</div>

## Finite Difference Operator
Finite difference operator is a type of edge detection kernel given by $D_x = [1, 0, -1,]$ and $D_y = [1, 0, -1,]^T$. They detect horizontal and vertical edges respetively. Observe its effect on image cameraman.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_dy.webp" alt="cameraman horizontal difference"/>
    <figcaption class="text-center">cameraman horizontal difference</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_dx.webp" alt="cameraman vertical difference"/>
    <figcaption class="text-center">cameraman vertical difference</figcaption>
  </figure>
</div>

We can sum the magnitude of both horizontal and vertical difference to obtain a gradient magnitude image. We make an edge detector by picking a threshold.

```bash
cameraman_grad_mag = (np.square(cameraman_dx) + np.square(cameraman_dy))**0.5
threshold = 0.35
cameraman_edge = (cameraman_grad_mag > threshold).astype(int)
```

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_grad_mag.webp" alt="cameraman gradient magnitude"/>
    <figcaption class="text-center">cameraman gradient magnitude</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_edge.webp" alt="cameraman edge"/>
    <figcaption class="text-center">cameraman edge</figcaption>
  </figure>
</div>

## Derivative of Gaussian (DoG) Filter

To reduce noise in the edge detector, we can smooth the image first then apply the finite difference operator. This produces a less noisy image.
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_smoothed.webp" alt="cameraman smoothed"/>
    <figcaption class="text-center">cameraman smoothed</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_smoothed_edge.webp" alt="cameraman edge"/>
    <figcaption class="text-center">cameraman edge smoothed</figcaption>
  </figure>
</div>

Equivalently, we can convolve the finite difference operator with the gaussian filter first to obtain the derivative of gaussian (DoG) filter, then convolve the image with the composite DoG filter. The output edge image is the same as the smoothed edge above.
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 25%;margin-top: 0.5rem">
    <img src="/images/proj2/cameraman/DoG_x.webp" alt="cameraman smoothed"/>
    <figcaption class="text-center">DoG_x</figcaption>
  </figure>
  
  <figure style="width: 25%; margin-top: 0.5rem">
    <img src="/images/proj2/cameraman/DoG_y.webp" alt="cameraman edge"/>
    <figcaption class="text-center">DoG_y</figcaption>
  </figure>
</div>


## Bells & Whistles: Gradient Orientations

We can also calculate the orientation of gradient by using arctan.
```bash
orient = np.arctan2(cameraman_dy, cameraman_dx)
```

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 40%;margin-top: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_orient.png" alt="cameraman_orient"/>
    <figcaption class="text-center">cameraman orientation</figcaption>
  </figure>
  
  <figure style="width: 38%; margin-top: 0.5rem">
    <img src="/images/proj2/cameraman/cameraman_orient_mag.png" alt="cameraman edge"/>
    <figcaption class="text-center">camerama edge with orientation</figcaption>
  </figure>
</div>

# Image Sharpening
We can obtain the high frequency details by subtracting a blurred image from the original image. If we add back the details to the original image, we obtain a sharpened image. We can also combine blurring, obtaining details, and addition of details into a single convolution kernel.

```bash
unsharp_mask_filter = (1+alpha) * impulse_kernel - alpha * gaussian_kernel
```

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/sharpen/taj.jpg" alt="taj sharpened"/>
    <figcaption class="text-center">Taj</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/sharpen/taj_sharpened.webp" alt="taj sharpened"/>
    <figcaption class="text-center">Taj sharpened</figcaption>
  </figure>
</div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/sharpen/cabage.webp" alt="cabage"/>
    <figcaption class="text-center">cabage</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/sharpen/cabage_sharpened.webp" alt="cabage sharpened"/>
    <figcaption class="text-center">cabage sharpened</figcaption>
  </figure>
</div>

We can verify our unsharp mask filter by sharpening a blurred image and compare it to the original image. Observe that fine details are loss in the sharpened image.

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 45%;margin: 0.5rem">
    <img src="/images/proj2/sharpen/jellyfish_original.webp" alt="jellyfish"/>
    <figcaption class="text-center">jellyfish original</figcaption>
  </figure>
  
  <figure style="width: 45%; margin: 0.5rem">
    <img src="/images/proj2/sharpen/jellifish_blurred.webp" alt="jellyfish blurred"/>
    <figcaption class="text-center">jellyfish blurred</figcaption>
  </figure>

  <figure style="width: 45%; margin: 0.5rem">
    <img src="/images/proj2/sharpen/jellyfish_sharpened.webp" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">jellyfish sharpened from blurred</figcaption>
  </figure>
</div>

# Hybrid Images
Humans sensitivity towards frequency varies based on viewing distance. We are more sensitive to high frequency details when close up and more senstive to low frequency trends when viewing from far away. Based on this trait, we can blend two images together by combining high frequency details from one image and low frequency from another image. Based on viewing distance, the image we see changes.

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 25%;margin: 0.5rem">
    <img src="/images/proj2/hybrid/derek.jpg" alt="jellyfish"/>
    <figcaption class="text-center">human (low frequency)</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/cat.jpg" alt="jellyfish blurred"/>
    <figcaption class="text-center">cat (high frequency)</figcaption>
  </figure>

  <figure style="width: 22%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/hybrid_derek_cat.jpg" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">hybrid image of human and cat</figcaption>
  </figure>
</div>


We can also observe these images in the fourier domain.
<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 25%;margin: 0.5rem">
    <img src="/images/proj2/hybrid/fft_im1.png" alt="jellyfish"/>
    <figcaption class="text-center">human</figcaption>
  </figure>
  
  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/fft_im2.png" alt="jellyfish blurred"/>
    <figcaption class="text-center">cat</figcaption>
  </figure>

  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/fft_low.png" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">human (low pass)</figcaption>
  </figure>

  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/fft_high.png" alt="jellyfish blurred"/>
    <figcaption class="text-center">cat (high pass)</figcaption>
  </figure>

  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/fft_hybrid.png" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">hybrid image of human and cat in fourier</figcaption>
  </figure>
</div>

Other hybrid: tree diagram, butterfly tie.

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/hybrid/tree.jpg" alt="jellyfish"/>
    <figcaption class="text-center">tree (low frequency)</figcaption>
  </figure>
  
  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/tree_diagram.jpg" alt="jellyfish blurred"/>
    <figcaption class="text-center">tree diagram (high frequency)</figcaption>
  </figure>

  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/hybrid_tree_diagram.jpg" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">hybrid tree diagram</figcaption>
  </figure>
</div>

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 30%;margin: 0.5rem">
    <img src="/images/proj2/hybrid/tie.jpg" alt="jellyfish"/>
    <figcaption class="text-center">tie (low frequency)</figcaption>
  </figure>
  
  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/butterfly.jpg" alt="jellyfish blurred"/>
    <figcaption class="text-center">butterfly (high frequency)</figcaption>
  </figure>

  <figure style="width: 30%; margin: 0.5rem">
    <img src="/images/proj2/hybrid/butterflytie.png" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">butterfly tie (蝴蝶结)</figcaption>
  </figure>
</div>

# Multi-Resolution Blending
Seamless multi-resolution blending works by matching the transition speed to the scale of image details: fine details require faster transitions, while coarse details benefit from slower ones. If the blending window is too narrow, seams appear abrupt, whereas a window that is too wide produces ghosting. In practice, the window should be about the size of the largest prominent feature to avoid seams, and less than twice the size of the smallest prominent feature to avoid ghosting and aliasing.

The following 3 pictures are taken from lecture slides.
<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 27%;margin: 0.5rem">
    <img src="/images/proj2/multires_blending/big_window.jpg" alt="jellyfish"/>
    <figcaption class="text-center">window too big (ghosting)</figcaption>
  </figure>
  
  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/multires_blending/good_window.jpg" alt="jellyfish blurred"/>
    <figcaption class="text-center">good window</figcaption>
  </figure>

  <figure style="width: 25%; margin: 0.5rem">
    <img src="/images/proj2/multires_blending/small_window.jpg" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">window too small (abrupt seams)</figcaption>
  </figure>
</div>

In most cases, an image contains features of varying scales. To handle this, we can use subband-pass filters to extract features of different sizes and blend each with an appropriately chosen window size. The full image is then reconstructed by collapsing the resulting subband pyramid.

The following 1 picture is taken from lecture slides.
  <figure style="width: 75%; margin: 0.5rem auto; text-align: center;">
    <img src="/images/proj2/multires_blending/image_blending.jpg" alt="jellyfish sharpened from blurred"/>
    <figcaption class="text-center">blending pyramid</figcaption>
  </figure>

## Gaussian and Laplacian Stacks
To prepare the subband pyramid for multi-resolution blending, we first build an image pyramid by progressively downsampling the image and applying a Gaussian filter at each level. By subtracting consecutive levels of blurred images, we obtain a pyramid that captures details at different frequency scales.

Note that I implemented pyramid instead of stack because it would save compute and storage as we downsample the image.

<div style="display: flex; gap: 0; justify-content: center; flex-wrap: wrap; max-width: 100%; overflow-x: hidden;">
  <!-- Row 0: Original images -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple.jpeg" alt="apple original" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple original</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange.jpeg" alt="orange original" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange original</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_bad.webp" alt="oraple result" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple (naive blending)</figcaption>
  </figure>
  
  <!-- Row 1: Level 0 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian0.webp" alt="apple laplacian level 0" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 0</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian0.webp" alt="orange laplacian level 0" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 0</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian0.webp" alt="oraple laplacian level 0" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 0</figcaption>
  </figure>
  
  <!-- Row 2: Level 1 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian1.webp" alt="apple laplacian level 1" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 1</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian1.webp" alt="orange laplacian level 1" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 1</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian1.webp" alt="oraple laplacian level 1" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 1</figcaption>
  </figure>
  
  <!-- Row 3: Level 2 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian2.webp" alt="apple laplacian level 2" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 2</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian2.webp" alt="orange laplacian level 2" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 2</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian2.webp" alt="oraple laplacian level 2" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 2</figcaption>
  </figure>
  
  <!-- Row 4: Level 3 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian3.webp" alt="apple laplacian level 3" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 3</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian3.webp" alt="orange laplacian level 3" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 3</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian3.webp" alt="oraple laplacian level 3" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 3</figcaption>
  </figure>
  
  <!-- Row 5: Level 4 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian4.webp" alt="apple laplacian level 4" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 4</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian4.webp" alt="orange laplacian level 4" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 4</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian4.webp" alt="oraple laplacian level 4" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 4</figcaption>
  </figure>
  
  <!-- Row 6: Level 5 -->
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/apple_laplacian5.webp" alt="apple laplacian level 5" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">apple laplacian level 5</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/orange_laplacian5.webp" alt="orange laplacian level 5" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">orange laplacian level 5</figcaption>
  </figure>
  
  <figure style="width: 33.33%; margin: 0; min-width: 0;">
    <img src="/images/proj2/oraple/oraple_laplacian5.webp" alt="oraple laplacian level 5" style="width: 100%; height: auto;"/>
    <figcaption class="text-center">oraple laplacian level 5</figcaption>
  </figure>
</div>

## Masking and Blending
Multi-resolution blending can be reconstructed by adding each level of the Laplacian pyramid to the base image at the lowest resolution, which corresponds to the final level of the Gaussian pyramid. 

In addition, we construct a mask pyramid with progressively larger window sizes, ensuring that each mask matches the resolution of its corresponding downsampled image level.
  <figure style="width: 70%; margin: 0 auto; text-align: center;">
    <img src="/images/proj2/oraple/mask_row.png" alt="oraple laplacian level 5"/>
    <figcaption class="text-center">Gaussian mask pyramid</figcaption>
  </figure>

The output image is computed using the following formula, starting from the blurriest level of the Gaussian pyramid. In the case of pyramid blending, each Laplacian level must be upsampled to match the resolution of the next level before being added.
$$
\text{Out} = \sum_{k \in \text{levels}} \big( Laplacian^A_k \, Mask_k \;+\; Laplacian^B_k \, (1 - Mask_k) \big)
$$

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <figure style="width: 45%;margin: 0.5rem">
    <img src="/images/proj2/oraple/oraple_bad.webp" alt="jellyfish"/>
    <figcaption class="text-center">oraple (naive blending)</figcaption>
  </figure>
  
  <figure style="width: 45%; margin: 0.5rem">
    <img src="/images/proj2/oraple/oraple0.webp" alt="jellyfish blurred"/>
    <figcaption class="text-center">oraple (multi-resolution blending)</figcaption>
  </figure>

  Other multi-resolution blending: cat yawning on mushroom, paper temple on lace.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 33%;margin-top: 0.5rem">
    <img src="/images/proj2/temple_lace/temple_lace0.webp" alt="paper temple on lace result"/>
    <figcaption class="text-center">paper temple on lace</figcaption>
  </figure>
  
  <figure style="width: 33%;margin-top: 0.5rem">
    <img src="/images/proj2/temple_lace/lace.jpg" alt="original lace"/>
    <figcaption class="text-center">lace</figcaption>
  </figure>
  
  <figure style="width: 33%; margin-top: 0.5rem">
    <img src="/images/proj2/temple_lace/temple.jpg" alt="original paper temple"/>
    <figcaption class="text-center">paper temple</figcaption>
  </figure>
</div>
</div>

## making cat picture 
Since the background of the original cat picture is messy, a simple edge detector cannot cleanly seperate the cat. I used a lightweight segmentation model DeepLabV3 to segment out the cat and built a cat mask pyramid.

  <figure style="width: 70%; margin: 0 auto; text-align: center;">
    <img src="/images/proj2/yawn_mushroom/cat_masks_row.png" alt="oraple laplacian level 5"/>
    <figcaption class="text-center">Gaussian cat mask pyramid</figcaption>
  </figure>

<figure style="margin-bottom: 0.5rem">
<img style="width:100%" src="/images/proj2/yawn_mushroom/cat_yawns_on_mushroom.jpg" alt="cat yawning on musrhoom"/>
<figcaption class="text-center">cat yawning on mushroom</figcaption>
</figure>
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 33%;margin-top: 0.5rem">
    <img src="/images/proj2/yawn_mushroom/cat_original.jpg" alt="original cat yawning"/>
    <figcaption class="text-center">cat yawning</figcaption>
  </figure>
  
  <figure style="width: 66%; margin-top: 0.5rem">
    <img src="/images/proj2/yawn_mushroom/mushroom_original.jpg" alt="original mushroom"/>
    <figcaption class="text-center">mushroom</figcaption>
  </figure>
</div>

<div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
  <!-- Row 0: Original images -->
  
  <!-- Row 1: Level 0 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian0.webp" alt="cat laplacian level 0"/>
    <figcaption class="text-center">cat laplacian level 0</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian0.webp" alt="mushroom laplacian level 0"/>
    <figcaption class="text-center">mushroom laplacian level 0</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom0.webp" alt="result laplacian level 0"/>
    <figcaption class="text-center">result laplacian level 0</figcaption>
  </figure>
  
  <!-- Row 2: Level 1 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian1.webp" alt="cat laplacian level 1"/>
    <figcaption class="text-center">cat laplacian level 1</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian1.webp" alt="mushroom laplacian level 1"/>
    <figcaption class="text-center">mushroom laplacian level 1</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom1.webp" alt="result laplacian level 1"/>
    <figcaption class="text-center">result laplacian level 1</figcaption>
  </figure>
  
  <!-- Row 3: Level 2 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian2.webp" alt="cat laplacian level 2"/>
    <figcaption class="text-center">cat laplacian level 2</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian2.webp" alt="mushroom laplacian level 2"/>
    <figcaption class="text-center">mushroom laplacian level 2</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom2.webp" alt="result laplacian level 2"/>
    <figcaption class="text-center">result laplacian level 2</figcaption>
  </figure>
  
  <!-- Row 4: Level 3 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian3.webp" alt="cat laplacian level 3"/>
    <figcaption class="text-center">cat laplacian level 3</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian3.webp" alt="mushroom laplacian level 3"/>
    <figcaption class="text-center">mushroom laplacian level 3</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom3.webp" alt="result laplacian level 3"/>
    <figcaption class="text-center">result laplacian level 3</figcaption>
  </figure>
  
  <!-- Row 5: Level 4 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian4.webp" alt="cat laplacian level 4"/>
    <figcaption class="text-center">cat laplacian level 4</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian4.webp" alt="mushroom laplacian level 4"/>
    <figcaption class="text-center">mushroom laplacian level 4</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom4.webp" alt="result laplacian level 4"/>
    <figcaption class="text-center">result laplacian level 4</figcaption>
  </figure>
  
  <!-- Row 6: Level 5 -->
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_laplacian5.webp" alt="cat laplacian level 5"/>
    <figcaption class="text-center">cat laplacian level 5</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/mushroom_laplacian5.webp" alt="mushroom laplacian level 5"/>
    <figcaption class="text-center">mushroom laplacian level 5</figcaption>
  </figure>
  
  <figure style="width: 30%; margin: 0;">
    <img src="/images/proj2/yawn_mushroom/yawn_mushroom5.webp" alt="result laplacian level 5"/>
    <figcaption class="text-center">result laplacian level 5</figcaption>
  </figure>
</div>

# Learning
It is fun to dig through past photos can think how to creatively reuse them!

