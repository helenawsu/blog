---
title: "CV Project 5: Diffusion Models"
author: Helena Su
pubDatetime: 2025-12-07
slug: cv-proj5
featured: false
draft: false
tags:
  - Computer Vision
description:
  diffusion, flow matching
---

# Overview

# Diffusion Model

In diffusion models, the forward process adds noise to an image while the reverse process, or denoising, predicts removes noise to reconstruct the image. This section explores image generation and editing using DeepFloyd’s diffusion model, where we focus on implementing the forward process of adding noise while the denoising is handled by DeepFloyd’s pretrained stage-1 U-Net. We demonstrate various ways to edit and manipulate text prompts and model inputs to create interesting AI-generated images.

I generate text embeddings for the following prompts:
`['a fish bass',
 'a electric bass',
 'a bird crane',
 'a construction crane',
 'a nature tree',
 'a binary tree',
 'a silicon chip',
 'a potato chip',
 'a sewing thread',
 'a cpu thread',]`

I generated the following images using seed 666. By comparing the different inference steps, we can see that more inference steps bring a more concentrated concept of the prompt into the image, which is sometimes appropriate and sometimes not. The potato chip became less realistic (possibly because it is a simple concept; 40 steps is too much), and the electric bass became more realistically detailed. Unfortunately, all of them look very AI; the highly saturated colors are uncomfortable.

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; align-items: start; width: 50%; margin: 0 auto;">
  <figure style="margin:0">
    <img src="/images/proj5/fish_bass_20.png" alt="fish bass step 20" style="width:100%;height:auto;"/>
    <figcaption class="text-center">fish bass — step 20</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/electric_bass_20.png" alt="electric bass step 20" style="width:100%;height:auto;"/>
    <figcaption class="text-center">electric bass — step 20</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/potato_chip20.png" alt="potato chip step 20" style="width:100%;height:auto;"/>
    <figcaption class="text-center">potato chip — step 20</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/silicon_chip_20.png" alt="silicon chip step 20" style="width:100%;height:auto;"/>
    <figcaption class="text-center">silicon chip — step 20</figcaption>
  </figure>

  <figure style="margin:0">
    <img src="/images/proj5/fish_bass_40.png" alt="fish bass step 40" style="width:100%;height:auto;"/>
    <figcaption class="text-center">fish bass — step 40</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/electric_bass_40.png" alt="electric bass step 40" style="width:100%;height:auto;"/>
    <figcaption class="text-center">electric bass — step 40</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/potato_chip_40.png" alt="potato chip step 40" style="width:100%;height:auto;"/>
    <figcaption class="text-center">potato chip — step 40</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/silicon_chip_40.png" alt="silicon chip step 40" style="width:100%;height:auto;"/>
    <figcaption class="text-center">silicon chip — step 40</figcaption>
  </figure>
</div>

## Noising (the forward process)

The forward process of a diffusion model simply adds noise to the original image. The level of noise is determined by the timestep, where $x_0$ is clean and $x_t$ is completely noise. $\bar{\alpha}_t$ (alpha cumulative product) is a precalculated paramter and tuned to the DeepFloyd denoising process.

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0,1)
$$

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_original.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower t=0</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_250.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=250</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=500</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=750</figcaption>
  </figure>
  </div>

## Denoising (the reverse process)
### classical denoising

The classical way of denoising simply apply a Gaussian blur filter.

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_original.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower t=0</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/classical_denoise_250.png" alt="sunset 500"/>
    <figcaption class="text-center">classical denoised tower t=250</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/classical_denoise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">classical denoised tower t=500</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/classical_denoise_750.png" alt="sunset 500"/>
    <figcaption class="text-center">classical denoised tower t=750</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_original.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower t=0</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_250.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=250</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=500</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_noise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">noisy tower t=750</figcaption>
  </figure>
  </div>

### one-step denoising

Using DeepFloyd’s stage 1 denoising U-Net model, images with different levels of noise are denoised to clean images in a single step.

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_original.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower t=0</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/one_step_denoise_250.png" alt="sunset 500"/>
    <figcaption class="text-center">one step denoise t=250</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/one_step_denoise_500.png" alt="sunset 500"/>
    <figcaption class="text-center">one step denoise t=500</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/one_step_denoise_750.png" alt="sunset 500"/>
    <figcaption class="text-center">one step denoise t=750</figcaption>
  </figure>
  </div>

  ### iterative denoising
Diffusion models are better at removing noise little by little, one timestep at a time. To skip steps and avoid running all 1,000 denoising steps, we can use strided timesteps. This works because the diffusion process is defined by differential equations that remain consistent even when timesteps are spaced apart. In the formula, x $x_{t}$ is the one step cleaner image at the new timestep $t'$, computed by blending the clean image, the noisy image at $t$, and added noise according to the diffusion coefficients.

  $$
  x_{t'} = \frac{\sqrt{\bar{\alpha}_{t'}}\beta_t}{1 - \bar{\alpha}_t}x_0 + \frac{\sqrt{\bar{\alpha}_t(1 - \bar{\alpha}_{t'})}}{1 - \bar{\alpha}_t}x_t + v_{\sigma}
  $$

$x_{t}$ is your image at timestep $t$. $x_{t'}$ is your noisy image at timestep $t'$ where $t' < t$ (less noisy). $x_0$ is our current estimate of the clean image using one-step denoising. $v_{\sigma}$ is variance.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_90.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower t=90</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_240.png" alt="sunset 500"/>
    <figcaption class="text-center">iterative denoise t=240</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_390.png" alt="sunset 500"/>
    <figcaption class="text-center">iterative denoise t=390</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_540.png" alt="sunset 500"/>
    <figcaption class="text-center">iterative denoise t=540</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_690.png" alt="sunset 500"/>
    <figcaption class="text-center">iterative denoise t=690</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_original.png" alt="sunset 500"/>
    <figcaption class="text-center">original tower</figcaption>
  </figure>
    </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/iter_clean.png" alt="sunset 500"/>
    <figcaption class="text-center">iterative denoised tower</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/one_step_denoise.png" alt="sunset 500"/>
    <figcaption class="text-center">one step denoise tower</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/classical_denoise_750.png" alt="sunset 500"/>
    <figcaption class="text-center">classical denoise</figcaption>
  </figure>
  </div>

### diffusion model sampling

Here are 5 diffusion result of the prompt `"a high quality photo"`.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/scratch0_1-5.png" alt="sunset 500"/>
    <figcaption class="text-center">first high quality photo</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/scratch1_3-5.png" alt="sunset 500"/>
    <figcaption class="text-center">second high quality photo</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/scratch1_5-5.png" alt="sunset 500"/>
    <figcaption class="text-center">third high quality photo</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/scratch3_1-5.png" alt="sunset 500"/>
    <figcaption class="text-center">fourth high quality photo</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/scratch4_1-5.png" alt="sunset 500"/>
    <figcaption class="text-center">fifth high quality photo</figcaption>
  </figure>
  </div>

## Classifier-Free Guidance (CFG)
The above result isn't high quality enough. To inject more awesomeness to our generated images, we apply CFG. We consider text embedding `"a high quality photo"` as the conditional input and empty text embedding `""` as the null unconditional input. The diffrence between the conditional noise ($\epsilon_c$) and unconditonal noise ($\epsilon_u$) represents the extent of the condition. The `"a high quality photo"` condition can be amplified by choosing a $\gamma > 1$ which yield the final noise.

$$
\epsilon = \epsilon_u + \gamma(\epsilon_c - \epsilon_u)
$$

Some samples of higher quality photos with $\gamma = 7$
<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cfg0.png" alt="sunset 500"/>
    <figcaption class="text-center">first higher quality photo</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cfg1.png" alt="sunset 500"/>
    <figcaption class="text-center">second higher quality photo</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cfg2.png" alt="sunset 500"/>
    <figcaption class="text-center">third higher quality photo</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cfg3.png" alt="sunset 500"/>
    <figcaption class="text-center">fourth higher quality photo</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cfg4.png" alt="sunset 500"/>
    <figcaption class="text-center">fifth higher quality photo</figcaption>
  </figure>
  </div>

## Image-to-Image Translation
We can edit an image by injecting different levels of noise and then iteratively denoising it with CFG. The more noise we add (lower $i$), the less similar the result will be to the original image.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_1.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_3.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_5.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_7.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_10.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/tower_edit_20.png" alt="sunset 500"/>
    <figcaption class="text-center">tower edit i=20</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_1.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_3.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_5.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_7.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_10.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/da_edit_20.png" alt="sunset 500"/>
    <figcaption class="text-center">da edit i=20</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_1.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_3.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_5.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_7.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_10.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/dan_edit_20.png" alt="sunset 500"/>
    <figcaption class="text-center">dan edit i=20</figcaption>
  </figure>
  </div>

#### Editing Hand-Drawn and Web Images

We can apply the same technique on hand-drawn images.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie1.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie3.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie5.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie7.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie10.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/cookie20.png" alt="sunset 500"/>
    <figcaption class="text-center">cookie i=20</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web1.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web3.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web5.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web7.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web10.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/web20.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn transamerica pyramid i=20</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h2.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h3.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h5.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h7.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h10.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/h20.png" alt="sunset 500"/>
    <figcaption class="text-center">hand drawn image i=20</figcaption>
  </figure>
  </div>

  #### Impainting
Another editing technique is impaiting. We create a mask over the area we want to edit, generate noise only in that masked region, and combine it with the original image elsewhere, so only the masked part gets modified while the rest remains unchanged. Because the denoising model predicts noise for the entire image, the inpainted region naturally blends with the surrounding unmasked areas.

  $$
  x_t \leftarrow \mathbf{m}x_t + (1 - \mathbf{m})\text{forward}(x_{orig}, t)
  $$

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/inpaint0.png" alt="sunset 500"/>
    <figcaption class="text-center">impainted campanile (with the Statue of Liberty! O.o)</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/mushroom_inpaint.png" alt="sunset 500"/>
    <figcaption class="text-center">impainted mushroom</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/window_inpaint.png" alt="sunset 500"/>
    <figcaption class="text-center">impainted window</figcaption>
  </figure>

  </div>

#### Text-Conditional Image-to-image Translation

Same as technique but with different prompt and images.

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_1.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_3.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_5.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_7.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_10.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/jellytree_20.png" alt="sunset 500"/>
    <figcaption class="text-center">jellytree i=20</figcaption>
  </figure>
  </div>

<div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_1.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_3.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_5.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_7.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_10.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/binary_arrow_20.png" alt="sunset 500"/>
    <figcaption class="text-center">binary arrow i=20</figcaption>
  </figure>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_1.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=1</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_3.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=3</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_5.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=5</figcaption>
  </figure>
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_7.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=7</figcaption>
  </figure>
      <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_10.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=10</figcaption>
  </figure>
        <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/potato_tower_20.png" alt="sunset 500"/>
    <figcaption class="text-center">potato tower i=20</figcaption>
  </figure>
  </div>

  #### Visual Anagrams

In visual anagrams, two noise predictions are generated: e1 from the original image with prompt 1, and e2 from a horizontally flipped image with prompt 2, which is then flipped back. The final noise e is obtained by averaging these two predictions, combining information from both orientations.

  $$
  \epsilon_1 = \text{CFG of UNet}(x_t, t, p_1) \\
\epsilon_2 = \text{flip}(\text{CFG of UNet}(\text{flip}(x_t), t, p_2)) \\
\epsilon = (\epsilon_1 + \epsilon_2)/2
  $$

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/nature_binary_tree.png" alt="sunset 500"/>
    <figcaption class="text-center">nature and binary tree</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/anagram_crane.png" alt="sunset 500"/>
    <figcaption class="text-center">construction crane and bird crane</figcaption>
  </figure>
  </div>

  #### Hybrid Images

In hybrid images, two noise predictions are generated from the same image: e1 using prompt 1 and e2 using prompt 2. The final noise e is formed by combining the low-frequency components of e1 with the high-frequency components of e2, blending the two images at different levels of detail.

$$
\epsilon_1 = \text{CFG of UNet}(x_t, t, p_1) \\
\epsilon_2 = \text{CFG of UNet}(x_t, t, p_2) \\
\epsilon = f_{\text{lowpass}}(\epsilon_1) + f_{\text{highpass}}(\epsilon_2)
$$

  <div style="display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/fish_electric_bass.png" alt="sunset 500"/>
    <figcaption class="text-center">electric bass and fish bass</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/hybrid3.png" alt="sunset 500"/>
    <figcaption class="text-center">potato chip and silicon chip</figcaption>
  </figure>
  <figure style="width: 15%;margin-top: 0.5rem">
    <img src="/images/proj5/hybrid1.png" alt="sunset 500"/>
    <figcaption class="text-center">construction crane and bird crane</figcaption>
  </figure>
  </div>

  # Flow Matching

  In the above section, we relied on pre-trained DeepFloyd to denoise the image. In this section, flow matching will be implemented form scratch for the denoising process. We first focus on MNIST hand-written digit dataset, then explore more complicated ones such as chinese calligraphy.

  ## Building a Single-Step Denoising UNet
  A single-step denoising UNet $D_{\theta}$, which aims to map a noisy image $z$ to a clean image $x$, is built and train with the following loss function: 

  $$
  L = \mathbb{E}_{z,x}||D_{\theta}(z) - x||^2
  $$

  The model architecture is the following. It has a encoder before the flatten operation, and decoder after the unflatten operation. The skip connections allows the decoder to deconstruct the image with both the high level feature from the bottleneck, and the spatial details during compression. 

<figure style="width: 100%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/unet.jpg" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">single-step denoising unet architecture</figcaption>
  </figure>

  ### training 

  The following formula is used to add different levels of noise to an image, where $x$ is normalized with a range of $[0, 1]$.

  $$
  z = x + \sigma\epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
  $$


<figure style="width: 75%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/noisy5.png" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">example noising with increasing sigmas</figcaption>
  </figure>

  The parameters include $sigma = 5$, learning rate of 1e-4, hidden dimension of 128, batch size of 256, with 5 epochs. Total training time is around 5 minunites on T4 GPU. Noise is applied when fetched from the dataloader to ensure the model see new noise every epoch.

  <figure style="width: 50%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/part1unetloss.png" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">training loss of denoising sigma=5</figcaption>
  </figure>

<div style="display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/epoch1.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 1 test result (top: ground truth, middle: input, bottom: output)</figcaption>
  </figure>
  <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/epoch5.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 5 test result (top: ground truth, middle: input, bottom: output)</figcaption>
  </figure>
  </div>

### testing
  #### testing with out-of-distribution sigma

<figure style="width: 75%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/oot.png" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">testing result on ut-of-distribution sigmas</figcaption>
  </figure>

  #### generate digits from pure noise

  To make denoising a generative task, the model is trained and used to denoise pure, random Gaussian noise. Notice how they look like hand-drawn scribbles and contain broken lines, but do not resemble any digits.

<figure style="width: 50%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/class_traincurve.png" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">training loss of denoising pure noise</figcaption>
  </figure>

<div style="display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/class_epoch1.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 1 generation result</figcaption>
  </figure>
  <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/class_epoch5.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 5 generation result</figcaption>
  </figure>
    <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/class_epoch10.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 10 generation result</figcaption>
  </figure>
  </div>

  ## Building a Flow Matching Model

  To convert the single-step denoise unet model from the previous section to a flow matching unet model, only two modifications need to be made: injecting time $t$ into the model, and compute the loss based on the predicted change that produces the denoising effect (the flow) rather than comparing pixel-by-pixel with a ground-truth image.

The value $t$ represents the noise level at a given timestep and is assumed to follow a simple linear interpolation.

$$
x_t = (1 - t)x_0 + t x_1 \quad \text{where } x_0 \sim \mathcal{N}(0, I), t \in [0, 1]. 
$$

The flow (velocity, change from noisy to clean image) is defined as follows.
$$
u(x_t, t) = \frac{d}{dt} x_t = x_1 - x_0.
$$

The new loss function that evaluates the flow intead of pixels is defined as follows.

$$
L = \mathbb{E}_{x_0 \sim p_0(x_0), x_1 \sim p_1(x_1), t \sim U[0, 1]} ||(x_1 - x_0) - u_{\theta}(x_t, t)||^2
$$

$t$ is injected at two levels of the decoder.
<figure style="width: 100%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/flowmatch_arch.jpg" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">flow match unet architecture</figcaption>
  </figure>

  ### training
  We train the model where we randomly sample $t$ from 0 to 1 inside each iteration. Batch size and hidden dimension is 64. Training time is around 8 seconds per epoch on A100.

<figure style="width: 50%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/timeconditionloss.jpg" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">training loss of time-conditioned unet</figcaption>
  </figure>

  ### sampling
The model outputs the full update needed to move from $x_t$ (the noisy image at time $t$ toward a cleaner state. Because diffusion model is better at removing noise little by little, we apply this update repeatedly across many timesteps: at each step we update the image using the model’s predicted change, advance $t$, and then repeat. This iterative process follows the update rule.

  $$
  \mathbf{x}_t = \mathbf{x}_t + \frac{1}{T} u_{\theta}(\mathbf{x}_t, t)
  $$


<div style="display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/flowmatch_epoch1.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 1 generation result</figcaption>
  </figure>
  <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/flowmatch_epoch5.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 5 generation result</figcaption>
  </figure>
    <figure style="width: 45%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/flowmatch_epoch10.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 10 generation result</figcaption>
  </figure>
  </div>

  ### adding class-conditioning to unet

In order to make model output valid digits instead of scribble, we inject one hot vector of digit class (from 0 to 9) to the model. We implement dropout rate of 10% where the class conditioning vector is set to 0. The class vector is injected as unflatten = c1 * unflatten + t1 and up1 = c2 * up1 + t2.

The sampling function is modified to take in class vector and classifier-free guidance with $\gamma = 5$. 

$$
u_{uncond} = u_{\theta}(\mathbf{x}_t, t, 0) \\
u_{cond} = u_{\theta}(\mathbf{x}_t, t, c) \\
u = u_{uncond} + \gamma(u_{cond} - u_{uncond}) \\
\mathbf{x}_t = \mathbf{x}_t + \frac{1}{T} u
$$

<div style="display: flex; gap: 1rem; justify-content: center;">
    <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/actual_digits_epoch1.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 1 generation result</figcaption>
  </figure>
  <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/actual_digits_epoch5.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 5 generation result</figcaption>
  </figure>
    <figure style="width: 100%;margin-top: 0.5rem">
    <img src="/images/proj5/proj5b/actual_digits_epoch10.png" alt="sunset 500"/>
    <figcaption class="text-center">epoch 10 generation result</figcaption>
  </figure>
  </div>

To get rid of the scheduler, I decrease the learning rate to 1e-4 which works well.

# Make AI Write Chinese Calligraphy
We can train the flow matching model on the 
[chinese calligraphy dataset](https://www.kaggle.com/datasets/yuanhaowang486/chinese-calligraphy-styles-by-calligraphers/data).

## generating calligrpahyer's scribbles

I picked four famous chinese calligrapher, 王羲之 (wxz), 于右任(yyr), 范文强(fwq), 宋徽宗(shz), and traind the model four times on each of their calligraphy. Since there is no conditioning, the generated samples look like scribbles / texture that has the touch of strokes but no structure whatsover.

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; width: 100%; max-width: 1100px; margin: 0.5rem auto; align-items: start;">
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/wxz_epoch10.png" alt="wxz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz scribble</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/shz_epoch10.png" alt="shz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz scribble</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/yyr_epoch10.png" alt="yyr epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated yyr scribble</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/fwq_epoch10.png" alt="fwq epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated fwq scribble</figcaption>
  </figure>

  <figure style="margin:0">
    <img src="/images/proj5/proj5b/wxz_real.png" alt="wxz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">real wxz samples</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/shz_real.png" alt="shz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">real shz samples</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/yyr_real.png" alt="yyr real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">real yyr samples</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/fwq_real.png" alt="fwq real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">real fwq samples</figcaption>
  </figure>
</div>

<br/> 

## character structure guide

Obviously, the model needs some structural guidance on how to write the Chinese characters. My first approach was to imprint a bit of a standard font structure into the noisy input image. To extract the actual characters from the calligraphy font, I used the Python library `easyocr` to detect and label them, and then rendered the corresponding standard font using `simsun.ttc` downloaded from web.

<figure style="width: 50%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/guidedinput.jpg" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">input noise with character structure hints</figcaption>
  </figure>

It helps but not much.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; width: 75%; max-width: 1100px; margin: 0.5rem auto; align-items: start;">
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/shz_guided.png" alt="wxz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz with structure hints</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/wxz_guided.png" alt="shz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz with structure hints</figcaption>
  </figure>


  <figure style="margin:0">
    <img src="/images/proj5/proj5b/shz_noguide.png" alt="wxz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz without structure hints</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/wxz_noguidance.png" alt="shz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz without structure hints</figcaption>
  </figure>
</div>

To ensure the model learns how to use the structural hints, the training loop provides it not with pure noise but with noise that already contains those hints. During sampling, the model also receives the same structural guidance. This consistency helps the model produce noticeably better outputs.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; width: 100%; max-width: 1100px; margin: 0.5rem auto; align-items: start;">
  <figure style="margin:0">
    <img src="/images/proj5/shz_train_guided.png" alt="wxz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz with structure hints trained train data</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/shz_test_guided.png" alt="shz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz with structure hints trained test data</figcaption>
  </figure>


  <figure style="margin:0">
    <img src="/images/proj5/wxz_train.png" alt="wxz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz without structure hints trained train data</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/wxz_test.png" alt="shz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz without structure hints trained test data</figcaption>
  </figure>
</div>

## classifier-free guidance

To improve results further, I use classifier-free guidance. Although I can’t assign a one-hot vector to each character (unlike digits), I can treat pure noise as the unconditional input and noise with structural hints as the conditional input.

Several parameters can be tuned here. Signal controls the strength of the structural hint—too much and the output starts to resemble the standard SimSun font. Gamma adjusts how strongly the conditional “structure” is emphasized. The offset in noise adjusts the contrast between the noise and the structural hint. Values are chosen empirically through unserious experimentation.

```
signal = mask * -2.5
noises = signal + torch.randn_like(shape) + 0.4
gamma=3
```

Although using CFG produces outputs that are more coherent, structurally accurate, and readable, I don't think they capture the stylistic nuances of calligraphy as well as the outputs generated above with more structural freedom. There is a tradeoff (at least in this model) between adhering to the structure of a standard font and allowing the expressive, freeform style of calligraphic strokes.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; width: 100%; max-width: 1100px; margin: 0.5rem auto; align-items: start;">
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/final_calligraphy/shz_train_output.png" alt="wxz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz with cfg on train data</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/final_calligraphy/shz_test_output1.png" alt="shz epoch 10" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz with cfg on test data</figcaption>
  </figure>


  <figure style="margin:0">
    <img src="/images/proj5/proj5b/final_calligraphy/wxz_train_output.png" alt="wxz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated shz cfg on train data</figcaption>
  </figure>
  <figure style="margin:0">
    <img src="/images/proj5/proj5b/final_calligraphy/wxz_test_output.png" alt="shz real" style="width:100%;height:auto;"/>
    <figcaption class="text-center">generated wxz cfg on test data</figcaption>
  </figure>
</div>

## AI's interpretation on chinese calligraphy

How does AI understand Chinese character structure? By setting a high gamma value for CFG, we observe outputs that emphasize the square shapes in characters, confirming that Chinese characters has a blocky square structure (方块字), lol.

<figure style="width: 75%; margin: 0.5rem auto;">
    <img src="/images/proj5/proj5b/final_calligraphy/shzstructure.png" alt="sunset 500" style="display:block; margin:0 auto; max-width:100%; height:auto;"/>
    <figcaption class="text-center">ai thinks chinese character structure is square</figcaption>
  </figure>

