---
title: "CV Project 0: Becoming Friends with Your Camera"
author: Helena Su
pubDatetime: 2025-08-30
slug: cv-proj0
featured: false
draft: false
tags:
  - Computer Vision
description:
  perspective, focal length, and dolly zoom
---

## Overview
This simple project explores the effect of focal length on perspective. Shorter focal length have a bigger field of view and the really short ones are called fisheye cameras. They distort and wrap subject the most, especially at the edges of frame. 
## 1. Selfie: The Wrong Way vs. The Right Way


## 2. Architectural Perspective Compression
<figure class="side-by-side">
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_5832.jpg" alt="a building" class="main-img" />
      <img src="/images/IMG_5833.JPG" alt="a building" class="hover-img" />
    </div>
    <figcaption class="text-center caption-default">building from far away</figcaption>
    <figcaption class="text-center caption-hover"></figcaption>
  </div>
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_5833.JPG" alt="another building" class="main-img" />
      <img src="/images/IMG_5832.jpg" alt="another building" class="hover-img" />
    </div>
    <figcaption class="text-center caption-default">building from close up</figcaption>
    <figcaption class="text-center caption-hover"></figcaption>
  </div>
</figure>

The columns of the building appear parrallel when viewing from far away. From a close up view, the parallel columns converge more sharply towards the vanishing point.

## 3. The Dolly Zoom
<figure>
<img style="width:50%;" src="/images/dolly_zoom.gif" alt="dolly zoom"/>
<figcaption class="text-center">dolly zoom</figcaption>
</figure>
