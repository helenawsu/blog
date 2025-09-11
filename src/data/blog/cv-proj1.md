---
title: "CV Project 1: Image Alignment"
author: Helena Su
pubDatetime: 2025-09-12
slug: cv-proj1
featured: false
draft: false
tags:
  - Computer Vision
description:
  alignment, edge Detection, and mipmap
---

# Overview

<div class="horizontal-gallery" aria-label="Gallery">
  <div class="hg-track">
    <!-- JS will populate images here -->
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const track = document.querySelector('.horizontal-gallery .hg-track');
  if (!track) return;

  const fallback = [
    'IMG_5832.webp',
    'IMG_5833.webp',
    'IMG_6043.webp',
    'IMG_6056.webp',
  ];

  function filenameLabel(name) {
    return name.replace(/\.[^/.]+$/, '') // remove extension
      .replace(/[_-]+/g, ' ')            // underscores/dashes to spaces
      .replace(/\b(\w)/g, (m) => m.toUpperCase()); // capitalize words
  }

  function populate(list) {
    track.innerHTML = '';
    list.forEach((name) => {
      const item = document.createElement('div');
      item.className = 'hg-item';

      const media = document.createElement('div');
      media.className = 'hg-media';
      const img = document.createElement('img');
      img.src = `/images/proj1/results/${name}`;
      img.alt = filenameLabel(name);
      img.loading = 'lazy';
      media.appendChild(img);

      const caption = document.createElement('figcaption');
      caption.className = 'hg-caption';
      caption.textContent = filenameLabel(name);

      item.appendChild(media);
      item.appendChild(caption);
      track.appendChild(item);
    });
  }

  fetch('/images/proj1/results/index.json')
    .then((res) => {
      if (!res.ok) throw new Error('manifest not found');
      return res.json();
    })
    .then((list) => {
      if (Array.isArray(list) && list.length) populate(list);
      else populate(fallback);
    })
    .catch(() => {
      // fallback to default images (ensure these exist or move them into proj1/results)
      populate(fallback);
    });
});
</script>

# Single-Scale Alignment
<!-- <figure class="side-by-side">
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_6043.webp" alt="a building" class="main-img" />
      <img src="/images/IMG_6056.webp" alt="a building" class="hover-img" />
    </div>
    <figcaption class="text-center caption-default">selfie from closeup</figcaption>
    <figcaption class="text-center caption-hover"></figcaption>
  </div>
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_6056.webp" alt="another building" class="main-img" />
      <img src="/images/IMG_6043.webp" alt="another building" class="hover-img" />
    </div>
    <figcaption class="text-center caption-default">selfie from far away</figcaption>
    <figcaption class="text-center caption-hover"></figcaption>
  </div>
</figure> -->

# Multi-Scale Alignment
# Image Pyramid (Mipmap)
<figure class="side-by-side">
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_5832.webp" alt="a building" class="main-img" />
      <img src="/images/IMG_5833.webp" alt="a building" class="hover-img" />
    </div>
    <figcaption class="text-center caption-default">building from far away</figcaption>
    <figcaption class="text-center caption-hover"></figcaption>
  </div>
  <div class="hover-swap">
    <div class="img-container">
      <img src="/images/IMG_5833.webp" alt="another building" class="main-img" />
      <img src="/images/IMG_5832.webp" alt="another building" class="hover-img" />
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
