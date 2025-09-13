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
<div class="side-by-side-arrows" aria-hidden="false" style="margin-top:0rem;">
  <figure class="ssa-item">
    <img src="/images/proj1/other/out.webp" alt="Step 1" loading="lazy" />
    <figcaption>Step 1: Misaligned</figcaption>
  </figure>

  <div class="ssa-arrow" aria-hidden="true">➜</div>

  <figure class="ssa-item">
    <img src="/images/proj1/other/church4.webp" alt="Step 2" loading="lazy" />
    <figcaption>Step 2: Downscaled</figcaption>
  </figure>

  <div class="ssa-arrow" aria-hidden="true">➜</div>

  <figure class="ssa-item">
    <img src="/images/proj1/results/church.webp" alt="Step 3" loading="lazy" />
    <figcaption>Step 3: Aligned</figcaption>
  </figure>
</div>


<script>
(function () {
  const fallback = [
    'IMG_5832.webp',
    'IMG_5833.webp',
    'IMG_6043.webp',
    'IMG_6056.webp',
  ];

  function filenameLabel(name) {
    return name.replace(/\.[^/.]+$/, '')
      .replace(/[_-]+/g, ' ')
      .replace(/\b(\w)/g, (m) => m.toUpperCase());
  }

  function createItem(entry) {
    const name = typeof entry === 'string' ? entry : entry.file;
    const captionText = (typeof entry === 'string') ? filenameLabel(name) : (entry.caption || filenameLabel(name));

    const item = document.createElement('div');
    item.className = 'gg-item';

    const media = document.createElement('div');
    media.className = 'gg-media';
    const img = document.createElement('img');
    img.src = `/images/proj1/results/${name}`;
    img.alt = captionText;
    img.loading = 'lazy';
    media.appendChild(img);

    const caption = document.createElement('figcaption');
    caption.className = 'gg-caption';
    caption.textContent = captionText;

    item.appendChild(media);
    item.appendChild(caption);

    // click to open modal; pass the original entry so modal can use caption if available
    item.addEventListener('click', () => openModal(entry));
    return item;
  }

  function populate(list) {
    const track = document.querySelector('.grid-gallery .gg-track');
    if (!track) return;
    track.innerHTML = '';
    list.forEach((entry) => track.appendChild(createItem(entry)));
  }

  async function buildGallery() {
    const track = document.querySelector('.grid-gallery .gg-track');
    if (!track) return;
    try {
      const res = await fetch('/images/proj1/results/index.json');
      if (!res.ok) throw new Error('manifest not found');
      const list = await res.json();
      if (Array.isArray(list) && list.length) { populate(list); return; }
    } catch (e) {
      // ignore
    }
    populate(fallback);
  }

  // Modal helper: create modal if not present
  function ensureModal() {
    if (document.getElementById('gg-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'gg-modal';
    modal.className = 'gg-modal';
    modal.hidden = true;
    modal.setAttribute('aria-hidden', 'true');
    modal.innerHTML = `
      <button class="gg-close" aria-label="Close">×</button>
      <figure class="gg-figure">
        <img class="gg-img" src="" alt="" />
        <figcaption class="gg-caption"></figcaption>
      </figure>
    `;
    document.body.appendChild(modal);

    const closeBtn = modal.querySelector('.gg-close');
    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  }

  function openModal(entry) {
    ensureModal();
    const modal = document.getElementById('gg-modal');
    if (!modal) return;
    const modalImg = modal.querySelector('.gg-img');
    const modalCaption = modal.querySelector('.gg-caption');

    const name = typeof entry === 'string' ? entry : entry.file;
    const captionText = (typeof entry === 'string') ? filenameLabel(name) : (entry.caption || filenameLabel(name));

    modalImg.src = `/images/proj1/results/${name}`;
    modalImg.alt = captionText;
    modalCaption.textContent = captionText;
    modal.hidden = false;
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    const modal = document.getElementById('gg-modal');
    if (!modal) return;
    modal.hidden = true;
    modal.setAttribute('aria-hidden', 'true');
    const modalImg = modal.querySelector('.gg-img');
    if (modalImg) modalImg.src = '';
    document.body.style.overflow = '';
  }

  // Init
  function init() {
    ensureModal();
    buildGallery();
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') init();
  else document.addEventListener('DOMContentLoaded', init, { once: true });

  document.addEventListener('astro:after-swap', () => setTimeout(init, 50));
})();
</script>

Given three seperate color channel images, the goal is to align them properly by searching for the correct alignment vector. A similarity score for each alignment vector is caluclated and the highest score will be selected. The naive method is simply doing an exhaustive search within a range.

For high-resolution images, exhaustive search is too slow since it scales as $O(n^2)$ with respect to the search range. To optimize this, I built an image pyramid with progressively downscaled levels. At each level, the search range remains small and the alignment is gradually refined until the final value is reached.


# Single-Scale Alignment
Single scale alignment works by selecting a range of shift vectors and compare each similarity score. The process is done twice, once aligning green channel to blue channel, twice aligning red channel to blue channel.



## Similarity Metric
There are many similarity metric for image alignment. I implemented the normalized cross correlation (NCC). It is a common similarity metric used in signal processing, normalized by vectors' magnitude. Given two vectors $a$ and $b$, the formula is: 

$$ NCC(a,b) = \frac{a \cdot b}{\|a\|\|b\|} $$

 Intuitively, NCC measures the cosine of the angle between the two vectors, with values closer to 1 indicating stronger similarity. Since our image are 2D matrices, each image matrix is flattened to 1D vector first.
## Search Range
Picking the correct size of search window required some tuning. One visualization that helped me is the similarity score matrix in the search window. If the search range is big enough, there should be one alignment vector that is noticeably better than any other pixels. Otherwise, the search range should be expanded.
<!-- two-square images side-by-side: good_range and bad_range -->
<div class="side-by-side equal">
  <figure>
    <img src="/images/proj1/other/good_range.jpg" alt="Good Range" loading="lazy" />
    <figcaption class="text-center">Good Search Range</figcaption>
  </figure>

  <figure>
    <img src="/images/proj1/other/bad_range.jpg" alt="Bad Range" loading="lazy" />
    <figcaption class="text-center">Bad Search Range (the correct alignment vector is outside of this window)</figcaption>
  </figure>
</div>

# Multi-Scale Alignment
To speed up alignment, a high-resolution image can be downscaled to reduce the search range. The resulting approximate alignment vector is then used as the new search center, allowing the range to be narrowed further. This process forms a recursive algorithm: at each level, single-scale alignment is performed with the help of the approximate vector, and the alignment is progressively refined.

## Image Pyramid (Mipmap)
To achieve multi-scale alignment, some preprocessing is required. My image pyramid builder generates multiple levels of images, where each level is half the size of the previous one. Each downsampled image is created by averaging every group of four neighboring pixels. The original image is padded so that its dimensions equals the nearest $2^n$, corresponding to the chosen number of levels to ensure integer dimension.

```bash
downscaled_image = image.reshape(image.shape[0]//2, 2, image.shape[1]//2, 2).mean(axis=(1,3))
```
The number of levels is chosen so that the smallest level has dimension of around 250x250. TIF images have dimension over 3000x3000, so 6 levels are generate.


<div class="side-by-side-arrows" aria-hidden="false" style="margin-top:0rem;">
  <figure class="ssa-item">
    <img src="/images/proj1/other/cathedral_level0.webp" alt="level 1" loading="lazy" />
    <figcaption>L1 dimension: 392 × 344</figcaption>
  </figure>

  <div class="ssa-arrow" aria-hidden="true">➜</div>

  <figure class="ssa-item">
    <img src="/images/proj1/other/cathedral_level1.webp" alt="level 2" loading="lazy" />
    <figcaption>L2 dimension: 196 × 172</figcaption>
  </figure>

  <div class="ssa-arrow" aria-hidden="true">➜</div>

  <figure class="ssa-item">
    <img src="/images/proj1/other/cathedral_level2.webp" alt="level 3" loading="lazy" />
    <figcaption>L3 dimension: 98 × 86</figcaption>
  </figure>
</div>

## Search Range
The initial search range is set to about 10% of the image height. In the recursive steps, the search range varies between 6 and 20 pixels, depending on the image. Because of aliasing and noise, this range sometimes needs to be significantly larger than 4, the scaling factor between levels.

# Gallery
<!-- Responsive grid gallery -->
<div class="grid-gallery" aria-label="Gallery">
  <div class="gg-track"></div>
</div>

# Bells and Whistles
I explored another similarity metric, which is the gradient of the image. I calculate the horizontal and vertical gradient by taking the difference across neighboring pixels in the x and y directions. This act as horizontal and vertical edge detector. The final scalar score is simply calculated as the sum of two squared gradient matrices. It produced finer alignment than the NCC metric.
```bash
horizontal_grad = np.diff(img, axis=1)
vertical_grad = np.diff(img, axis=0)
score = np.sum((v_grad_blue - v_grad_green)**2) + np.sum((h_grad_blue - h_grad_green)**2)
```

<div class="side-by-side equal">
  <figure>
    <img src="/images/proj1/other/h_gradient.png" alt="Good Range" loading="lazy" />
    <figcaption class="text-center">Horizontal Gradient of Cathedral</figcaption>
  </figure>

  <figure>
    <img src="/images/proj1/other/v_gradient.png" alt="Bad Range" loading="lazy" />
    <figcaption class="text-center">Vertical Gradient of Cathedral</figcaption>
  </figure>
</div>

# Observations
Most images are easy to align. "self_portrait" is more difficult due to messy backgrounds and indistinctive contours (mostly rocks and blurry grass). Cropping out the black strips on the edges helped.

Moving objects are still inevitably misaligned. This creates a beautiful water surface.
  <figure>
    <img src="/images/proj1/other/water.jpg" alt="water" loading="lazy" />
    <figcaption class="text-center">Colorful water</figcaption>
  </figure>