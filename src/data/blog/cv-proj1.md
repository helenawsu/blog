---
title: "CV Project 1: Image Alignment"
author: Helena Su
pubDatetime: 2025-09-11
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

Given three seperate color channel images, the goal is to align them properly by searching for the correct alignment vector. A similarity score for each alignment vector is caluclated and the highest score will be selected. The naive method is simply doing an exhaustive search within a  range.

For images with higher resolution, exhaustive search is too slow because it scales $O(n^2)$ wrt to search range. To optimize the search, I built an image pyramid of progressive level of downscaleness. This way, the search range can be small at each level and gradually refined to the final value.


# Single-Scale Alignment
Single scale alignment works by selecting a range of shift vectors and compare each similarity score. The process is done twice, once aligning green channel to blue channel, twice aligning red channel to blue channel.

<!-- two-square images side-by-side: good_range and bad_range -->
<div class="side-by-side equal">
  <figure>
    <img src="/images/proj1/other/good_range.jpg" alt="Good Range" loading="lazy" />
    <figcaption class="text-center">Good range</figcaption>
  </figure>

  <figure>
    <img src="/images/proj1/other/bad_range.jpg" alt="Bad Range" loading="lazy" />
    <figcaption class="text-center">Bad range</figcaption>
  </figure>
</div>

## Similairy Metric
There are different similarity metric for image alignment. The most common one simply bla bla. However, colors channels may naturally have different levels of value even in the same area. Thus, I calculate the horizontal and vertical gradient by taking the difference across neighboring pixels in the x and y directions. The final scalar score is simply calculated as the sum of two squared gradient matrices.
## Search Range
Picking the correct size of search window required some tuning. One visualization that helped me is the similarity score matrix in the search window. If the search range is big enough, there should be one alignment vector that is noticeably better than any other pixels. Otherwise, the search range should be expanded.

# Multi-Scale Alignment
# Image Pyramid (Mipmap)

# Gallery
<!-- Responsive grid gallery -->
<div class="grid-gallery" aria-label="Gallery">
  <div class="gg-track"></div>
</div>

