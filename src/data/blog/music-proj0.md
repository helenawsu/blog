---
title: "Continous Frequency Chord Progression Generator"
author: Helena Su
pubDatetime: 2025-12-18
slug: music-proj0
featured: false
draft: false
tags:
  - music
description:
  chord, progressions
---

# Overview

Try it out here! https://harmonicgraph.vercel.app/

Harmonizatoin / chord progression accompanies melody and give momuntum to a piece of music. Even though melody pattern differ so much, there seems to be underlying structure seen in common chord progression. much music theory and heuristic exist to descfibes the structure needed to produce emotional momentum, but this project attempts to uncover the sturcture through pure computatoinal / physical / frequency analysis by building chords and progression that moving away traditional 12 note setting.

The general approach of this project is, propose an idea on the frequency space and test it out on established rules (12 key notes, major vs minor chords, or the classical V-I progressoin). If it completely violates human listening experience and or music theory rules, the idea will be tweaked and iterate.

# Quantifying Tension

I used RQA (recurrent quantification analysis) following this [paper](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00381/full) to calculate recurrence score for each chord. On the high level, rqa takes sliding window samples and convert each sample to high dimension vector. Pairwise distance between each vector is calcualted. distance within a certain threshold is consider recurrent. When a frequency is more consonant, more repeats / cycles happen which led to more frequent distance that are considered recurrent.

Starting with simple two-note intervals. Following the paper's parameters, I was able to produce similar results on intervals. There were minor variations but the ranking is the same.

  <div style="display: flex; gap: 1rem; justify-content: center;">
  <figure style="width: 20%;margin-top: 0.5rem">
    <img src="/images/music/interval_recurrence_my.jpg" alt="sunset 500"/>
    <figcaption class="text-center">calculated normalized recurrence</figcaption>
  </figure>
  <figure style="width: 50%;margin-top: 0.5rem">
    <img src="/images/music/interval_recurrence_paper.jpg" alt="sunset 500"/>
    <figcaption class="text-center">normalized recurrence (first column) from Trulla et al</figcaption>
  </figure>

  </div>

Some modifications were made to the raw recurrent analysis to better reflect musical listening experience.

## ratio snapping 
There are two common tuning systems: just intonation and 12-tone equal temperament (12-TET). Just intonation expresses musical intervals as exact integer ratios, whereas 12-TET divides the octave into 12 equal logarithmic steps. Although the perceptual difference between these systems is often indiscernable for human listeners, it has a significant impact on RQA-based analysis. For example, a ratio difference such as 2:3 versus 2:3.01 is hardly distinguishable to the human ear, but it can substantially disrupt the RQA score.

The table below demonstrates this discrepancy. Notably large differences appear in the major sixth, major third, and minor third intervals.
| Interval | Equal Temperment | Just Intonation | RQA_ET | RQA_Just |
| :--- | :--- | :--- | :--- | :--- |
| **Octave** | 2.00 | 2.00  | 1.00 | 1.00 |
| **Perfect Fifth** | 1.498 | 1.500  | 0.447 | 0.478 | 
| **Perfect Fourth** | 1.335 | 1.333  | 0.333 | 0.345 | 
| **Major Sixth** | 1.681 | 1.667  | 0.005 | 0.304 | 
| **Major Third** | 1.260 | 1.250 | 0.028 | 0.283 | 
| **Minor Third** | 1.189 | 1.200  | 0.071 | 0.230 | 

To address this issue, a snapping system is introduced. Each interval is snapped to the closest simple rational ratio within a 1% tolerance. A set of candidate ratios is generated using a Farey grid, with the maximum grid spacing constrained to be below 0.01 to satisfy the tolerance requirement. Before computing the final RQA score of a chord, each note is snapped to the simplest valid ratio—defined as the ratio with the minimum sum of numerator and denominator within ±1% of its interval relative to the chord’s root.

## bass boost
When I applied the rqa method to three-note triads, a discrepancy emerges between human perception of consonance and how RQA evaluates a chord. An example is the C major triad (C3–E3–G3) compared with the second-inversion F major triad (C3–F3–A3). From a musical perspective, C major is perceived as the more stable and consonant chord. However, when evaluated purely in terms of frequency ratios, the C–F–A triad appears more consonant, since it corresponds to the simpler ratio 3:4:5, whereas C–E–G corresponds to 4:5:6.

This discrepancy can be explained by the role of harmonics and perceived root. Human listeners tend to identify the lowest pitch as the root of the chord. In the case of C–E–G, the bass note C aligns with the harmonic root, reinforcing stability. In contrast, for C–F–A, the lowest note C is not the harmonic root of the triad, which creates a sense of instability despite its simpler frequency ratio.

To account for this perceptual effect, a bass penalty term is added to the RQA calculation. The method first determines a virtual fundamental frequency whose harmonic series contains all notes of the triad. This virtual frequency is computed by expressing the chord tones as ratios relative to the perceived root and then finding the least common multiple (LCM) of the denominators. For example, the C–E–G triad with ratio 4:5:6 can be rewritten as $1/1,5/4,6/4$, yielding a virtual frequency of 4. Similarly, the virtual frequency for C–F–A is 3.

If the virtual frequency is a power of two, it corresponds to a direct octave relationship with the bass and incurs no penalty. If it is not a power of two, a penalty coefficient proportional to $log(lcm)$ is applied. Larger LCM values indicate a more complex harmonic alignment with the bass, and therefore a more perceptually unstable chord.

The following table demonstrate the effect with and without bass stability.

| Chord | raw rqa value | rqa value with bass penalty |
| :--- | :--- | :--- |
| **C3-E3-G3** (4:5:6) | 0.011 | 0.011 |
| **C3-F3-A3** (3:4:5) | 0.015 | 0.010 |

# Tension Progression

There are many specific musical theory and heuristic on the structure of harmonic progression  

## Voice Leading 

# Chord Pallete

The main idea of chord pallete is to limit the optimzier to look at a subset of all possible triads that can be made up in the scale.

# Progression Optimizer


# Other considerations
## four note chord?
## no tonnetz

