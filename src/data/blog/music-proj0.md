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

Try it out here! https://harmonicgraph.vercel.app/. Users can select a list of arbitrary frequencies as a scale, define a tension curve, and then generate a chord progression!

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 40%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/music/app.jpg" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">app interface  </figcaption>
  </figure>
  </div>

Harmonization and chord progressions accompany a melody and provide momentum to a piece of music. Although melodic patterns vary widely, there appears to be an underlying structure shared by many common chord progressions. Traditional music theory and heuristics attempt to describe the structures that produce such momentum in music. In contrast, this project seeks to convey these structures through a purely computational and frequency-based analysis. It does so by constructing chords and progressions that are outside of the traditional 12-note system.

The general approach of this project is to propose ideas directly in frequency space and evaluate them against established musical conventions, such as the 12-tone scale, major and minor chords, and classical progressions like V–I. If an idea strongly conflicts with human listening experience or well-established music theory, it will be refined and modified.

# Quantifying Tension

I used Recurrence Quantification Analysis (RQA), following the methodology described in this [paper](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00381/full) to compute a recurrence score for each chord. At a high level, RQA operates by extracting sliding-window samples from the signal and mapping each sample into a high-dimensional vector space. Pairwise distances between these vectors are then computed, and pairs whose distances fall below a fixed threshold are considered recurrent. More consonant frequency relationships produce stronger periodicity, leading to more repeated or cyclic patterns in the signal. As a result, a greater number of vector pairs fall within the recurrence threshold, yielding higher RQA scores.

Starting with simple two-note intervals, and using the parameters specified in the paper, I was able to reproduce similar results for individual intervals. While there exists minor numerical differences, the overall ranking of interval consonance is consistent.

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

Some modifications were made to the above recurrent analysis to better reflect musical listening experience.

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

There are many music theory rules and heuristics that describe the structure of harmonic progressions, but few examine them from a frequency-based perspective. The motivation behind this project is the observation that, regardless of genre, culture, or scale system, music tends to exhibit a rise and fall in tension. For example, harmonic progressions often build tension near the beginning and resolve back to a home or tonic chord toward the end.

## distance from home
The progression of tension is not determined solely by the level of dissonance within individual chords; it is also perceived through a chord’s distance from the home key. For example, given a key, the V chord is felt to be further from the tonic than the I chord, even though both are major triads.

To account for this effect, the tension of a chord within a harmonic progression is computed by evaluating the RQA of the chord’s three notes played simultaneously with the home note, with an amplitude boost applied to the home note. This approach rewards chords that are more closely related to the home key, resulting in higher RQA scores for harmonically stable chords.

| Roman | Type | Notes | RQA(alone) | RQA(with home) | Norm RQA with home & bass |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | Major | C3-E3-G3 | 0.011 | 0.018 | 1.000 |
| **IV** | Major | F3-A3-C3 | 0.013 | 0.024 | 0.955 |
| **i** | minor | C3-D#3-G3 | 0.003 | 0.017 | 0.936 |
| **iv** | minor | F3-G#3-C3 | 0.001 | 0.015 | 0.600 |
| **ii** | minor | D3-F3-A3 | 0.003 | 0.010 | 0.586 |
| **iii** | minor | E3-G3-B3 | 0.002 | 0.010 | 0.561 |
| **II** | Major | D3-F#3-A3 | 0.010 | 0.008 | 0.481 |
| **III** | Major | E3-G#3-B3 | 0.012 | 0.008 | 0.457 |
| **V** | Major | G3-B3-D3 | 0.012 | 0.011 | 0.442 |
| **vi** | minor | A3-C3-E3 | 0.002 | 0.011 | 0.442 |
| **VI** | Major | A3-C#3-E3 | 0.006 | 0.009 | 0.368 |
| **vii** | minor | B3-D3-F#3 | 0.003 | 0.006 | 0.233 |
| **VII** | Major | B3-D#3-F#3 | 0.006 | 0.006 | 0.231 |
| **v** | minor | G3-A#3-D3 | 0.002 | 0.006 | 0.219 |

## Voice Leading 
I want to reward smooth voice leading that produces large changes in perceived tension. One reason the V–I cadence is such an effective resolution is that the V chord is harmonically far from the home key, while the I chord represents complete resolution, yet the transition between them requires only three semitone movements.

| Progression | From Tension | To Tension | ΔT | VL Dist | ΔT Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **iii (E3-G3-B3) → I** | 0.439 | 0.000 | -0.439 | 1.0 | -1.756 |
| **VII (B3-D#3-F#3) → I** | 0.769 | 0.000 | -0.769 | 3.0 | -1.190 |
| **vi (A3-C3-E3) → I** | 0.558 | 0.000 | -0.558 | 2.0 | -1.117 |
| **III (E3-G#3-B3) → I** | 0.543 | 0.000 | -0.543 | 2.0 | -1.086 |
| **v (G3-A#3-D3) → I** | 0.781 | 0.000 | -0.781 | 4.0 | -1.042 |
| **vii (B3-D3-F#3) → I** | 0.767 | 0.000 | -0.767 | 4.0 | -1.022 |
| **VI (A3-C#3-E3) → I** | 0.632 | 0.000 | -0.632 | 3.0 | -0.979 |
| **V (G3-B3-D3) → I** | 0.558 | 0.000 | -0.558 | 3.0 | -0.864 |
| **iv (F3-G#3-C3) → I** | 0.400 | 0.000 | -0.400 | 2.0 | -0.800 |
| **II (D3-F#3-A3) → I** | 0.519 | 0.000 | -0.519 | 8.0 | -0.519 |
| **ii (D3-F3-A3) → I** | 0.414 | 0.000 | -0.414 | 5.0 | -0.498 |
| **i (C3-D#3-G3) → I** | 0.064 | 0.000 | -0.064 | 1.0 | -0.256 |
| **IV (F3-A3-C3) → I** | 0.045 | 0.000 | -0.045 | 3.0 | -0.070 |

## tension curve in real world music
With the tension metric now fully defined, it can be applied to the analysis of chord progressions. To examine how tension evolves over time in real music, we analyze chord progressions from the [CHORDONOMICON](https://arxiv.org/abs/2410.22046) dataset. Four-bar chord progressions are identified and transposed into a common major key. The five most frequently occurring chord progression types are shown below.

<div style="display: flex; gap: 1rem; justify-content: center;">

  <figure style="width: 50%;margin-top: 0.5rem; margin-bottom:0rem">
    <img src="/images/music/top5_rock.jpg" alt="2D Neural Field Architecture"/>
    <figcaption class="text-center">top 5 chord progression in rock and their chnage in tension  </figcaption>
  </figure>
  </div>

The fluctuation patterns vary widely: some progressions begin near the home key, wander outward, and resolve at the end, while others start far from the home key. This variability motivates the use of a user-defined tension curve that the algorithm is designed to follow.

# Chord Pallete

The main idea behind the chord palette is to constrain the optimizer to a manageable subset of all possible triads that can be formed from a given scale.

To ensure harmonic color and richness, a simple heuristic is used: for each note in the scale, a pair of major- and minor-equivalent chords is generated using that note as the root. This is done by first identifying the most stable triad according to the snapped, bass-boosted RQA metric. Then, analogous to forming a minor chord, the least stable note in the triad is shifted to its nearest neighboring pitch (either upward or downward), and the option that results in the lower RQA score is selected. This process is repeated for every note in the scale.

# Progression Optimizer
Given the chord palette, the algorithm enumerates all possible four-chord sequences and selects the progression whose tension changes most closely match the user-provided tension curve. Fitting to changes in tension, rather than absolute tension values, makes the comparison relative and helps remove bias.

# Other considerations

## four note chord
The project can be easily extended to handle more complex four-note chords. Doing so may require additional heuristics for the optimizer to prevent the search space from growing non-polynomially.

## tonnetz
Another possible way for defining a chord palette (i.e., a subset of chords to choose from) is to use the[tonnetz](https://en.wikipedia.org/wiki/Tonnetz). However, chords that are close on the Tonnetz tend to favor smooth voice leading and may not support strong functional resolutions, such as the V–I cadence.
