#pragma once

#include <vector>

#include "image.hpp"

namespace seg {

// SLIC superpixels (Achanta et al., "SLIC Superpixels Compared to State-of-the-Art Superpixel
// Methods", TPAMI 2012), written from the paper.
struct SlicParams {
    // Desired number of superpixels K. The result is close to K, not exactly K.
    int num_superpixels = 400;
    // m: trades colour similarity against compactness. The assignment distance is
    //     D^2 = d_lab^2 + (d_xy / S)^2 * m^2,
    // where S is the grid spacing. Larger m -> rounder, more grid-like superpixels;
    // smaller m -> superpixels hug colour edges more tightly but get irregular.
    double compactness = 10.0;
    // Assignment/update rounds. The paper reports that 10 is enough for most images.
    int iterations = 10;
    // Connected pieces smaller than min_region_fraction * (N / K) pixels are merged into a
    // neighbouring superpixel during the connectivity pass.
    double min_region_fraction = 0.25;
};

struct SlicResult {
    // labels[p] for pixel p = y * width + x. Labels are 0..num_labels-1 with no gaps,
    // and every label is ONE 4-connected region of pixels.
    std::vector<int> labels;
    int num_labels = 0;
    double grid_step = 0.0;  // S, the initial spacing between cluster centres (pixels)
};

// Runs SLIC on a 3-channel CIELAB image. Deterministic (no randomness).
SlicResult slic(const ImageD& lab, const SlicParams& params);

// Relabels so that every label is a single 4-connected component, and merges components
// smaller than min_size pixels (and any unassigned pixels, label -1) into an adjacent one.
// Returns the number of labels; `labels` is rewritten in place with labels 0..count-1.
int enforce_connectivity(std::vector<int>& labels, int width, int height, int min_size);

}  // namespace seg
