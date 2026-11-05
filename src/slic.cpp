#include "slic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace seg {
namespace {

struct Center {
    double l, a, b;  // mean colour
    double x, y;     // mean position
};

// Squared Lab gradient magnitude at (x, y), using central differences clamped at the border.
double gradient_squared(const ImageD& lab, int x, int y) {
    const int xl = std::max(x - 1, 0), xr = std::min(x + 1, lab.width - 1);
    const int yu = std::max(y - 1, 0), yd = std::min(y + 1, lab.height - 1);
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
        const double dx = lab.at(xr, y, c) - lab.at(xl, y, c);
        const double dy = lab.at(x, yd, c) - lab.at(x, yu, c);
        sum += dx * dx + dy * dy;
    }
    return sum;
}

}  // namespace

int enforce_connectivity(std::vector<int>& labels, int width, int height, int min_size) {
    const int n = width * height;
    if (static_cast<int>(labels.size()) != n) {
        throw std::invalid_argument("enforce_connectivity: labels size != width * height");
    }
    std::vector<int> out(static_cast<std::size_t>(n), -1);
    std::vector<int> component;
    std::vector<int> stack;
    int next_label = 0;

    // Raster scan. When we reach an unvisited pixel p, every pixel before p already has a final
    // label, in particular p's left (or, at x = 0, upper) neighbour: that is a neighbouring
    // component a too-small piece can be merged into, keeping the merged region connected.
    for (int p = 0; p < n; ++p) {
        if (out[static_cast<std::size_t>(p)] != -1) {
            continue;
        }
        const int x = p % width;
        int adjacent = -1;
        if (x > 0) {
            adjacent = out[static_cast<std::size_t>(p - 1)];
        } else if (p >= width) {
            adjacent = out[static_cast<std::size_t>(p - width)];
        }

        // Flood fill the 4-connected piece of p's original label.
        const int original = labels[static_cast<std::size_t>(p)];
        component.clear();
        stack.assign(1, p);
        out[static_cast<std::size_t>(p)] = next_label;
        while (!stack.empty()) {
            const int q = stack.back();
            stack.pop_back();
            component.push_back(q);
            const int qx = q % width;
            const int qy = q / width;
            const int neighbours[4][2] = {{qx - 1, qy}, {qx + 1, qy}, {qx, qy - 1}, {qx, qy + 1}};
            for (const auto& nb : neighbours) {
                if (nb[0] < 0 || nb[0] >= width || nb[1] < 0 || nb[1] >= height) {
                    continue;
                }
                const int r = nb[1] * width + nb[0];
                if (out[static_cast<std::size_t>(r)] == -1 &&
                    labels[static_cast<std::size_t>(r)] == original) {
                    out[static_cast<std::size_t>(r)] = next_label;
                    stack.push_back(r);
                }
            }
        }

        const bool too_small = static_cast<int>(component.size()) < min_size || original < 0;
        if (too_small && adjacent >= 0) {
            for (int q : component) {
                out[static_cast<std::size_t>(q)] = adjacent;
            }
        } else {
            ++next_label;
        }
    }
    labels.swap(out);
    return next_label;
}

SlicResult slic(const ImageD& lab, const SlicParams& params) {
    if (lab.channels != 3 || lab.width <= 0 || lab.height <= 0) {
        throw std::invalid_argument("slic: expected a non-empty 3-channel Lab image");
    }
    if (params.num_superpixels < 1 || params.compactness <= 0.0 || params.iterations < 1) {
        throw std::invalid_argument("slic: invalid parameters");
    }
    const int width = lab.width;
    const int height = lab.height;
    const int n = width * height;
    const int k_target = std::min(params.num_superpixels, n);

    // Lay out an nx x ny grid of centres with roughly square cells of area N / K.
    const double approx_step = std::sqrt(static_cast<double>(n) / k_target);
    const int nx = std::max(1, static_cast<int>(std::lround(width / approx_step)));
    const int ny = std::max(1, static_cast<int>(std::lround(height / approx_step)));
    const double step_x = static_cast<double>(width) / nx;
    const double step_y = static_cast<double>(height) / ny;
    const double step = std::sqrt(step_x * step_y);  // S
    // Each centre searches a (2S x 2S) window; use the larger cell side so the initial
    // windows cover the image.
    const int radius = static_cast<int>(std::ceil(std::max(step_x, step_y)));

    std::vector<Center> centers;
    centers.reserve(static_cast<std::size_t>(nx * ny));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int cx = std::min(static_cast<int>((i + 0.5) * step_x), width - 1);
            int cy = std::min(static_cast<int>((j + 0.5) * step_y), height - 1);
            // Move the seed to the lowest-gradient pixel in its 3x3 neighbourhood, so seeds
            // don't start on an edge or a noisy pixel.
            int best_x = cx, best_y = cy;
            double best_gradient = gradient_squared(lab, cx, cy);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int sx = cx + dx, sy = cy + dy;
                    if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                        continue;
                    }
                    const double g = gradient_squared(lab, sx, sy);
                    if (g < best_gradient) {
                        best_gradient = g;
                        best_x = sx;
                        best_y = sy;
                    }
                }
            }
            cx = best_x;
            cy = best_y;
            centers.push_back({lab.at(cx, cy, 0), lab.at(cx, cy, 1), lab.at(cx, cy, 2),
                               static_cast<double>(cx), static_cast<double>(cy)});
        }
    }

    const double spatial_factor = (params.compactness / step) * (params.compactness / step);
    std::vector<int> labels(static_cast<std::size_t>(n), -1);
    std::vector<double> distance(static_cast<std::size_t>(n));
    const int num_centers = static_cast<int>(centers.size());

    for (int iter = 0; iter < params.iterations; ++iter) {
        // Assignment: each centre claims the pixels in its window that are closer to it
        // (in the combined colour + position distance) than to any centre seen so far.
        std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
        std::fill(labels.begin(), labels.end(), -1);
        for (int k = 0; k < num_centers; ++k) {
            const Center& c = centers[static_cast<std::size_t>(k)];
            const int x0 = std::max(0, static_cast<int>(c.x) - radius);
            const int x1 = std::min(width - 1, static_cast<int>(c.x) + radius);
            const int y0 = std::max(0, static_cast<int>(c.y) - radius);
            const int y1 = std::min(height - 1, static_cast<int>(c.y) + radius);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const double dl = lab.at(x, y, 0) - c.l;
                    const double da = lab.at(x, y, 1) - c.a;
                    const double db = lab.at(x, y, 2) - c.b;
                    const double dxp = x - c.x;
                    const double dyp = y - c.y;
                    const double d = dl * dl + da * da + db * db +
                                     (dxp * dxp + dyp * dyp) * spatial_factor;
                    const std::size_t p = static_cast<std::size_t>(y * width + x);
                    if (d < distance[p]) {
                        distance[p] = d;
                        labels[p] = k;
                    }
                }
            }
        }

        // Update: move every centre to the mean colour and position of its pixels.
        // A centre that lost all its pixels keeps its old position.
        std::vector<Center> sums(centers.size(), Center{0, 0, 0, 0, 0});
        std::vector<int> counts(centers.size(), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int k = labels[static_cast<std::size_t>(y * width + x)];
                if (k < 0) {
                    continue;
                }
                Center& s = sums[static_cast<std::size_t>(k)];
                s.l += lab.at(x, y, 0);
                s.a += lab.at(x, y, 1);
                s.b += lab.at(x, y, 2);
                s.x += x;
                s.y += y;
                ++counts[static_cast<std::size_t>(k)];
            }
        }
        for (std::size_t k = 0; k < centers.size(); ++k) {
            if (counts[k] > 0) {
                const double inv = 1.0 / counts[k];
                centers[k] = {sums[k].l * inv, sums[k].a * inv, sums[k].b * inv,
                              sums[k].x * inv, sums[k].y * inv};
            }
        }
    }

    // SLIC labels are not guaranteed to be connected (a cluster can own two separate blobs),
    // and pixels outside every window stay -1. Fix both so each label is one region.
    const double expected_size = static_cast<double>(n) / num_centers;
    const int min_size =
        std::max(1, static_cast<int>(std::lround(params.min_region_fraction * expected_size)));
    SlicResult result;
    result.grid_step = step;
    result.num_labels = enforce_connectivity(labels, width, height, min_size);
    result.labels = std::move(labels);
    return result;
}

}  // namespace seg
