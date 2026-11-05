// SLIC output contract (what the region graph relies on):
//   every pixel has a label, labels are exactly 0..R-1, and each label is ONE 4-connected region.

#include <vector>

#include <gtest/gtest.h>

#include "slic.hpp"

namespace {

// Lab image: background colour with an off-grid rectangle of a second colour.
seg::ImageD two_colour_image(int width, int height) {
    seg::ImageD lab(width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool inside = x >= 23 && x < 71 && y >= 13 && y < 47;
            lab.at(x, y, 0) = inside ? 70.0 : 30.0;
            lab.at(x, y, 1) = inside ? -40.0 : 20.0;
            lab.at(x, y, 2) = 10.0;
        }
    }
    return lab;
}

// Number of pixels reachable from `start` through side-sharing pixels with the same label.
int flood_fill_size(const std::vector<int>& labels, int width, int height, int start) {
    std::vector<char> seen(labels.size(), 0);
    std::vector<int> stack{start};
    seen[static_cast<std::size_t>(start)] = 1;
    const int label = labels[static_cast<std::size_t>(start)];
    int size = 0;
    while (!stack.empty()) {
        const int p = stack.back();
        stack.pop_back();
        ++size;
        const int x = p % width, y = p / width;
        const int neighbours[4][2] = {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
        for (const auto& nb : neighbours) {
            if (nb[0] < 0 || nb[0] >= width || nb[1] < 0 || nb[1] >= height) continue;
            const auto q = static_cast<std::size_t>(nb[1] * width + nb[0]);
            if (!seen[q] && labels[q] == label) {
                seen[q] = 1;
                stack.push_back(static_cast<int>(q));
            }
        }
    }
    return size;
}

}  // namespace

TEST(Slic, LabelsAreContiguousAndEachRegionIsConnected) {
    const int width = 96, height = 64;
    seg::SlicParams params;
    params.num_superpixels = 30;
    const seg::SlicResult result = seg::slic(two_colour_image(width, height), params);

    ASSERT_EQ(result.labels.size(), static_cast<std::size_t>(width * height));
    ASSERT_GT(result.num_labels, 1);

    // Every label value lies in 0..R-1 and every value is used (no gaps).
    std::vector<int> size(static_cast<std::size_t>(result.num_labels), 0);
    std::vector<int> first_pixel(size.size(), -1);
    for (int p = 0; p < width * height; ++p) {
        const int label = result.labels[static_cast<std::size_t>(p)];
        ASSERT_GE(label, 0);
        ASSERT_LT(label, result.num_labels);
        if (size[static_cast<std::size_t>(label)]++ == 0) first_pixel[static_cast<std::size_t>(label)] = p;
    }
    for (int l = 0; l < result.num_labels; ++l) {
        ASSERT_GT(size[static_cast<std::size_t>(l)], 0) << "label " << l << " unused";
        // Connected <=> a flood fill from one pixel reaches all pixels of the label.
        EXPECT_EQ(flood_fill_size(result.labels, width, height, first_pixel[static_cast<std::size_t>(l)]),
                  size[static_cast<std::size_t>(l)])
            << "label " << l << " is split into several pieces";
    }
}

TEST(Slic, ConnectivityPassSplitsALabelThatFormsTwoPieces) {
    // Label 0 appears as two separate blobs; afterwards they must carry different labels.
    std::vector<int> labels = {0, 0, 1, 1, 0, 0,
                               0, 0, 1, 1, 0, 0};
    const int count = seg::enforce_connectivity(labels, 6, 2, /*min_size=*/1);
    EXPECT_EQ(count, 3);
    EXPECT_EQ(labels, (std::vector<int>{0, 0, 1, 1, 2, 2,
                                        0, 0, 1, 1, 2, 2}));
}
