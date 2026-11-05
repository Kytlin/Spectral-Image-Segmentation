// graph_viz: image -> SLIC superpixels -> region adjacency graph, written out as pictures
// and a CSV of edges so the graph can be inspected by eye.
//
// Usage: graph_viz IMAGE [--out DIR] [--superpixels K] [--compactness M] [--sigma-color S]
//                        [--spatial-scale S] [--max-dim D] [--blur SIGMA] [--scale F]

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "image_io.hpp"
#include "preprocess.hpp"
#include "region_graph.hpp"
#include "slic.hpp"
#include "visualize.hpp"

namespace {

struct Options {
    std::string input;
    std::string out_dir = ".";
    seg::SlicParams slic;
    seg::GraphParams graph;
    seg::PreprocessParams pre;  // max_dim 1000 by default; --max-dim 0 = full resolution
    int scale = 2;       // upscaling of graph.png
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--out") o.out_dir = value();
        else if (arg == "--superpixels") o.slic.num_superpixels = std::stoi(value());
        else if (arg == "--compactness") o.slic.compactness = std::stod(value());
        else if (arg == "--sigma-color") o.graph.sigma_color = std::stod(value());
        else if (arg == "--spatial-scale") o.graph.spatial_scale = std::stod(value());
        else if (arg == "--max-dim") o.pre.max_dim = std::stoi(value());
        else if (arg == "--blur") o.pre.blur_sigma = std::stod(value());
        else if (arg == "--scale") o.scale = std::stoi(value());
        else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("unknown option " + arg);
        else o.input = arg;
    }
    if (o.input.empty()) {
        throw std::invalid_argument(
            "usage: graph_viz IMAGE [--out DIR] [--superpixels K] [--compactness M] "
            "[--sigma-color S] [--spatial-scale S] [--max-dim D] [--blur SIGMA] [--scale F]");
    }
    return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const seg::ImageU8 original = seg::load_rgb(o.input);

        auto t0 = std::chrono::steady_clock::now();
        const seg::Preprocessed pre = seg::preprocess(original, o.pre);
        const seg::ImageD& lab = pre.lab;
        const seg::ImageU8& working_rgb = pre.rgb;
        const double preprocess_ms = ms_since(t0);

        t0 = std::chrono::steady_clock::now();
        const seg::SlicResult sp = seg::slic(lab, o.slic);
        const double slic_ms = ms_since(t0);

        t0 = std::chrono::steady_clock::now();
        const seg::RegionGraph graph = seg::build_region_graph(lab, sp.labels, o.graph);
        const double graph_ms = ms_since(t0);

        const Eigen::VectorXd degrees = seg::node_degrees(graph.affinity);
        double min_w = 1.0, max_w = 0.0;
        for (const auto& e : graph.edges) {
            min_w = std::min(min_w, e.weight);
            max_w = std::max(max_w, e.weight);
        }

        seg::save_png(o.out_dir + "/superpixels.png",
                      seg::draw_region_boundaries(working_rgb, sp.labels, {255, 0, 0}));
        seg::save_png(o.out_dir + "/graph.png", seg::draw_region_graph(working_rgb, graph, o.scale));
        std::ofstream csv(o.out_dir + "/edges.csv");
        csv << "a,b,boundary_length,color_distance,spatial_distance,weight\n";
        for (const auto& e : graph.edges) {
            csv << e.a << ',' << e.b << ',' << e.boundary_length << ',' << e.color_distance << ','
                << e.spatial_distance << ',' << e.weight << '\n';
        }

        std::printf("image          %d x %d (working %d x %d)\n", original.width, original.height,
                    lab.width, lab.height);
        std::printf("superpixels    %d (requested %d, grid step S = %.2f px)\n", sp.num_labels,
                    o.slic.num_superpixels, sp.grid_step);
        std::printf("edges          %zu (nonzeros in W: %ld)\n", graph.edges.size(),
                    static_cast<long>(graph.affinity.nonZeros()));
        std::printf("weights        min %.3g  max %.3g   sigma_color %.2f  sigma_spatial %.2f px\n",
                    min_w, max_w, o.graph.sigma_color, graph.sigma_spatial);
        std::printf("degrees        min %.3g  max %.3g\n", degrees.minCoeff(), degrees.maxCoeff());
        std::printf("components     %d\n", seg::count_connected_components(graph.affinity));
        std::printf("time (ms)      preprocess %.1f  slic %.1f  graph %.1f\n", preprocess_ms,
                    slic_ms, graph_ms);
        std::printf("wrote          %s/{superpixels.png, graph.png, edges.csv}\n", o.out_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
