#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "segmenter.hpp"

namespace fs = std::filesystem;

namespace {

void write_outputs(const fs::path& dir, const seg::SegmentationResult& r) {
    fs::create_directories(dir);
    for (const seg::OutputImage which : seg::kOutputImages) {
        const cv::Mat img = seg::render_output(r, which);
        if (!img.empty()) cv::imwrite((dir / (std::string(seg::to_string(which)) + ".png")).string(), img);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 == 1 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

std::string num(double v) {
    if (std::isnan(v)) return "NA";
    std::ostringstream os;
    os << std::setprecision(6) << v;
    return os.str();
}

std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + "\"";
}

struct Setting {
    std::string experiment;
    std::string category;
    std::string file;
    int superpixels;
    int k;
    bool save_outputs;
    seg::Method method;
    double ncut;
    std::string output_name;
};

const char* kHeader =
    "experiment,image,file,width,height,superpixels_requested,superpixels_actual,k_requested,"
    "regions_connected,graph_nodes,graph_nnz,graph_mem_est_bytes,preprocess_ms,slic_ms,graph_ms,"
    "eigen_ms,kmeans_ms,total_ms,eigenvalues,eig_converged,eig_iterations,eig_matvecs,eig_ncv,"
    "eig_retried,max_rel_residual,kmeans_inertia,kmeans_iterations,graph_components,"
    "degenerate_rows,deterministic,seed,warmup,reps,status,message,method,ncut_threshold,"
    "segments,splits_accepted,splits_rejected,failed_solves,gap_k,spectral_gap,min_split_gap,"
    "orthogonality_error,ncut_final";

void bench_one(std::ostream& csv, const Setting& s, const fs::path& data_dir, const fs::path& out_dir,
               int warmup, int reps, std::uint64_t seed, const seg::SegmentParams& base) {
    const fs::path path = data_dir / s.file;
    const std::string prefix = s.experiment + "," + s.category + "," + s.file + ",";
    const cv::Mat bgr = fs::exists(path) ? cv::imread(path.string(), cv::IMREAD_COLOR) : cv::Mat();
    if (bgr.empty()) {
        csv << prefix << ",," << s.superpixels << ",," << s.k << std::string(23, ',') << seed << ","
            << warmup << "," << reps << ",skipped_missing," << csv_quote("image not found: " + path.string()) << ","
            << seg::to_string(s.method) << "," << s.ncut << ",,,,,,,,,\n";
        std::cout << "  SKIP " << s.file << " (missing)\n";
        return;
    }
    seg::SegmentParams p = base;
    p.superpixels = s.superpixels;
    p.k = s.k;
    p.method = s.method;
    p.ncut_threshold = s.ncut;
    p.seed = seed;

    for (int i = 0; i < warmup; ++i) (void)seg::segment(bgr, p);
    std::vector<seg::SegmentationResult> runs;
    for (int i = 0; i < reps; ++i) runs.push_back(seg::segment(bgr, p));
    const seg::SegmentationResult& r = runs.front();

    bool deterministic = true;
    for (const auto& other : runs) {
        if (other.ok != r.ok) deterministic = false;
        else if (r.ok && cv::countNonZero(other.cluster_map != r.cluster_map) != 0) deterministic = false;
    }
    auto med = [&runs](double seg::StageTimings::*field) {
        std::vector<double> v;
        for (const auto& run : runs) v.push_back(run.timings.*field);
        return median(v);
    };

    std::ostringstream eig;
    eig << std::setprecision(10);
    const seg::SpectralSummary& sm = r.spectral;
    for (Eigen::Index i = 0; i < sm.eigenvalues.size(); ++i) {
        eig << (i ? ";" : "") << sm.eigenvalues(i);
    }
    const bool solved = sm.ncv > 0;
    csv << std::setprecision(6) << prefix << r.image.cols << "," << r.image.rows << "," << s.superpixels << ","
        << r.superpixels_actual << "," << s.k << "," << (r.ok ? std::to_string(r.region_count) : "") << ","
        << r.graph_nodes << "," << r.graph_nnz << "," << r.graph_mem_est_bytes << ","
        << med(&seg::StageTimings::preprocess_ms) << "," << med(&seg::StageTimings::slic_ms) << ","
        << med(&seg::StageTimings::graph_ms) << "," << med(&seg::StageTimings::eigen_ms) << ","
        << med(&seg::StageTimings::kmeans_ms) << "," << med(&seg::StageTimings::total_ms) << ","
        << eig.str() << "," << (solved ? (sm.converged ? "1" : "0") : "") << ","
        << (solved ? std::to_string(sm.iterations) : "") << ","
        << (solved ? std::to_string(sm.matvecs) : "") << ","
        << (solved ? std::to_string(sm.ncv) : "") << ","
        << (solved ? (sm.retried ? "1" : "0") : "") << ",";
    if (solved) csv << std::scientific << sm.max_residual << std::defaultfloat;
    csv << ",";
    if (r.ok) csv << sm.inertia;
    csv << ",NA," << sm.components << "," << sm.degenerate_rows << "," << (deterministic ? 1 : 0)
        << "," << seed << "," << warmup << "," << reps << "," << r.status << "," << csv_quote(r.message) << ","
        << seg::to_string(s.method) << "," << s.ncut << "," << sm.segments << "," << sm.splits_accepted << ","
        << sm.splits_rejected << "," << sm.failed_solves << "," << sm.gap_k << "," << num(sm.spectral_gap) << ","
        << num(sm.min_split_gap) << "," << num(sm.orthogonality_error) << "," << num(sm.ncut) << "\n";

    std::cout << "  " << std::left << std::setw(15) << s.experiment << " " << std::setw(22) << s.category << " "
              << std::setw(10) << seg::to_string(s.method) << " " << " S=" << std::setw(5) << s.superpixels << " k="
              << std::setw(3) << s.k << " " << r.status << "  regions=" << r.region_count
              << "  total=" << med(&seg::StageTimings::total_ms) << " ms\n";
    if (s.save_outputs) write_outputs(out_dir / "images" / (s.output_name.empty() ? s.category : s.output_name), r);
}

int run_bench(const fs::path& data_dir, const fs::path& out_dir, int warmup, int reps, std::uint64_t seed,
              const seg::SegmentParams& base, const std::string& sweep_file) {
    std::vector<std::pair<std::string, std::string>> images;
    for (const fs::path& p : seg::list_images(data_dir)) images.emplace_back(p.stem().string(), p.filename().string());
    if (images.empty()) throw std::runtime_error("no images found in " + data_dir.string());
    std::pair<std::string, std::string> sweep_image = images.front();
    if (!sweep_file.empty()) {
        const auto it = std::find_if(images.begin(), images.end(), [&](const auto& im) { return im.second == sweep_file; });
        if (it == images.end()) throw std::runtime_error("--sweep-image " + sweep_file + " not found in " + data_dir.string());
        sweep_image = *it;
    }
    std::cout << "images: " << images.size() << ", sweeps on " << sweep_image.second << "\n";
    const auto rec = seg::Method::Recursive;
    const auto flat = seg::Method::Flat;
    const double th = base.ncut_threshold;
    const int S = base.superpixels;
    const int K = base.k;
    std::vector<Setting> settings;
    for (const auto& [cat, file] : images) settings.push_back({"per_image", cat, file, S, K, true, rec, th, ""});
    for (const auto& [cat, file] : images)
        settings.push_back({"per_image_flat", cat, file, 400, 6, true, flat, th, cat + "_flat_S400_k6"});
    for (int s : {200, 400, 800}) settings.push_back({"sweep_A_S", sweep_image.first, sweep_image.second, s, 6, false, flat, th, ""});
    for (int s : {400, 800, 1200, 2000}) settings.push_back({"sweep_A_S_rec", sweep_image.first, sweep_image.second, s, K, false, rec, th, ""});
    for (int k : {2, 4, 6, 8}) settings.push_back({"sweep_B_k", sweep_image.first, sweep_image.second, 400, k, false, flat, th, ""});
    for (double t : {0.05, 0.1, 0.2}) settings.push_back({"sweep_C_ncut", sweep_image.first, sweep_image.second, S, K, false, rec, t, ""});

    fs::create_directories(out_dir);
    std::ofstream csv(out_dir / "results.csv");
    csv << kHeader << "\n";
    for (const Setting& s : settings) bench_one(csv, s, data_dir, out_dir, warmup, reps, seed, base);
    std::cout << "wrote " << (out_dir / "results.csv").string() << "\n";
    return 0;
}

int run_single(const fs::path& image, const fs::path& out_dir, const seg::SegmentParams& p) {
    const cv::Mat bgr = cv::imread(image.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("cannot read image " + image.string());
    const seg::SegmentationResult r = seg::segment(bgr, p);
    write_outputs(out_dir, r);
    std::cout << "status=" << r.status << (r.message.empty() ? "" : " (" + r.message + ")") << "\n"
              << "size=" << r.image.cols << "x" << r.image.rows << " superpixels=" << r.superpixels_actual
              << " segments=" << r.spectral.segments << " regions=" << r.region_count << " nnz=" << r.graph_nnz
              << " splits=" << r.spectral.splits_accepted << "/" << r.spectral.splits_rejected << "\n"
              << "ms: pre=" << r.timings.preprocess_ms << " slic=" << r.timings.slic_ms << " graph=" << r.timings.graph_ms
              << " eigen=" << r.timings.eigen_ms << " kmeans=" << r.timings.kmeans_ms
              << " total=" << r.timings.total_ms << "\n"
              << "max_rel_residual=" << r.spectral.max_residual << " orthogonality_error="
              << r.spectral.orthogonality_error << " spectral_gap(k=" << r.spectral.gap_k
              << ")=" << r.spectral.spectral_gap << " ncut=" << r.spectral.ncut << "\n";
    return r.ok ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage:\n  segment run IMAGE OUT_DIR [--superpixels S] [--k K] [--seed N] "
                         "[--sigma-color X] [--spatial-scale X] [--max-dim D] [--blur SIGMA]\n"
                         "                [--method recursive|flat] [--ncut T]\n"
                         "  segment bench DATA_DIR OUT_DIR [--reps R] [--warmup W] [--seed N] [--sweep-image FILE]\n";
            return 1;
        }
        const std::string mode = argv[1];
        seg::SegmentParams p;
        int reps = 3;
        int warmup = 1;
        std::string sweep_file;
        for (int i = 4; i < argc; i += 2) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            const std::string v = argv[i + 1];
            if (arg == "--superpixels") p.superpixels = std::stoi(v);
            else if (arg == "--k") p.k = std::stoi(v);
            else if (arg == "--seed") p.seed = std::stoull(v);
            else if (arg == "--sigma-color") p.graph.sigma_color = std::stod(v);
            else if (arg == "--spatial-scale") p.graph.spatial_scale = std::stod(v);
            else if (arg == "--max-dim") p.max_dim = std::stoi(v);
            else if (arg == "--blur") p.blur_sigma = std::stod(v);
            else if (arg == "--ncut") p.ncut_threshold = std::stod(v);
            else if (arg == "--method") {
                if (v == "flat") p.method = seg::Method::Flat;
                else if (v == "recursive") p.method = seg::Method::Recursive;
                else throw std::invalid_argument("--method must be flat or recursive");
            }
            else if (arg == "--reps") reps = std::stoi(v);
            else if (arg == "--warmup") warmup = std::stoi(v);
            else if (arg == "--sweep-image") sweep_file = v;
            else throw std::invalid_argument("unknown option " + arg);
        }
        if (mode == "run") return run_single(argv[2], argv[3], p);
        if (mode == "bench") return run_bench(argv[2], argv[3], warmup, std::max(1, reps), p.seed, p, sweep_file);
        throw std::invalid_argument("unknown mode " + mode);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
