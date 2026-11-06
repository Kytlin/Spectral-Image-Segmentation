#pragma once

// Ng-Jordan-Weiss spectral clustering: M = D^{-1/2} W D^{-1/2}; the k smallest eigenvectors of
// L_sym = I - M are the k largest of M. Their rows are normalized and clustered with k-means++.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "region_graph.hpp"

namespace seg {

enum class SpectralStatus {
    Ok,
    InvalidK,
    ZeroDegreeNode,
    DisconnectedGraph,
    EigenNotConverged,
    DegenerateEmbedding,
};
const char* to_string(SpectralStatus s);

struct SpectralParams {
    int k = 6;
    std::uint64_t seed = 42;
    int kmeans_attempts = 5;
    int kmeans_max_iter = 100;
    int eig_max_iter = 1000;
    double eig_tol = 1e-10;
    bool shift_invert = true;
};

struct EigenResult {
    bool converged = false;
    Eigen::VectorXd values;  // descending
    Eigen::MatrixXd vectors;
    int iterations = 0;
    int matvecs = 0;
    int ncv = 0;
    bool retried = false;
    bool shift_invert = false;
};

struct KMeansResult {
    std::vector<int> labels;
    double inertia = 0.0;
    int nonempty_clusters = 0;
};

struct SpectralResult {
    SpectralStatus status = SpectralStatus::Ok;
    std::string message;
    int components = 0;
    EigenResult eig;  // k + 1 pairs when k + 1 < n, so the gap can be reported
    double max_residual = 0.0;
    double spectral_gap = std::numeric_limits<double>::quiet_NaN();         // l_k - l_{k+1}
    double orthogonality_error = std::numeric_limits<double>::quiet_NaN();  // ||U^T U - I||_F
    int degenerate_rows = 0;
    KMeansResult kmeans;
    double eigen_ms = 0.0;
    double residual_ms = 0.0;
    double kmeans_ms = 0.0;

    bool ok() const { return status == SpectralStatus::Ok; }
};

SparseMatrix normalized_affinity(const SparseMatrix& w);

// Leading k eigenpairs of M (lambda_max <= 1). Shift-invert runs Lanczos on
// (M - (1 + 1e-2) I)^{-1}, which separates eigenvalues clustered near 1.
EigenResult leading_eigenpairs(const SparseMatrix& m, int k, int max_iter, double tol, bool shift_invert = true);

double orthogonality_error(const Eigen::MatrixXd& u);

// ||M v - lambda v|| / (max(1, |lambda|) ||v||)
double eigenpair_residual(const SparseMatrix& m, double lambda, const Eigen::VectorXd& v);

Eigen::MatrixXd row_normalize(const Eigen::MatrixXd& u, double eps, int* degenerate_rows);

KMeansResult kmeans_rows(const Eigen::MatrixXd& t, int k, const SpectralParams& params);

SpectralResult spectral_cluster(const SparseMatrix& w, const SpectralParams& params);

// Recursive two-way NJW. Flat k-way Ncut favours equal-volume clusters and slices uniform
// areas first; bipartitioning while Ncut stays below a threshold lets the segment count
// follow the image. Disconnected segments are split into components without an eigensolve.
struct RecursiveParams {
    int max_segments = 80;
    double ncut_threshold = 0.05;
    int min_segment_nodes = 2;
    // Try splitting along components of the subgraph without edges below this weight first.
    // Nearly decoupled subgraphs have lambda_2 close to 1, where Lanczos converges slowly.
    double weak_edge = 1e-4;
    SpectralParams split;  // k is forced to 2
};

struct RecursiveResult {
    SpectralStatus status = SpectralStatus::Ok;
    std::string message;
    std::vector<int> labels;
    int segments = 0;
    int components = 0;
    int splits_accepted = 0;
    int splits_rejected = 0;
    int component_splits = 0;
    int weak_edge_splits = 0;
    int failed_solves = 0;
    Eigen::VectorXd root_eigenvalues;  // first spectral split
    double root_gap = std::numeric_limits<double>::quiet_NaN();
    double min_gap = std::numeric_limits<double>::quiet_NaN();
    double max_orthogonality_error = 0.0;
    double root_inertia = 0.0;
    double root_ncut = 0.0;
    int total_iterations = 0;
    int total_matvecs = 0;
    int max_ncv = 0;
    bool any_retried = false;
    double max_residual = 0.0;
    double eigen_ms = 0.0;
    double kmeans_ms = 0.0;

    bool ok() const { return status == SpectralStatus::Ok; }
};

SparseMatrix induced_subgraph(const SparseMatrix& w, const std::vector<int>& nodes);

// sum_i cut(A_i, rest) / vol(A_i)
double ncut_value(const SparseMatrix& w, const std::vector<int>& part);

RecursiveResult recursive_spectral_cluster(const SparseMatrix& w, const RecursiveParams& params);

}  // namespace seg
