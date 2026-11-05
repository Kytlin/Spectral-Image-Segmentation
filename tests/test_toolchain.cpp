// Stage 0 smoke tests: they prove that the compiler, Eigen, Spectra, GoogleTest and the
// sample data are wired up correctly. They test the toolchain, not segmentation code.

#include <array>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <gtest/gtest.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

// Path graph P_n: node i is joined to i+1 with weight 1. Its Laplacian L = D - W has the
// known eigenvalues 2 - 2 cos(pi * j / n), j = 0..n-1, which gives us an exact reference.
Eigen::SparseMatrix<double> path_graph_laplacian(int n) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        const double degree = (i == 0 || i == n - 1) ? 1.0 : 2.0;
        entries.emplace_back(i, i, degree);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> laplacian(n, n);
    laplacian.setFromTriplets(entries.begin(), entries.end());
    return laplacian;
}

}  // namespace

TEST(Toolchain, CompiledAsCpp17) {
    EXPECT_GE(__cplusplus, 201703L);
}

TEST(Toolchain, EigenDenseSolve) {
    Eigen::Matrix3d a;
    a << 4, 1, 0,
         1, 3, 1,
         0, 1, 2;
    const Eigen::Vector3d b(1, 2, 3);
    const Eigen::Vector3d x = a.ldlt().solve(b);
    EXPECT_LT((a * x - b).norm(), 1e-12);
}

TEST(Toolchain, SpectraLargestEigenvaluesOfPathLaplacian) {
    const int n = 200;
    const int k = 4;
    const Eigen::SparseMatrix<double> laplacian = path_graph_laplacian(n);

    Spectra::SparseSymMatProd<double> op(laplacian);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> solver(op, k, 3 * k);
    solver.init();
    solver.compute(Spectra::SortRule::LargestAlge, 1000, 1e-12);
    ASSERT_EQ(solver.info(), Spectra::CompInfo::Successful);

    const Eigen::VectorXd values = solver.eigenvalues();      // largest first
    const Eigen::MatrixXd vectors = solver.eigenvectors();
    ASSERT_EQ(values.size(), k);
    for (int i = 0; i < k; ++i) {
        const double expected = 2.0 - 2.0 * std::cos(kPi * (n - 1 - i) / n);
        EXPECT_NEAR(values(i), expected, 1e-9) << "eigenvalue " << i;
        // Each returned pair must actually satisfy L v = lambda v with unit-length v.
        const Eigen::VectorXd v = vectors.col(i);
        EXPECT_NEAR(v.norm(), 1.0, 1e-10);
        EXPECT_LT((laplacian * v - values(i) * v).norm(), 1e-8) << "residual " << i;
    }
}

TEST(SampleData, ShapesPngExistsAndIsPng) {
    const std::string path = std::string(SEG_SAMPLE_DIR) + "/shapes.png";
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file) << "missing " << path;

    // Every PNG starts with this fixed 8-byte signature.
    const std::array<unsigned char, 8> png_signature = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::array<char, 8> header{};
    file.read(header.data(), header.size());
    ASSERT_EQ(file.gcount(), 8);
    for (std::size_t i = 0; i < header.size(); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(header[i]), png_signature[i]) << "byte " << i;
    }
}
