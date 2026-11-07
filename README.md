# Spectral Image Segmentation

[**Watch Demo Video**](img_seg_demo.mp4)

A C++17 spectral clustering engine with a React frontend and PostgreSQL-backed HTTP API. It partitions images into colour- and spatially coherent regions using sparse graph methods and numerical linear algebra.

The application supports interactive segmentation, caching of expensive graph computations, and persistent experiment logging.

## How to run

Run `docker compose up --build` in terminal and then go to http://localhost:8080 on web browser.

## Algorithm

The engine uses a superpixel-based implementation of Ng–Jordan–Weiss (NJW) spectral clustering based on Luxenburg's paper <i>A Tutorial on Spectral Clustering</i>:

1. **Preprocessing:** OpenCV converts the image to CIELAB colour space and applies SLIC superpixel clustering.
2. **Graph construction:** Adjacent superpixels form a sparse weighted graph, with affinities derived from colour and spatial similarity.
3. **Spectral embedding:** Spectra computes eigenvectors of the symmetric normalized affinity matrix

   $$
   M=D^{-1/2}WD^{-1/2},
   $$

   where $W$ is the affinity matrix and $D$ is the diagonal degree matrix.
4. **Clustering:** OpenCV k-means++ clusters the row-normalized eigenvector embedding. Cluster labels are projected back onto image pixels.

<i>Recursive</i> NJW is the method used for the app. It recursively bipartitions regions according to a normalized-cut stopping criterion, producing finer segmentations than flat NJW on the tested images. Standard flat NJW implementation is preserved for benchmarking.

<b>Important note:</b> This is <i>region segmentation</i> algorithm based on color and locality, <b>not</b> semantic object detection. Regions reflect the constructed graph's colour and adjacency relationships rather than object identity.

## Architecture

```text
React frontend
      |
      v
C++ HTTP API (cpp-httplib)
      |
      +--- Numerical engine (Eigen / Spectra / OpenCV)
      |
      +--- Thread-safe LRU graph cache
      |
      +--- PostgreSQL (libpq connection pool)
```

The numerical engine is independent of HTTP and persistence. Both the CLI and API use the same computation pipeline.

**Computation and caching.** Graph construction is separated from spectral clustering through `prepare_graph()` and `segment_graph()`. A bounded LRU cache stores immutable prepared graphs, keyed by image run and graph parameters.

Concurrent requests for the same key share one graph build. `shared_ptr<const PreparedGraph>` preserves object lifetime during eviction, while mutex-protected cache metadata avoids locking during expensive computation.

**Persistence.** PostgreSQL stores uploaded images, immutable segmentation results, parameters, metrics, and label maps. Saving a result is an idempotent operation on its result ID. Results remain retrievable after server restarts, independently of the in-memory cache.

**Request consistency.** Frontend request ordering prevents stale responses from replacing newer results. Stored metrics distinguish computation time from cache lookup and HTTP request latency.

## Performance

Benchmarked in Docker on a 6-core laptop using WSL2. Timings are medians of three measured runs following one warm-up.

| Image | Method | Segments | Compute (ms) | Max eigenpair residual |
|---|---|---:|---:|---:|
| Coffee cup | Recursive | 62 | 1,586 | 3e-11 |
| Downtown street | Recursive | 80 | 797 | 5e-11 |
| Tibet mountains | Recursive | 80 | 1,150 | 5e-11 |
| Coffee cup | Flat | 6 | 856 | 1e-12 |
| Downtown street | Flat | 6 | 112 | 5e-13 |
| Tibet mountains | Flat | 6 | 193 | 3e-13 |

Recursive NJW produces more regions at greater computational cost. The methods use different segment counts, so their runtime and normalized-cut values are not direct measures of equivalent segmentation quality.

### Numerical validation

Computed eigenvectors are validated using:

- **Max eigenpair residual:** Find eigenvector $v$ that maximizes $\|Mv-\lambda v\|_2/(\max(1,|\lambda|)\|v\|_2)$
- **Orthogonality error:** Find the Frobenius norm $\|U^TU-I\|_F$, where $U=[v_1\mid v_2\mid \ldots \mid v_n]$ with $v_i$ as eigenvectors, which is measured before row normalization
- **Spectral gap:** Difference between adjacent eigenvalues, ordered in increasing order, at the selected embedding dimension
- **Normalized cut (Ncut):** The ratio of edges crossing one segment to another relative to total number edges in affinity graph. Ideally, the segments correspond to disconnected components on a graph and thus Ncut would be zero in this case

Measured orthogonality errors range approximately from $10^{-15}$ to $10^{-13}$. Small eigenpair residuals confirm numerical accuracy but do not guarantee perceptually meaningful segments.

### Graph resolution trade-off

Downtown street image, recursive NJW:

| SLIC superpixels | Output segments | Compute (ms) |
|---:|---:|---:|
| 400 | 22 | 117 |
| 750 | 40 | 154 |
| 1,157 | 49 | 232 |
| 1,954 | 61 | 425 |
| 3,005 | 80 | 797 |

Increasing superpixel resolution captures finer regions but increases computational cost.

### Cache performance

The graph cache avoids repeated preprocessing, SLIC, and affinity construction.

| Image | Cold request (ms) | Cached request (ms) | Speed-up |
|---|---:|---:|---:|
| Coffee cup | 1,739 | 968 | 1.8× |
| Downtown street | 744 | 519 | 1.4× |
| Tibet mountains | 1,106 | 788 | 1.4× |

These measurements use recursive NJW with default settings. Eigenvectors are recomputed for each segmentation, limiting the benefit of graph caching.
