# Spectral Image Segmentation

A C++17 engine that segments an image with spectral clustering.

1. OpenCV SLIC groups the pixels into superpixels.
2. The superpixels and their neighbours form a colour-similarity graph.
3. Spectra computes eigenvectors of the normalized affinity matrix.
4. k-means++ splits the graph into segments.

The default method is **recursive NJW**, which splits regions in two until a split would cut
through similar colours. **Flat NJW**, the textbook k-way version, is also available.

Segments follow colour and adjacency. They are not object detection.

## Run with Docker (no local OpenCV needed)

```bash
docker build -t spectral-seg .            # builds the engine and runs the unit tests (~5 min the first time)
mkdir -p out

# Segment one image (./data is mounted at /data, results go to ./out; use any file in data/)
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg run /data/tibet_mountains.jpg /out

# Finer or coarser output, with custom parameters
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg \
    run /data/downtown_toronto.jpeg /out --superpixels 1500 --k 40 --ncut 0.1 --blur 2

# Benchmark every image in data/ and write results.csv plus images to ./out
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg \
    bench /data /out --sweep-image downtown_toronto.jpeg
```

To add or remove test images, put files in `data/` (`.png .jpg .jpeg .bmp .tif .tiff .webp`).
The unit tests and the benchmark both pick up whatever is in that folder. `run` and `bench` read
the mounted folder, so no rebuild is needed for them. The unit tests in `docker build` use the
copy of `data/` taken at build time.

On Linux, add `--user "$(id -u):$(id -g)"` so the files in `./out` are owned by you rather than
root.

<!-- ## Build locally (Ubuntu 24.04)

```bash
sudo apt install -y build-essential cmake ninja-build libeigen3-dev libopencv-dev libopencv-contrib-dev
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # fetches Spectra 1.1.0 and GoogleTest
cmake --build build -j
./build/tests/segcore_tests                               # or: ctest --test-dir build
./build/segment run data/tibet_mountains.jpg out
./build/segment bench data benchmarks --sweep-image downtown_toronto.jpeg
``` -->

## Outputs (`run`)

| File | Content |
|---|---|
| `original.png` | working image (shrunk to at most 800 px on the long side) |
| `superpixels.png` | superpixel boundaries |
| `segmentation.png` | the photo with a dim colour wash per segment and thin borders |
| `boundaries.png` | segment boundaries in red |

## Parameters

| Flag | Default | Effect |
|---|---|---|
| `--method recursive\|flat` | recursive | flat produces exactly k clusters |
| `--superpixels S` | 3000 | finer detail as it rises; slower |
| `--k K` | 80 | recursive: maximum number of segments. flat: number of clusters |
| `--ncut T` | 0.05 | similarity threshold: higher means more splits and finer output |
| `--sigma-color X` | 5 | colour scale of the Gaussian similarity (Lab units): lower means more sensitive to colour change |
| `--blur SIGMA` | 1 | Gaussian pre-smoothing: higher suppresses texture and noise |
| `--spatial-scale X` | 2 | spatial similarity scale, in superpixel spacings (0 = off) |
| `--max-dim D` | 800 | the working image's long side (0 = full size) |
| `--seed N` | 42 | k-means++ seed (results are deterministic) |

Benchmark flags: `--reps R` (default 3), `--warmup W` (default 1), and `--sweep-image FILE`
(default: the first image by name). Full results are written to `results.csv`; a summary follows.

## Benchmark

These results come from running `bench` in Docker on the three images in `data/` on (6-core laptop, WSL2). Each setting was run once to warm up and then 3 more times, where the time stat is median of those 3. Every run produced identical segments across repetitions. The app uses the recursive method; flat (set to 6 segments) is the baseline for algo suggested in Luxenburg's survey paper.

| Image | Method | Segments | Time (ms) | Residual | Orthogonality | Spectral gap | Ncut |
|---|---|---|---|---|---|---|---|
| Coffee cup, 735×749 | recursive | 62 | 1586 | 3e-11 | 1e-13 | 1.5e-6 | 1.23 |
| Downtown street, 387×516 | recursive | 80 (cap) | 797 | 5e-11 | 3e-14 | 3.3e-6 | 2.16 |
| Tibet mountains, 728×410 | recursive | 80 (cap) | 1150 | 5e-11 | 4e-14 | 1.8e-6 | 1.89 |
| Coffee cup | flat | 6 | 856 | 1e-12 | 1e-14 | 5.1e-5 | 6e-5 |
| Downtown street | flat | 6 | 112 | 5e-13 | 6e-15 | 1.0e-4 | 9e-5 |
| Tibet mountains | flat | 6 (7 regions) | 193 | 3e-13 | 9e-15 | 2.9e-6 | 2e-4 |

What each column means:
- **\# of Segments**: The number of pieces the image was cut into. Recursive mode chooses this itself, up to the cap `--k`. Flat mode always produces k.

- **Time**: Time it takes to segment an image

- **Residual error**: Find eigenvector $v$ that maximizes 
$\lVert Mv − λv\rVert / (\max (1, \lvert λ\rvert)\cdot \lVert v\rVert)$ over the computed eigenpairs of $M = D^{-1/2} W D^{-1/2}$. It shows how exactly each vector satisfies Mv = λv; 0 is exact.

- **Orthogonality**: Compute the Frobenius norm ${\lVert UᵀU − I\rVert}_F$, where $U$ consists of clustered eigenvectors appended as columns, which measures how far the eigenvectors is from being an orthogonal basis. Eigenvectors of a symmetric matrix should be mutually perpendicular, so we expect result to be close to 0. Recursive mode reports slightly worse split.

- **Spectral gap**: Find $\Delta\lambda_k  := \lambda_k − \lambda_{k+1}$, the distance from the last eigenvalue used to the next one, sorted in increasing order.
  Recursive mode reports it for the first split (k = 2, on the full graph). A large gap means
  the clustering is stable. Here every gap is small (about 1e-6) because the top eigenvalues of an image graph all sit just below 1. `results.csv` also records the smallest gap over all recursive splits.

- **Normalized cut (Ncut)**: Compute $\sum_i \text{cut}(A_i, S / A_i) / \text{vol}(A_i)$ over the displayed regions $A_1, A_2,\ldots, A_k$, where $S$ is the whole directed graph constructed from image and the cut of $A$ and $B$ is defined as the sum of all weighted edges that crosses from $A$ to $B$. Smaller value means the regions are better separated. It grows with the number of regions and thus one needs to refer to the segment count to compare Ncut among the results.

The residual and orthogonality are close to double-precision rounding everywhere, so the
eigenvectors are accurate and any segmentation errors come from the method, not the numerics.

**Fine versus coarse** (downtown street, recursive, default settings except `--superpixels`):

| Superpixels | Segments | Time (ms) |
|---|---|---|
| 400 | 22 | 117 |
| 750 | 40 | 154 |
| 1157 | 49 | 232 |
| 1954 | 61 | 425 |
| 3005 (default) | 80 (cap) | 797 |


## HTTP API

```bash
docker run --rm -p 8080:8080 --entrypoint seg_server spectral-seg   # or ./build/seg_server [--port P]

curl -F image=@data/downtown_toronto.jpeg localhost:8080/api/runs                # -> run_id
curl -H 'Content-Type: application/json' -d '{"k": 6, "method": "flat", "superpixels": 400}' \
     localhost:8080/api/runs/RUN_ID/results                                      # -> result_id + metrics
curl localhost:8080/api/results/RESULT_ID
curl -o seg.png localhost:8080/api/results/RESULT_ID/images/segmentation
```

| Endpoint | Success |
|---|---|
| `GET /api/health` | 200 |
| `POST /api/runs` (multipart field `image`, ≤ 20 MB, ≤ 40 MP) | 201 run |
| `GET /api/runs/{run_id}` | 200 run with `result_ids` |
| `POST /api/runs/{run_id}/results` (JSON; `k` required) | 201 result: `params` + `metrics` |
| `GET /api/results/{result_id}` | 200 result |
| `GET /api/results/{result_id}/images/{original\|superpixels\|segmentation\|boundaries}` | 200 PNG |

Result parameters are `k`, `method`, `superpixels`, `ncut_threshold`, `sigma_color`, `spatial_scale`,
`blur_sigma`, `max_dim` and `seed`. Each has the same meaning and default as the CLI flag. Success
responses are `{"data": ...}`. Errors are `{"error": {"code", "message", "details"?}}`, with status 400
(bad JSON or parameter), 404 (unknown ID or route), 413 (too large), 415 (wrong content type or
undecodable image), 422 (the engine rejected the input, e.g. `k` above the superpixel count), 503
(store full) or 500. Runs and results live in memory only, and a restart drops them.

## Layout

```
src/region_graph.*   superpixel graph (features, adjacency, affinity W)
src/spectral.*       NJW: normalized affinity, Spectra eigenpairs, k-means++, recursive splitting
src/segmenter.*      public entry point seg::segment(), rendering, image discovery
src/api/run_store.*  in-memory runs/results (UUIDs, one mutex)
src/api/codec.*      request validation and JSON serialization
src/api/routes.*     HTTP handlers (cpp-httplib)
apps/segment.cpp     CLI (run, bench)
apps/server.cpp      HTTP server entry point
tests/               GoogleTest correctness tests
```
