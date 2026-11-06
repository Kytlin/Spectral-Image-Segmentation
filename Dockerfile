# Spectral segmentation engine: build + test in one stage, run from a slim runtime stage.
#
#   docker build -t spectral-seg .
#   docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg run /data/IMAGE.jpg /out
#   docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg bench /data /out

# ---- Build: compilers, headers, OpenCV dev packages; unit tests run here ---------------------
FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build ca-certificates \
        libeigen3-dev libopencv-dev libopencv-contrib-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src src
COPY apps apps
COPY tests tests
COPY data data
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

# ---- Runtime: only the shared libraries the binary links against ---------------------------
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 libopencv-contrib406t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/segment /usr/local/bin/segment
WORKDIR /work
ENTRYPOINT ["segment"]
