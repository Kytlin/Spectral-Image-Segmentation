# Spectral segmentation engine, HTTP API and React UI: build + test stages, slim runtime stage.
#
#   docker build -t spectral-seg .
#   docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg run /data/IMAGE.jpg /out
#   docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/out:/out" spectral-seg bench /data /out
#   docker compose up --build        (UI + HTTP API + PostgreSQL on http://localhost:8080)

# ---- Web: React UI built to static files; its unit tests run here ----------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web ./
RUN npm test && npm run build

# ---- Build: compilers, headers, OpenCV dev packages; unit tests run here ---------------------
FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build ca-certificates \
        libeigen3-dev libopencv-dev libopencv-contrib-dev libpq-dev \
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
        libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 libopencv-contrib406t64 libpq5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/segment /src/build/seg_server /usr/local/bin/
COPY --from=web /web/dist /usr/share/seg/web
ENV SEG_WEB_ROOT=/usr/share/seg/web
EXPOSE 8080
WORKDIR /work
ENTRYPOINT ["segment"]
