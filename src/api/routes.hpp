#pragma once

#include <cstddef>

namespace httplib {
class Server;
}

namespace seg::api {

class RunStore;

struct ApiLimits {
    std::size_t max_upload_bytes = 20u << 20;
    long max_image_pixels = 40'000'000;
    int min_image_side = 16;
};

// Registers the /api routes and JSON error handlers; store must outlive server.
void register_routes(httplib::Server& server, RunStore& store, const ApiLimits& limits);

}  // namespace seg::api
