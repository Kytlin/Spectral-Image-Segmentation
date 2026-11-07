#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>

#include "api/routes.hpp"
#include "api/run_store.hpp"

#include <httplib.h>

namespace {

// Write end of the shutdown pipe; the only state the signal handler touches.
int g_stop_fd = -1;

extern "C" void on_stop_signal(int) {
    const char byte = 1;
    [[maybe_unused]] const ssize_t n = write(g_stop_fd, &byte, 1);
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::size_t max_runs = 64;
    std::size_t max_results = 512;
    try {
        for (int i = 1; i < argc; i += 2) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            const std::string v = argv[i + 1];
            if (arg == "--host") host = v;
            else if (arg == "--port") port = std::stoi(v);
            else if (arg == "--max-runs") max_runs = std::stoul(v);
            else if (arg == "--max-results") max_results = std::stoul(v);
            else throw std::invalid_argument("unknown option " + arg);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n"
                  << "usage: seg_server [--host H] [--port P] [--max-runs N] [--max-results N]\n";
        return 1;
    }

    const seg::api::ApiLimits limits;
    // OpenCV refuses to decode beyond this, which bounds memory for decompression bombs.
    // Read once on first decode, so it must be set before any thread starts.
    setenv("OPENCV_IO_MAX_IMAGE_PIXELS", std::to_string(limits.max_image_pixels).c_str(), 1);

    // Self-pipe shutdown: libraries start threads before main, so SIGINT/SIGTERM may land on
    // any thread. The handler only writes a byte; the waiter calls stop(), which is not
    // async-signal-safe.
    int stop_pipe[2];
    if (pipe(stop_pipe) != 0) {
        std::cerr << "error: pipe() failed\n";
        return 1;
    }
    g_stop_fd = stop_pipe[1];
    struct sigaction action{};
    action.sa_handler = on_stop_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    seg::api::RunStore store(max_runs, max_results);
    httplib::Server server;
    seg::api::register_routes(server, store, limits);
    server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::cout << req.method << " " << req.path << " " << res.status << std::endl;
    });

    std::thread waiter([&server, read_fd = stop_pipe[0]] {
        char byte = 0;
        while (read(read_fd, &byte, 1) < 0) {
        }
        server.stop();
    });
    std::cout << "listening on " << host << ":" << port << std::endl;
    const bool ok = server.listen(host, port);
    if (!ok) {
        std::cerr << "error: cannot listen on " << host << ":" << port << "\n";
        on_stop_signal(0);
    }
    waiter.join();
    std::cout << "stopped" << std::endl;
    return ok ? 0 : 1;
}
