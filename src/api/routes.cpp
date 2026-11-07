// Project headers (which pull in Eigen) must come before httplib.h: httplib includes
// <resolv.h>, which #defines `_res`, a name Eigen uses for function parameters.
#include "api/routes.hpp"

#include <algorithm>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "api/codec.hpp"
#include "api/run_store.hpp"
#include "segmenter.hpp"

#include <httplib.h>

namespace seg::api {

namespace {

using nlohmann::json;

constexpr const char* kUuid = "([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})";

void send_data(httplib::Response& res, int status, const json& data) {
    res.status = status;
    res.set_content(json{{"data", data}}.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, const std::string& code, const std::string& message,
                const json& details = nullptr) {
    json error{{"code", code}, {"message", message}};
    if (!details.is_null()) error["details"] = details;
    res.status = status;
    res.set_content(json{{"error", error}}.dump(), "application/json");
}

bool has_json_content_type(const httplib::Request& req) {
    const std::string type = req.get_header_value("Content-Type");
    return type.compare(0, 16, "application/json") == 0;
}

void create_run(const httplib::Request& req, httplib::Response& res, RunStore& store, const ApiLimits& limits) {
    if (!req.is_multipart_form_data()) {
        send_error(res, 415, "unsupported_media_type", "expected multipart/form-data with an 'image' file field");
        return;
    }
    if (!req.has_file("image")) {
        send_error(res, 400, "missing_field", "multipart field 'image' is required", {{"field", "image"}});
        return;
    }
    const httplib::MultipartFormData file = req.get_file_value("image");
    if (file.content.empty()) {
        send_error(res, 400, "invalid_image", "uploaded image is empty", {{"field", "image"}});
        return;
    }
    if (file.content.size() > limits.max_upload_bytes) {
        send_error(res, 413, "payload_too_large",
                   "image exceeds " + std::to_string(limits.max_upload_bytes) + " bytes");
        return;
    }
    // Same decode flag as the CLI's cv::imread, so the engine sees identical pixels.
    const cv::Mat bgr = cv::imdecode(
        cv::_InputArray(reinterpret_cast<const uchar*>(file.content.data()), static_cast<int>(file.content.size())),
        cv::IMREAD_COLOR);
    if (bgr.empty()) {
        send_error(res, 415, "unsupported_image",
                   "could not decode image (unsupported format, corrupt, or over " +
                       std::to_string(limits.max_image_pixels) + " pixels)");
        return;
    }
    if (static_cast<long>(bgr.cols) * bgr.rows > limits.max_image_pixels) {
        send_error(res, 413, "image_too_large", "image exceeds " + std::to_string(limits.max_image_pixels) + " pixels",
                   {{"width", bgr.cols}, {"height", bgr.rows}});
        return;
    }
    if (std::min(bgr.cols, bgr.rows) < limits.min_image_side) {
        send_error(res, 422, "image_too_small",
                   "image sides must be at least " + std::to_string(limits.min_image_side) + " pixels",
                   {{"width", bgr.cols}, {"height", bgr.rows}});
        return;
    }
    const std::optional<Run> run = store.add_run(file.filename, file.content.size(), bgr);
    if (!run) {
        send_error(res, 503, "store_full", "run storage is full");
        return;
    }
    res.set_header("Location", "/api/runs/" + run->id);
    send_data(res, 201, run_to_json(*run));
}

void create_result(const httplib::Request& req, httplib::Response& res, RunStore& store) {
    const std::string run_id = req.matches[1];
    const std::optional<Run> run = store.find_run(run_id);
    if (!run) {
        send_error(res, 404, "run_not_found", "no run with id " + run_id);
        return;
    }
    if (!has_json_content_type(req)) {
        send_error(res, 415, "unsupported_media_type", "expected Content-Type: application/json");
        return;
    }
    const json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) {
        send_error(res, 400, "invalid_json", "request body is not valid JSON");
        return;
    }
    SegmentParams params;
    try {
        params = parse_segment_params(body);
    } catch (const ValidationError& e) {
        send_error(res, 400, "invalid_parameter", e.what(), {{"field", e.field()}});
        return;
    }

    // Computed outside the store lock; the run's pixels are read-only.
    SegmentationResult output = segment(run->image, params);
    if (!output.ok) {
        send_error(res, 422, "segmentation_failed", output.message.empty() ? output.status : output.message,
                   {{"status", output.status}, {"metrics", metrics_to_json(output)}});
        return;
    }
    const std::optional<Result> result = store.add_result(run_id, params, std::move(output));
    if (!result) {
        send_error(res, 503, "store_full", "result storage is full");
        return;
    }
    res.set_header("Location", "/api/results/" + result->id);
    send_data(res, 201, result_to_json(*result));
}

void get_result_image(const httplib::Request& req, httplib::Response& res, const RunStore& store) {
    const std::string result_id = req.matches[1];
    const std::string name = req.matches[2];
    const std::optional<Result> result = store.find_result(result_id);
    if (!result) {
        send_error(res, 404, "result_not_found", "no result with id " + result_id);
        return;
    }
    for (const OutputImage which : kOutputImages) {
        if (name != to_string(which)) continue;
        std::vector<uchar> png;
        if (!cv::imencode(".png", render_output(result->output, which), png)) {
            throw std::runtime_error("PNG encoding failed");
        }
        res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
        return;
    }
    send_error(res, 404, "image_not_found",
               "unknown image " + name + " (expected original, superpixels, segmentation or boundaries)");
}

}  // namespace

void register_routes(httplib::Server& server, RunStore& store, const ApiLimits& limits) {
    // Leaves room for multipart framing around a maximal image.
    server.set_payload_max_length(limits.max_upload_bytes + (64u << 10));

    server.Get("/api/health", [&store](const httplib::Request&, httplib::Response& res) {
        send_data(res, 200, {{"status", "ok"}, {"runs", store.run_count()}, {"results", store.result_count()}});
    });
    server.Post("/api/runs", [&store, limits](const httplib::Request& req, httplib::Response& res) {
        create_run(req, res, store, limits);
    });
    server.Get(std::string("/api/runs/") + kUuid, [&store](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        const std::optional<Run> run = store.find_run(id);
        if (run) send_data(res, 200, run_to_json(*run));
        else send_error(res, 404, "run_not_found", "no run with id " + id);
    });
    server.Post(std::string("/api/runs/") + kUuid + "/results",
                [&store](const httplib::Request& req, httplib::Response& res) { create_result(req, res, store); });
    server.Get(std::string("/api/results/") + kUuid, [&store](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        const std::optional<Result> result = store.find_result(id);
        if (result) send_data(res, 200, result_to_json(*result));
        else send_error(res, 404, "result_not_found", "no result with id " + id);
    });
    server.Get(std::string("/api/results/") + kUuid + "/images/([a-z]+)",
               [&store](const httplib::Request& req, httplib::Response& res) { get_result_image(req, res, store); });

    // Runs for every >= 400 response; only fills in responses the handlers left empty
    // (unmatched routes, oversized payloads).
    server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        const std::string code = res.status == 404 ? "not_found"
                                 : res.status == 413 ? "payload_too_large"
                                                     : "http_error";
        send_error(res, res.status, code, httplib::status_message(res.status));
        return httplib::Server::HandlerResponse::Handled;
    });
    server.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string message = "internal error";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            message = e.what();
        } catch (...) {
        }
        send_error(res, 500, "internal_error", message);
    });
}

}  // namespace seg::api
