// Project headers (which pull in Eigen) must come before httplib.h: httplib includes
// <resolv.h>, which #defines `_res`, a name Eigen uses for function parameters.
#include "api/routes.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "api/codec.hpp"
#include "api/ids.hpp"
#include "db/database.hpp"
#include "segmenter.hpp"

#include <httplib.h>

namespace seg::api {

namespace {

using nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr const char* kUuidPattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const std::string kUuid = std::string("(") + kUuidPattern + ")";

// meta.request_ms is the server-side time spent handling this HTTP request (parsing, DB,
// and for POST /results the engine call too). It is never stored; the persisted engine
// time of a result is metrics.compute_ms.
void send_data(httplib::Response& res, int status, const json& data, Clock::time_point started) {
    const double request_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    res.status = status;
    res.set_content(json{{"data", data}, {"meta", {{"request_ms", request_ms}}}}.dump(), "application/json");
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

// Same decode flag as the CLI's cv::imread, so the engine sees identical pixels.
cv::Mat decode_image(const std::string& encoded) {
    return cv::imdecode(
        cv::_InputArray(reinterpret_cast<const uchar*>(encoded.data()), static_cast<int>(encoded.size())),
        cv::IMREAD_COLOR);
}

void create_run(const httplib::Request& req, httplib::Response& res, db::Database& db, const ApiLimits& limits) {
    const auto t0 = Clock::now();
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
    const cv::Mat bgr = decode_image(file.content);
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
    const db::RunRecord run = db.insert_run(new_uuid(), file.filename, bgr.cols, bgr.rows, file.content);
    res.set_header("Location", "/api/runs/" + run.id);
    send_data(res, 201, run_to_json(run), t0);
}

void create_result(const httplib::Request& req, httplib::Response& res, db::Database& db) {
    const auto t0 = Clock::now();
    const std::string run_id = req.matches[1];
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
    const std::optional<std::string> encoded = db.run_image(run_id);
    if (!encoded) {
        send_error(res, 404, "run_not_found", "no run with id " + run_id);
        return;
    }
    const cv::Mat bgr = decode_image(*encoded);
    if (bgr.empty()) throw std::runtime_error("stored image of run " + run_id + " does not decode");

    const SegmentationResult output = segment(bgr, params);
    if (!output.ok) {
        send_error(res, 422, "segmentation_failed", output.message.empty() ? output.status : output.message,
                   {{"status", output.status}, {"metrics", metrics_to_json(output)}});
        return;
    }
    db::NewResult row;
    row.id = new_uuid();
    row.run_id = run_id;
    row.params_json = params_to_json(params).dump();
    row.metrics_json = metrics_to_json(output).dump();
    row.superpixel_labels_png = encode_label_png(output.superpixel_labels);
    row.region_labels_png = encode_label_png(output.region_map);
    const std::optional<db::ResultRecord> stored = db.insert_result(row);
    if (!stored) {
        send_error(res, 404, "run_not_found", "no run with id " + run_id);
        return;
    }
    res.set_header("Location", "/api/results/" + stored->id);
    send_data(res, 201, result_to_json(*stored), t0);
}

void get_result_image(const httplib::Request& req, httplib::Response& res, db::Database& db) {
    const std::string result_id = req.matches[1];
    const std::string name = req.matches[2];
    const auto which = std::find_if(std::begin(kOutputImages), std::end(kOutputImages),
                                    [&name](OutputImage w) { return name == to_string(w); });
    if (which == std::end(kOutputImages)) {
        send_error(res, 404, "image_not_found",
                   "unknown image " + name + " (expected original, superpixels, segmentation or boundaries)");
        return;
    }
    const std::optional<db::ResultArtifacts> a = db.result_artifacts(result_id);
    if (!a) {
        send_error(res, 404, "result_not_found", "no result with id " + result_id);
        return;
    }
    // Re-rendered from the stored upload and label maps; the segmentation is not recomputed.
    SegmentationResult r;
    r.ok = true;
    r.image = working_image(decode_image(a->image), a->max_dim);
    r.superpixel_labels = decode_label_png(a->superpixel_labels_png);
    r.region_map = decode_label_png(a->region_labels_png);
    if (r.image.empty() || r.superpixel_labels.size() != r.image.size() || r.region_map.size() != r.image.size()) {
        throw std::runtime_error("stored artifacts of result " + result_id + " are inconsistent");
    }
    std::vector<uchar> png;
    if (!cv::imencode(".png", render_output(r, *which), png)) throw std::runtime_error("PNG encoding failed");
    res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
}

void list_experiments(const httplib::Request& req, httplib::Response& res, db::Database& db) {
    const auto t0 = Clock::now();
    int limit = 50;
    if (req.has_param("limit")) {
        const std::string v = req.get_param_value("limit");
        const bool digits = !v.empty() && v.size() <= 3 && std::all_of(v.begin(), v.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        limit = digits ? std::stoi(v) : 0;
        if (limit < 1 || limit > 200) {
            send_error(res, 400, "invalid_parameter", "limit must be an integer in [1, 200]", {{"field", "limit"}});
            return;
        }
    }
    std::optional<std::string> run_id;
    if (req.has_param("run_id")) {
        run_id = req.get_param_value("run_id");
        static const std::regex uuid(kUuidPattern);
        if (!std::regex_match(*run_id, uuid)) {
            send_error(res, 400, "invalid_parameter", "run_id must be a lowercase UUID", {{"field", "run_id"}});
            return;
        }
    }
    json items = json::array();
    for (const db::ResultRecord& rec : db.list_saved(limit, run_id)) items.push_back(result_to_json(rec));
    send_data(res, 200, {{"experiments", items}, {"count", items.size()}}, t0);
}

}  // namespace

void register_routes(httplib::Server& server, db::Database& db, const ApiLimits& limits) {
    // Leaves room for multipart framing around a maximal image.
    server.set_payload_max_length(limits.max_upload_bytes + (64u << 10));

    server.Get("/api/health", [&db](const httplib::Request&, httplib::Response& res) {
        const auto t0 = Clock::now();
        db.ping();  // throws db::Unavailable -> 503
        send_data(res, 200, {{"status", "ok"}, {"database", "ok"}}, t0);
    });
    server.Post("/api/runs", [&db, limits](const httplib::Request& req, httplib::Response& res) {
        create_run(req, res, db, limits);
    });
    server.Get("/api/runs/" + kUuid, [&db](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = Clock::now();
        const std::string id = req.matches[1];
        const std::optional<db::RunRecord> run = db.find_run(id);
        if (run) send_data(res, 200, run_to_json(*run), t0);
        else send_error(res, 404, "run_not_found", "no run with id " + id);
    });
    server.Post("/api/runs/" + kUuid + "/results",
                [&db](const httplib::Request& req, httplib::Response& res) { create_result(req, res, db); });
    server.Get("/api/results/" + kUuid, [&db](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = Clock::now();
        const std::string id = req.matches[1];
        const std::optional<db::ResultRecord> result = db.find_result(id);
        if (result) send_data(res, 200, result_to_json(*result), t0);
        else send_error(res, 404, "result_not_found", "no result with id " + id);
    });
    server.Get("/api/results/" + kUuid + "/images/([a-z]+)",
               [&db](const httplib::Request& req, httplib::Response& res) { get_result_image(req, res, db); });

    // Saving is idempotent: 201 the first time, 200 with the same saved_at afterwards.
    server.Put("/api/experiments/" + kUuid, [&db](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = Clock::now();
        const std::string id = req.matches[1];
        const db::SaveReply reply = db.save_result(id);
        switch (reply.outcome) {
            case db::SaveOutcome::Saved: send_data(res, 201, result_to_json(*reply.record), t0); break;
            case db::SaveOutcome::AlreadySaved: send_data(res, 200, result_to_json(*reply.record), t0); break;
            case db::SaveOutcome::NotFound: send_error(res, 404, "result_not_found", "no result with id " + id); break;
        }
    });
    server.Get("/api/experiments",
               [&db](const httplib::Request& req, httplib::Response& res) { list_experiments(req, res, db); });
    server.Get("/api/experiments/" + kUuid, [&db](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = Clock::now();
        const std::string id = req.matches[1];
        const std::optional<db::ResultRecord> result = db.find_result(id);
        if (result && result->saved_at) send_data(res, 200, result_to_json(*result), t0);
        else send_error(res, 404, "experiment_not_found", "result " + id + " is not in the experiment log");
    });

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
        try {
            std::rethrow_exception(ep);
        } catch (const db::Unavailable& e) {
            send_error(res, 503, "database_unavailable", e.what());
        } catch (const db::Error& e) {
            // SQL details stay in the server log.
            std::cerr << "database error [" << e.sqlstate() << "]: " << e.what() << std::endl;
            send_error(res, 500, "database_error", "database error", {{"sqlstate", e.sqlstate()}});
        } catch (const std::exception& e) {
            std::cerr << "internal error: " << e.what() << std::endl;
            send_error(res, 500, "internal_error", e.what());
        } catch (...) {
            send_error(res, 500, "internal_error", "internal error");
        }
    });
}

}  // namespace seg::api
