#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace seg::db {

// The database is unreachable or dropped the connection; the request may be retried later.
class Unavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A statement failed for another reason; sqlstate is the 5-character PostgreSQL error code.
class Error : public std::runtime_error {
public:
    Error(std::string sqlstate, const std::string& message)
        : std::runtime_error(message), sqlstate_(std::move(sqlstate)) {}
    const std::string& sqlstate() const noexcept { return sqlstate_; }

private:
    std::string sqlstate_;
};

// Timestamps are ISO 8601 UTC with microseconds, formatted by PostgreSQL.
struct RunRecord {
    std::string id;
    std::string created_at;
    std::string filename;
    int width = 0;
    int height = 0;
    std::size_t bytes = 0;                // size of the stored upload
    std::vector<std::string> result_ids;  // in creation order
};

// Byte fields hold binary data (std::string as a byte buffer).
struct NewResult {
    std::string id;
    std::string run_id;
    std::string params_json;
    std::string metrics_json;
    std::string superpixel_labels_png;
    std::string region_labels_png;
};

struct ResultRecord {
    std::string id;
    std::string run_id;
    std::string created_at;
    std::optional<std::string> saved_at;  // set once the result is saved to the experiment log
    std::string params_json;              // jsonb text, as stored
    std::string metrics_json;
};

// Everything needed to re-render a result's images without recomputing it.
struct ResultArtifacts {
    std::string image;  // the run's encoded upload
    std::string superpixel_labels_png;
    std::string region_labels_png;
    int max_dim = 0;
};

enum class SaveOutcome { Saved, AlreadySaved, NotFound };

struct SaveReply {
    SaveOutcome outcome = SaveOutcome::NotFound;
    std::optional<ResultRecord> record;
};

// PostgreSQL access over a fixed-size pool of libpq connections, safe to call from many
// threads. Every statement is idempotent, so one lost connection is retried once on a fresh
// one; callers see Unavailable only when the database stays unreachable.
class Database {
public:
    // conninfo: a libpq URL or key=value string.
    Database(const std::string& conninfo, std::size_t pool_size);
    ~Database();
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&&) = delete;
    Database& operator=(Database&&) = delete;

    // Applies the embedded schema in one transaction, serialized across servers.
    void migrate();
    void ping();

    RunRecord insert_run(const std::string& id, const std::string& filename, int width, int height,
                         const std::string& image);
    std::optional<RunRecord> find_run(const std::string& id);
    std::optional<std::string> run_image(const std::string& run_id);

    // nullopt when the run does not exist.
    std::optional<ResultRecord> insert_result(const NewResult& result);
    std::optional<ResultRecord> find_result(const std::string& id);
    std::optional<ResultArtifacts> result_artifacts(const std::string& id);

    // Sets saved_at only if it is still NULL, so repeated saves keep the first timestamp.
    SaveReply save_result(const std::string& id);
    // Deletes every saved result; returns how many. Unsaved results are kept.
    std::size_t delete_saved();
    // Saved results, most recently saved first; run_id optionally restricts to one run.
    std::vector<ResultRecord> list_saved(int limit, const std::optional<std::string>& run_id);

private:
    struct Pool;
    std::unique_ptr<Pool> pool_;
};

}  // namespace seg::db
