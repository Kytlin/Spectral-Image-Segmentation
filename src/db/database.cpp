#include "db/database.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <libpq-fe.h>

#include "schema_sql.hpp"

namespace seg::db {

namespace {

struct ConnDeleter {
    void operator()(PGconn* c) const noexcept { PQfinish(c); }
};
using Conn = std::unique_ptr<PGconn, ConnDeleter>;

struct ResultDeleter {
    void operator()(PGresult* r) const noexcept { PQclear(r); }
};
using PgResult = std::unique_ptr<PGresult, ResultDeleter>;

constexpr int kText = 0;
constexpr int kBinary = 1;

// A statement parameter; a null value is SQL NULL. Text values must be NUL-terminated,
// which std::string guarantees.
struct Param {
    const std::string* value = nullptr;
    int format = kText;
};
Param text(const std::string& v) { return {&v, kText}; }
Param bytes(const std::string& v) { return {&v, kBinary}; }
Param null() { return {}; }

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// SQLSTATEs meaning the server is gone or refusing connections, as opposed to a bad statement:
// class 08 (connection exception), 57P01-57P05 (shutdown, cannot connect now), 53300 (too many
// connections).
bool is_unavailable(const std::string& sqlstate) {
    return sqlstate.compare(0, 2, "08") == 0 || sqlstate.compare(0, 3, "57P") == 0 || sqlstate == "53300";
}

std::string value(const PGresult* r, int row, int col) {
    return std::string(PQgetvalue(r, row, col), static_cast<std::size_t>(PQgetlength(r, row, col)));
}

std::optional<std::string> nullable(const PGresult* r, int row, int col) {
    if (PQgetisnull(r, row, col)) return std::nullopt;
    return value(r, row, col);
}

// Text form of a uuid[] such as {a,b}; UUIDs never need quoting.
std::vector<std::string> parse_uuid_array(const std::string& pg) {
    std::vector<std::string> out;
    if (pg.size() <= 2) return out;
    std::size_t start = 1;
    while (start < pg.size() - 1) {
        const std::size_t comma = std::min(pg.find(',', start), pg.size() - 1);
        out.push_back(pg.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

constexpr const char* kResultColumns =
    "id::text, run_id::text, iso_utc(created_at), iso_utc(saved_at), params::text, metrics::text";

ResultRecord read_result(const PGresult* r, int row) {
    ResultRecord rec;
    rec.id = value(r, row, 0);
    rec.run_id = value(r, row, 1);
    rec.created_at = value(r, row, 2);
    rec.saved_at = nullable(r, row, 3);
    rec.params_json = value(r, row, 4);
    rec.metrics_json = value(r, row, 5);
    return rec;
}

}  // namespace

// Connections are opened lazily and reopened after a failure, so the pool recovers on its
// own when the database comes back.
struct Database::Pool {
    Pool(std::string conninfo, std::size_t size) : conninfo_(std::move(conninfo)), idle_(size) {}

    // One statement with retry-once on connection loss; only for idempotent statements.
    PgResult exec(const char* sql, std::initializer_list<Param> params, int result_format = kText) {
        std::vector<const char*> values;
        std::vector<int> lengths;
        std::vector<int> formats;
        for (const Param& p : params) {
            values.push_back(p.value ? p.value->data() : nullptr);
            lengths.push_back(p.value ? static_cast<int>(p.value->size()) : 0);
            formats.push_back(p.format);
        }
        return run(true, [&](PGconn* c) {
            return PQexecParams(c, sql, static_cast<int>(values.size()), nullptr, values.data(), lengths.data(),
                                formats.data(), result_format);
        });
    }

    // Several statements in one implicit transaction (all or nothing); not retried.
    PgResult exec_script(const std::string& sql) {
        return run(false, [&](PGconn* c) { return PQexec(c, sql.c_str()); });
    }

private:
    // Returns its connection to the pool on every exit path, including exceptions.
    struct Checkout {
        Checkout(Pool& owner, Conn c) : pool(owner), conn(std::move(c)) {}
        ~Checkout() { pool.release(std::move(conn)); }
        Checkout(const Checkout&) = delete;
        Checkout& operator=(const Checkout&) = delete;
        Checkout(Checkout&&) = delete;
        Checkout& operator=(Checkout&&) = delete;
        Pool& pool;
        Conn conn;
    };

    template <class Send>
    PgResult run(bool retry, Send&& send) {
        Checkout co(*this, acquire());
        for (int attempt = 0;; ++attempt) {
            if (!co.conn || PQstatus(co.conn.get()) != CONNECTION_OK) co.conn = connect();
            PgResult r(send(co.conn.get()));
            const ExecStatusType status = r ? PQresultStatus(r.get()) : PGRES_FATAL_ERROR;
            if (status == PGRES_COMMAND_OK || status == PGRES_TUPLES_OK) return r;
            const std::string message = trim(r ? PQresultErrorMessage(r.get()) : PQerrorMessage(co.conn.get()));
            const char* state = r ? PQresultErrorField(r.get(), PG_DIAG_SQLSTATE) : nullptr;
            const std::string sqlstate = state ? state : "";
            if (PQstatus(co.conn.get()) == CONNECTION_BAD || is_unavailable(sqlstate)) {
                co.conn.reset();
                if (retry && attempt == 0) continue;
                throw Unavailable("database unavailable: " + message);
            }
            throw Error(sqlstate, message);
        }
    }

    Conn connect() const {
        const char* keys[] = {"dbname", "connect_timeout", "application_name", "options", nullptr};
        const char* vals[] = {conninfo_.c_str(), "3", "seg_server", "-c statement_timeout=30000", nullptr};
        Conn c(PQconnectdbParams(keys, vals, 1));
        if (!c || PQstatus(c.get()) != CONNECTION_OK) {
            throw Unavailable("cannot connect to database: " + trim(c ? PQerrorMessage(c.get()) : "out of memory"));
        }
        return c;
    }

    Conn acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [this] { return !idle_.empty(); });
        Conn c = std::move(idle_.back());
        idle_.pop_back();
        return c;
    }

    void release(Conn c) noexcept {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            idle_.push_back(std::move(c));
        }
        available_.notify_one();
    }

    const std::string conninfo_;
    std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Conn> idle_;  // null entries are slots whose connection is not open yet
};

Database::Database(const std::string& conninfo, std::size_t pool_size)
    : pool_(std::make_unique<Pool>(conninfo, pool_size == 0 ? 1 : pool_size)) {}

Database::~Database() = default;

void Database::migrate() {
    // The advisory lock serializes concurrent migrations; it is released when the implicit
    // transaction of this multi-statement query ends.
    pool_->exec_script(std::string("SELECT pg_advisory_xact_lock(4242);\n") + kSchemaSql);
}

void Database::ping() { pool_->exec("SELECT 1", {}); }

RunRecord Database::insert_run(const std::string& id, const std::string& filename, int width, int height,
                               const std::string& image) {
    const std::string w = std::to_string(width);
    const std::string h = std::to_string(height);
    // ON CONFLICT makes a retry after a lost commit acknowledgement a no-op.
    pool_->exec("INSERT INTO runs (id, filename, width, height, image) "
                "VALUES ($1::uuid, $2, $3::int, $4::int, $5) ON CONFLICT (id) DO NOTHING",
                {text(id), text(filename), text(w), text(h), bytes(image)});
    std::optional<RunRecord> run = find_run(id);
    if (!run) throw Error("", "run " + id + " missing right after insert");
    return *run;
}

std::optional<RunRecord> Database::find_run(const std::string& id) {
    const PgResult r = pool_->exec(
        "SELECT id::text, iso_utc(created_at), filename, width, height, octet_length(image), "
        "ARRAY(SELECT x.id FROM results x WHERE x.run_id = runs.id ORDER BY x.created_at, x.id)::text "
        "FROM runs WHERE id = $1::uuid",
        {text(id)});
    if (PQntuples(r.get()) == 0) return std::nullopt;
    RunRecord run;
    run.id = value(r.get(), 0, 0);
    run.created_at = value(r.get(), 0, 1);
    run.filename = value(r.get(), 0, 2);
    run.width = std::stoi(value(r.get(), 0, 3));
    run.height = std::stoi(value(r.get(), 0, 4));
    run.bytes = std::stoul(value(r.get(), 0, 5));
    run.result_ids = parse_uuid_array(value(r.get(), 0, 6));
    return run;
}

std::optional<std::string> Database::run_image(const std::string& run_id) {
    const PgResult r = pool_->exec("SELECT image FROM runs WHERE id = $1::uuid", {text(run_id)}, kBinary);
    if (PQntuples(r.get()) == 0) return std::nullopt;
    return value(r.get(), 0, 0);
}

std::optional<ResultRecord> Database::insert_result(const NewResult& n) {
    try {
        pool_->exec("INSERT INTO results (id, run_id, params, metrics, superpixel_labels, region_labels) "
                    "VALUES ($1::uuid, $2::uuid, $3::jsonb, $4::jsonb, $5, $6) ON CONFLICT (id) DO NOTHING",
                    {text(n.id), text(n.run_id), text(n.params_json), text(n.metrics_json),
                     bytes(n.superpixel_labels_png), bytes(n.region_labels_png)});
    } catch (const Error& e) {
        if (e.sqlstate() == "23503") return std::nullopt;  // foreign_key_violation: no such run
        throw;
    }
    std::optional<ResultRecord> rec = find_result(n.id);
    if (!rec) throw Error("", "result " + n.id + " missing right after insert");
    return rec;
}

std::optional<ResultRecord> Database::find_result(const std::string& id) {
    const PgResult r =
        pool_->exec((std::string("SELECT ") + kResultColumns + " FROM results WHERE id = $1::uuid").c_str(), {text(id)});
    if (PQntuples(r.get()) == 0) return std::nullopt;
    return read_result(r.get(), 0);
}

std::optional<ResultArtifacts> Database::result_artifacts(const std::string& id) {
    const PgResult r = pool_->exec(
        "SELECT ru.image, re.superpixel_labels, re.region_labels, re.params->>'max_dim' "
        "FROM results re JOIN runs ru ON ru.id = re.run_id WHERE re.id = $1::uuid",
        {text(id)}, kBinary);
    if (PQntuples(r.get()) == 0) return std::nullopt;
    ResultArtifacts a;
    a.image = value(r.get(), 0, 0);
    a.superpixel_labels_png = value(r.get(), 0, 1);
    a.region_labels_png = value(r.get(), 0, 2);
    a.max_dim = std::stoi(value(r.get(), 0, 3));
    return a;
}

SaveReply Database::save_result(const std::string& id) {
    // Row-locked single UPDATE: concurrent saves serialize, and only the first sees NULL.
    const PgResult r =
        pool_->exec("UPDATE results SET saved_at = now() WHERE id = $1::uuid AND saved_at IS NULL", {text(id)});
    const bool updated = std::strcmp(PQcmdTuples(r.get()), "1") == 0;
    SaveReply reply;
    reply.record = find_result(id);
    if (!reply.record) reply.outcome = SaveOutcome::NotFound;
    else reply.outcome = updated ? SaveOutcome::Saved : SaveOutcome::AlreadySaved;
    return reply;
}

std::vector<ResultRecord> Database::list_saved(int limit, const std::optional<std::string>& run_id) {
    const std::string lim = std::to_string(limit);
    const PgResult r = pool_->exec(
        (std::string("SELECT ") + kResultColumns +
         " FROM results WHERE saved_at IS NOT NULL AND ($2::uuid IS NULL OR run_id = $2::uuid) "
         "ORDER BY saved_at DESC, id DESC LIMIT $1::int")
            .c_str(),
        {text(lim), run_id ? text(*run_id) : null()});
    std::vector<ResultRecord> out;
    const int n = PQntuples(r.get());
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) out.push_back(read_result(r.get(), i));
    return out;
}

}  // namespace seg::db
