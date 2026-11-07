-- Applied by seg_server at startup inside one transaction; every statement is idempotent.

-- One uploaded image. Rows are never updated.
CREATE TABLE IF NOT EXISTS runs (
    id          uuid        PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    filename    text        NOT NULL,
    width       integer     NOT NULL CHECK (width > 0),
    height      integer     NOT NULL CHECK (height > 0),
    image       bytea       NOT NULL  -- the encoded upload, byte for byte
);

-- One segmentation of a run. Everything except saved_at is written once at creation;
-- saved_at goes from NULL to a timestamp at most once (saving to the experiment log).
-- Clearing the experiment log deletes saved rows; unsaved rows cannot be deleted.
CREATE TABLE IF NOT EXISTS results (
    id                 uuid        PRIMARY KEY,
    run_id             uuid        NOT NULL REFERENCES runs (id),
    created_at         timestamptz NOT NULL DEFAULT now(),
    params             jsonb       NOT NULL,  -- every effective engine parameter
    metrics            jsonb       NOT NULL,  -- as returned by the engine, incl. compute timings
    superpixel_labels  bytea       NOT NULL,  -- 16-bit PNG label map of the working image
    region_labels      bytea       NOT NULL,  -- 16-bit PNG label map of the final regions
    saved_at           timestamptz
);

CREATE INDEX IF NOT EXISTS results_run_id_idx ON results (run_id, created_at);
CREATE INDEX IF NOT EXISTS results_saved_idx ON results (saved_at DESC, id) WHERE saved_at IS NOT NULL;

-- ISO 8601 UTC with microseconds, the format every API timestamp uses.
CREATE OR REPLACE FUNCTION iso_utc(t timestamptz) RETURNS text LANGUAGE sql STABLE STRICT AS
$$ SELECT to_char(t AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') $$;

CREATE OR REPLACE FUNCTION reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% rows cannot be updated or deleted', TG_TABLE_NAME
        USING ERRCODE = 'integrity_constraint_violation';
END $$;

CREATE OR REPLACE FUNCTION results_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.id <> OLD.id OR NEW.run_id <> OLD.run_id OR NEW.created_at <> OLD.created_at
       OR NEW.params <> OLD.params OR NEW.metrics <> OLD.metrics
       OR NEW.superpixel_labels <> OLD.superpixel_labels OR NEW.region_labels <> OLD.region_labels
       OR (OLD.saved_at IS NOT NULL AND NEW.saved_at IS DISTINCT FROM OLD.saved_at) THEN
        RAISE EXCEPTION 'results are immutable except for setting saved_at once'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION results_delete_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.saved_at IS NULL THEN
        RAISE EXCEPTION 'only saved results can be deleted (by clearing the experiment log)'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN OLD;
END $$;

CREATE OR REPLACE TRIGGER runs_immutable BEFORE UPDATE OR DELETE ON runs
    FOR EACH ROW EXECUTE FUNCTION reject_mutation();
CREATE OR REPLACE TRIGGER results_immutable BEFORE UPDATE ON results
    FOR EACH ROW EXECUTE FUNCTION results_immutable();
CREATE OR REPLACE TRIGGER results_no_delete BEFORE DELETE ON results
    FOR EACH ROW EXECUTE FUNCTION results_delete_guard();
