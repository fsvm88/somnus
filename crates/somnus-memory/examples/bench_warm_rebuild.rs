// Throwaway benchmark: times `SemanticStore::rebuild_from_recent` against a real,
// production-scale cold archive, through the same `somnus_metrics::timed_core!(SleepWarmRebuild, ..)`
// wrapper `sleep.rs` uses at its real call site — to verify the parallel_insert_slice change
// (vs. the prior serial-insert loop) live, not just via `cargo build`.
//
// Usage: cargo run --release --features fine-metrics -p somnus-memory --example bench_warm_rebuild -- <cold_archive_dir> <n>

use somnus_memory::cold::{ColdArchive, COLD_DIM};
use somnus_memory::semantic::SemanticStore;

fn main() {
    let mut args = std::env::args().skip(1);
    let cold_dir = args.next().expect("usage: bench_warm_rebuild <cold_dir> <n>");
    let n: usize = args
        .next()
        .expect("usage: bench_warm_rebuild <cold_dir> <n>")
        .parse()
        .expect("n must be a positive integer");

    let cold = ColdArchive::open(std::path::Path::new(&cold_dir)).expect("failed to open cold archive");
    println!("cold.len() = {}", cold.len());
    assert_eq!(COLD_DIM, 640, "sanity check on cold dim const");

    let mut store = SemanticStore::new(COLD_DIM);
    let (loaded, elapsed) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::SleepWarmRebuild,
        store.rebuild_from_recent(&cold, n)
    );
    println!("loaded = {loaded}, elapsed = {elapsed:?}");
    println!("registry: {}", somnus_metrics::METRICS.snapshot_and_reset());
}
