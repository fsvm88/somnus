mod e2e_bench;

fn main() {
    println!("End-to-end NREM step benchmark — burn+wgpu vs candle CPU");
    println!("N_IN={}, N_UNITS={}, K={}, BATCH={}", 896, 512, 25, 64);
    println!();
    e2e_bench::run();
}
