unsafe extern "C" {
    fn reussir_cache_probe_value() -> i32;
}

pub fn value() -> i32 {
    unsafe { reussir_cache_probe_value() }
}
