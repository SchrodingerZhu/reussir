use std::slice;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_panic(message: *const u8, length: usize) -> ! {
    let slice = unsafe { slice::from_raw_parts(message, length) };
    let message = String::from_utf8_lossy(slice);
    let bt = backtrace::Backtrace::new();
    eprintln!("Panic: {}", message);
    eprintln!("Backtrace:\n{:#?}", bt);
    std::process::abort()
}

macro_rules! panic {
    () => {
        let msg = "runtime panic";
        $crate::panic::__reussir_panic(msg.as_bytes().as_ptr(), msg.len())
    };
    ($msg:expr) => {
        $crate::panic::__reussir_panic($msg.as_bytes().as_ptr(), $msg.len())
    };
    ($fmt:expr, $($arg:tt)*) => {
        let msg = format!($fmt, $($arg)*);
        $crate::panic::__reussir_panic(msg.as_bytes().as_ptr(), msg.len())
    };
}
pub(crate) use panic;
