extern crate rustc_version;
use rustc_version::{version_meta, Channel};

#[cfg(features = "llvm_asm")]
use {
    field_assembly::generate_macro_string,
    std::{env, fs, path::Path},
};

#[cfg(features = "llvm_asm")]
const NUM_LIMBS: usize = 8;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let is_nightly = version_meta().expect("nightly check failed").channel == Channel::Nightly;

    let _should_use_asm = cfg!(all(
        feature = "llvm_asm",
        target_feature = "bmi2",
        target_feature = "adx",
        target_arch = "x86_64"
    )) && is_nightly;

    #[cfg(features = "llvm_asm")]
    if should_use_asm {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let dest_path = Path::new(&out_dir).join("field_assembly.rs");
        fs::write(&dest_path, generate_macro_string(NUM_LIMBS)).unwrap();
        println!("cargo:rustc-cfg=use_asm");
    }
}
