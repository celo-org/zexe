#[cfg(feature = "llvm_asm")]
use {
    field_assembly::generate_macro_string,
    std::{env, fs},
};

use std::path::{Path, PathBuf};

#[cfg(feature = "llvm_asm")]
const NUM_LIMBS: usize = 8;

fn replace_extension(file_name: &str, extension: &str) -> String {
    let mut new_name = PathBuf::from(Path::new(file_name));
    new_name.set_extension(extension);
    new_name.as_os_str().to_str().unwrap().to_string()
}

fn preprocess_and_replace_semicolon(file_name: &str, tag: &str) {
    const EXPANDED_SUFFIX: &str = "SCE.S";
    const WITHOUT_SEMICOLON_SUFFIX: &str = "SCR.S";
    let expanded = cc::Build::new()
        .file(file_name)
        .expand();
    let expanded_path = replace_extension(file_name, EXPANDED_SUFFIX);
    std::fs::write(&expanded_path, expanded).expect("Should have expanded");
    let replaced = std::fs::read_to_string(expanded_path).expect("Should have read expanded file").replace(";", "\n");
    let replaced_path = replace_extension(file_name, WITHOUT_SEMICOLON_SUFFIX);
    std::fs::write(&replaced_path, replaced).expect("Should have written replaced file");
    cc::Build::new()
        .file(replaced_path)
        .compile(tag);
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "llvm_asm")]
    {
        use rustc_version::{version_meta, Channel};

        let is_nightly = version_meta().expect("nightly check failed").channel == Channel::Nightly;

        let should_use_asm = cfg!(all(
            feature = "llvm_asm",
            target_feature = "bmi2",
            target_feature = "adx",
            target_arch = "x86_64"
        )) && is_nightly;

        if should_use_asm {
            let out_dir = env::var_os("OUT_DIR").unwrap();
            let dest_path = Path::new(&out_dir).join("field_assembly.rs");
            fs::write(&dest_path, generate_macro_string(NUM_LIMBS)).unwrap();
            println!("cargo:rustc-cfg=use_asm");
        }
    }

    let should_use_bw6_asm = cfg!(any(
        all(
            feature = "bw6_asm",
            target_feature = "bmi2",
            target_feature = "adx",
            target_arch = "x86_64"
        ),
        feature = "force_bw6_asm"
    ));
    if should_use_bw6_asm {
        cc::Build::new()
            .file("bw6-assembly/modmul768-sos1-adx.S")
            .compile("modmul768");
        cc::Build::new()
            .file("bw6-assembly/modadd768.S")
            .compile("modadd768");
        cc::Build::new()
            .file("bw6-assembly/modsub768.S")
            .compile("modsub768");
        println!("cargo:rustc-cfg=use_bw6_asm");
    }

    let should_use_bw6_asm_armv8 = cfg!(any(
        all(feature = "bw6_asm", target_arch = "aarch64"),
        feature = "force_bw6_asm_armv8"
    ));
    if should_use_bw6_asm_armv8 {
        preprocess_and_replace_semicolon("bw6-assembly/modmul768-armv8-kos.S", "modmul768");
        preprocess_and_replace_semicolon("bw6-assembly/modadd768-armv8.S", "modadd768");
        preprocess_and_replace_semicolon("bw6-assembly/modsub768-armv8.S", "modsub768");
        preprocess_and_replace_semicolon("bw6-assembly/modsqr768-armv8-kos.S", "modsqr768");
        println!("cargo:rustc-cfg=use_bw6_asm");
    }
}