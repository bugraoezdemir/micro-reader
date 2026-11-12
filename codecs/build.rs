//! Builds the C decoders micro-reader-codecs links statically (vendor/):
//!
//! - libjpeg-turbo, through CMake, with its SIMD (x86 needs NASM; ARM's NEON is C).
//!   `MICRO_READER_REQUIRE_SIMD=1` (the wheel builds) fails when the SIMD cannot be
//!   built; otherwise a build without NASM gets the plain C code and a warning.
//!   `NASM` names the assembler when it is not on PATH.
//! - jxrlib (JPEG XR), with cc: plain C.
//! - the two shims in src/c that call them.
//!
//! Both are cargo features (default on): `--no-default-features` builds without them,
//! and micro-reader falls back to its other decoders.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/c");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-env-changed=MICRO_READER_REQUIRE_SIMD");
    println!("cargo:rerun-if-env-changed=NASM");
    if env::var_os("CARGO_FEATURE_LIBJPEG_TURBO").is_some() {
        libjpeg_turbo();
    }
    if env::var_os("CARGO_FEATURE_JXRLIB").is_some() {
        jxrlib();
    }
}

fn libjpeg_turbo() {
    let require = env::var("MICRO_READER_REQUIRE_SIMD").map(|v| v == "1").unwrap_or(false);
    let target = env::var("TARGET").unwrap();
    let msvc = target.contains("msvc");
    let mut cfg = cmake::Config::new("vendor/libjpeg-turbo");
    cfg.profile("Release")
        .define("ENABLE_SHARED", "OFF")
        .define("ENABLE_STATIC", "ON")
        .define("WITH_TURBOJPEG", "OFF")
        .define("WITH_TOOLS", "OFF")
        .define("WITH_TESTS", "OFF")
        .define("WITH_JNA", "OFF")
        .define("WITH_FUZZ", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("REQUIRE_SIMD", if require { "ON" } else { "OFF" })
        // the C runtime Rust links on Windows (/MD)
        .define("WITH_CRT_DLL", "ON");
    if let Ok(nasm) = env::var("NASM") {
        cfg.define("CMAKE_ASM_NASM_COMPILER", nasm);
    }
    if target.starts_with("x86_64-apple") {
        // cross-built on an arm64 runner: tell CMake (and NASM's object format)
        cfg.define("CMAKE_SYSTEM_PROCESSOR", "x86_64")
            .define("CMAKE_OSX_ARCHITECTURES", "x86_64");
    }
    if msvc {
        // Visual Studio's generator nests deep (CMakeFiles/CMakeScratch/TryCompile-*/...):
        // under a long OUT_DIR (a pip build in %TEMP%, a deep checkout) it passes
        // Windows' 260-character limit.  Build in a short folder then.
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        if out.as_os_str().len() > 90 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            out.hash(&mut h);
            cfg.out_dir(env::temp_dir().join(format!("mrc-{:016x}", h.finish())));
        }
    }
    // the static library only (no install step: it wants the docs trimmed from vendor/)
    let dst = cfg.build_target("jpeg-static").build();
    let build = dst.join("build");
    for dir in [build.join("Release"), build.clone()] {     // multi-config generators: Release/
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static={}", if msvc { "jpeg-static" } else { "jpeg" });
    if !simd_built(&build) {
        let text = "libjpeg-turbo was built without SIMD (no NASM?): JPEG decodes \
                    several times slower";
        if require {
            panic!("{text}");
        }
        println!("cargo:warning={text}");
    }

    cc::Build::new()
        .file("src/c/jpeg_shim.c")
        .include("vendor/libjpeg-turbo/src")          // jpeglib.h, jmorecfg.h, jerror.h
        .include(&build)                               // the generated jconfig.h
        .warnings(false)
        .opt_level(2)
        .compile("mr_jpeg_shim");
}

/// Whether libjpeg-turbo built its SIMD extensions: object files under build/simd
/// (when NASM or the intrinsics are missing, CMake leaves that folder without any).
fn simd_built(build: &Path) -> bool {
    fn objects(dir: &Path) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else { return false };
        entries.flatten().any(|e| {
            let p = e.path();
            if p.is_dir() {
                objects(&p)
            } else {
                matches!(p.extension().and_then(|x| x.to_str()), Some("o" | "obj"))
            }
        })
    }
    objects(&build.join("simd"))
}

fn jxrlib() {
    let root = PathBuf::from("vendor/jxrlib");
    // the codec core, decoder and glue as one unit (src/c/jxr_unity_dec.c: inlining
    // across files); the encoder as it is
    let files = [
        "image/encode/encode.c",
        "image/encode/segenc.c",
        "image/encode/strenc.c",
        "image/encode/strFwdTransform.c",
        "image/encode/strPredQuantEnc.c",
    ];
    let mut build = cc::Build::new();
    build
        .files(files.iter().map(|f| root.join(f)))
        .file("src/c/jxr_shim.c")
        .file("src/c/jxr_unity_dec.c")
        .include(&root)
        .include(root.join("common/include"))
        .include(root.join("image/sys"))
        .include(root.join("jxrgluelib"))
        .define("__ANSI__", None)
        .define("DISABLE_PERF_MEASUREMENT", None)
        // as a CMake Release build (libCZI's, imagecodecs'): jxrlib's bit reader asserts
        // on every symbol it reads (strcodec.h) unless NDEBUG is set; cc does not set it
        .define("NDEBUG", None)
        .warnings(false)
        .opt_level(3);
    for flag in [
        "-Wno-constant-conversion", "-Wno-unused-const-variable", "-Wno-deprecated-declarations",
        "-Wno-comment", "-Wno-unused-value", "-Wno-unused-function", "-Wno-unknown-pragmas",
        "-Wno-extra-tokens", "-Wno-missing-field-initializers", "-Wno-shift-negative-value",
        "-Wno-dangling-else", "-Wno-sign-compare", "-Wno-strict-aliasing",
        "-Wno-implicit-fallthrough", "-Wno-old-style-declaration", "-Wno-endif-labels",
        "-Wno-parentheses", "-Wno-misleading-indentation", "-Wno-unused-but-set-variable",
        "-Wno-incompatible-pointer-types", "-Wno-int-conversion",
    ] {
        build.flag_if_supported(flag);
    }
    if env::var("TARGET").unwrap().contains("msvc") {
        // whole-program optimisation, as imagecodecs' jxrlib has: link.exe then links
        // with /LTCG (it says so once); with the unity build, JPEG XR decodes at
        // 0.90-0.95x imagecodecs per tile (0.82x without either, 2026-11-10)
        build.flag("/GL");
        println!("cargo:rustc-link-arg=/LTCG");     // said up front: no restart, no notice
    }
    build.compile("mr_jxrlib");
}
