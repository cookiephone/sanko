//! Embeds a fingerprint of the solver source in the binary (`SANKO_SOURCE`), so
//! that every run log names the exact source it was built from. The hash is
//! 64-bit FNV-1a over the workspace manifest and each crate's manifest and
//! `src/` files, in sorted path order, with carriage returns removed so that
//! Windows and Unix checkouts of the same source give the same value.

use std::fs;
use std::path::{Path, PathBuf};

const CRATES: [&str; 3] = ["sanko-core", "sanko-search", "sanko-cli"];

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest.join("..").join("..");
    let mut files = vec![root.join("Cargo.toml")];
    for name in CRATES {
        let dir = root.join("crates").join(name);
        files.push(dir.join("Cargo.toml"));
        println!("cargo:rerun-if-changed={}", dir.join("Cargo.toml").display());
        println!("cargo:rerun-if-changed={}", dir.join("src").display());
        collect(&dir.join("src"), &mut files);
    }
    println!("cargo:rerun-if-changed={}", root.join("Cargo.toml").display());

    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| {
            let rel = p.strip_prefix(&root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
            (rel, p)
        })
        .collect();
    named.sort();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for (rel, path) in &named {
        let text: Vec<u8> = fs::read(path).expect("read source file").into_iter().filter(|&b| b != b'\r').collect();
        eat(rel.as_bytes());
        eat(&[0]);
        eat(&text);
        eat(&[0]);
    }
    println!("cargo:rustc-env=SANKO_SOURCE={hash:016x}");
}
