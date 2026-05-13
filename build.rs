use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SWCLI_MODELS_DEV_API_JSON");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let out_path = out_dir.join("models_dev_api.json");
    let build_cache_path = profile_dir(&out_dir).join("models_dev_api.json");

    let data = match env::var("SWCLI_MODELS_DEV_API_JSON") {
        Ok(path) => fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read SWCLI_MODELS_DEV_API_JSON={path}: {err}")),
        Err(_) => match fetch_models_dev_catalog() {
            Ok(data) => {
                let _ = fs::write(&build_cache_path, &data);
                data
            }
            Err(fetch_err) => fs::read_to_string(&build_cache_path).unwrap_or_else(|cache_err| {
                panic!(
                    "failed to fetch {MODELS_DEV_URL}: {fetch_err}; also failed to read build cache {}: {cache_err}",
                    build_cache_path.display()
                )
            }),
        },
    };

    fs::write(&out_path, data)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_path.display()));
}

fn fetch_models_dev_catalog() -> Result<String, reqwest::Error> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build reqwest client");
    let response = client.get(MODELS_DEV_URL).send()?;
    let status = response.status();
    if !status.is_success() {
        panic!("failed to fetch {MODELS_DEV_URL}: HTTP {status}");
    }
    response.text()
}

fn profile_dir(out_dir: &std::path::Path) -> PathBuf {
    // OUT_DIR is target/<profile>/build/<pkg-hash>/out.
    out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has profile ancestor")
        .to_path_buf()
}
