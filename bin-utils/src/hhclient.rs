use clap::{Arg, Command};
use serde_json::Value;

use crate::get_sketch_params;
pub struct Options {
    pub alice: String,
    pub bob: String,
    pub num_clients: usize,
    pub num_bad_clients: usize,
    pub recovery_threshold: f32,
    pub client_reps: usize,
    pub client_sockets: usize,
    pub num_buckets: u16,
    pub batch_size: usize,
    pub log_level: tracing_core::Level,
}

impl Options {
    pub fn load_from_json(program_name: &str) -> Self {
        let matches = Command::new(program_name)
            .version("0.1")
            .arg(
                Arg::new("config")
                    .short('c')
                    .long("config")
                    .required(true)
                    .takes_value(true)
                    .help("json to get the client config"),
            )
            .get_matches();

        let filename = matches.value_of("config").unwrap();

        let json_data = &std::fs::read_to_string(filename).expect("Cannot open JSON file");
        let v: Value = serde_json::from_str(json_data).expect("Cannot parse JSON config");

        let alice = v["alice"].as_str().unwrap().to_string();
        let bob = v["bob"].as_str().unwrap().to_string();
        let num_clients = v["num_clients"].as_u64().unwrap() as usize;
        let num_bad_clients = v["num_bad_clients"].as_u64().unwrap() as usize;
        let recovery_threshold = v["recovery_threshold"].as_f64().unwrap() as f32;
        let batch_size = v["batch_size"].as_u64().unwrap() as usize;
        let (client_reps, num_buckets) = get_sketch_params(recovery_threshold, num_bad_clients);

        let log_level = match v["log_level"].as_str() {
            Some("debug") => tracing_core::Level::DEBUG,
            Some("info") => tracing_core::Level::INFO,
            Some("warn") => tracing_core::Level::WARN,
            Some("error") => tracing_core::Level::ERROR,
            _ => panic!("Invalid log level"),
        };
        Options {
            alice,
            bob,
            num_clients,
            num_bad_clients,
            recovery_threshold,
            client_reps,
            client_sockets: 32,
            log_level,
            num_buckets,
            batch_size,
        }
    }
}
