use clap::{Arg, Command};
use serde_json::Value;

use crate::get_sketch_params;

pub struct Options {
    pub client_port: u16,
    pub num_clients: usize,
    pub client_reps: usize,
    pub client_sockets: usize,
    pub is_bob: bool, 
    pub mpc_addr: String,
    pub num_mpc_sockets: usize,
    pub num_buckets: u16,
    pub recovery_threshold: f32,
    pub batch_size: usize,
    pub num_bad_clients: usize,
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

        let client_port = v["client_port"].as_u64().expect("Can't parse client_port") as u16;
        let num_clients = v["num_clients"].as_u64().expect("Can't parse num_clients") as usize;
        let is_bob = v["is_bob"].as_bool().expect("Can't parse is_bob");
        let mpc_addr = v["mpc_addr"]
            .as_str()
            .expect("Can't parse mpc_addr")
            .to_string();
        let num_mpc_sockets = v["num_mpc_sockets"]
            .as_u64()
            .expect("Can't parse num_mpc_sockets") as usize;
        let num_bad_clients = v["num_bad_clients"]
            .as_u64()
            .expect("Can't parse num_bad_clients") as usize;
        let recovery_threshold = v["recovery_threshold"]
            .as_f64()
            .expect("Can't parse recovery_threshold") as f32;

        let (client_reps, num_buckets) = get_sketch_params(recovery_threshold, num_bad_clients);

        let batch_size = v["batch_size"].as_u64().expect("Can't parse batch_size") as usize;

        let log_level = match v["log_level"].as_str() {
            Some("debug") => tracing_core::Level::DEBUG,
            Some("info") => tracing_core::Level::INFO,
            Some("warn") => tracing_core::Level::WARN,
            Some("error") => tracing_core::Level::ERROR,
            _ => panic!("Invalid log level"),
        };

        Options {
            client_port,
            num_clients,
            client_reps,
            client_sockets: 32,
            is_bob,
            mpc_addr,
            num_mpc_sockets,
            num_buckets,
            recovery_threshold,
            batch_size,
            num_bad_clients,
            log_level,
        }
    }
    #[inline]
    pub fn is_alice(&self) -> bool {
        !self.is_bob
    }
}
