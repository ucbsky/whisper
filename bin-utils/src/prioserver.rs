use clap::{Arg, Command};
use serde_json::Value;

use crate::AggFunc;

pub struct Options {
    pub client_port: u16,
    pub num_clients: usize,
    pub is_bob: bool, // TODO: For consistency, change this to agg_id
    pub mpc_addr: String,
    pub num_mpc_sockets: usize,
    pub num_bad_clients: usize,
    pub agg_fn: AggFunc,
    pub chunk_size: u32,
    pub vec_size: u32,
    pub single_tag: bool,
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
        let agg_fn = match v["agg_fn"].as_str().expect("Can't parse agg_fn") {
            "sv" => AggFunc::SumVec,
            "hs" => AggFunc::Average,
            "av" => AggFunc::Average,
            _ => panic!("Invalid aggregation function"),
        };
        let chunk_size = v["chunk_size"].as_u64().expect("Can't parse chunk_size") as u32;
        let vec_size = v["vec_size"].as_u64().expect("Can't parse vec_size") as u32;
        let single_tag = v["single_tag"].as_bool().expect("Can't parse single_tag");
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
            is_bob,
            mpc_addr,
            num_mpc_sockets,
            num_bad_clients,
            agg_fn,
            chunk_size,
            vec_size,
            single_tag,
            log_level,
        }
    }

    #[inline]
    pub fn is_alice(&self) -> bool {
        !self.is_bob
    }
}
