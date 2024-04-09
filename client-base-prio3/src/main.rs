use std::collections::HashSet;

use bin_utils::{prioclient::Options, AggFunc};
use bin_utils::{Prio3Gadgets, SumVecType, SEED_SIZE};
use bridge::{client_server::batch_meta_clients, id_tracker::IdGen};
use futures::stream::FuturesUnordered;
use prio::codec::Encode;
use prio::vdaf::prio3::Prio3;
use prio::vdaf::xof::XofShake128;
use prio::vdaf::{Client as PrioClient, VdafKey};
use rand::seq::IteratorRandom;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serialize::UseSerde;
use tracing::info;

const NUM_CORES: usize = 32;

async fn batch_send_measurements(
    options: Options,
    _bad_clients: HashSet<usize>,
    _rng: &mut impl Rng,
) {
    let prio3_len: usize = options.vec_size as usize;
    let prio3_chunk_len: usize = options.chunk_size as usize;

    let prio3: Prio3Gadgets = Prio3Gadgets::new(&options.agg_fn, prio3_len, prio3_chunk_len);

    let conns = batch_meta_clients(NUM_CORES, 0, &options.alice, &options.bob).await;

    let handles = FuturesUnordered::new();

    let all_keys = (0..NUM_CORES)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::thread_rng();
            let num_to_send = if i == NUM_CORES - 1 {
                options.num_clients - (NUM_CORES - 1) * (options.num_clients / NUM_CORES)
            } else {
                options.num_clients / NUM_CORES
            };

            let (alice_keys, bob_keys): (Vec<_>, Vec<_>) = (0..num_to_send)
                .map(|_| {
                    let nonce: [u8; 16] = rng.gen();
                    let (public_share, input_shares) = match options.agg_fn {
                        AggFunc::SumVec => {
                            let measurement = (0..prio3_len)
                                .map(|_| rng.gen::<u16>() as u128)
                                .collect::<Vec<_>>();

                            prio3
                                .prio3sv
                                .as_ref()
                                .unwrap()
                                .shard(&measurement, &nonce)
                                .unwrap()
                        }
                        AggFunc::Histogram => {
                            let measurement = (rng.gen::<u16>() % prio3_len as u16) as usize;

                            prio3
                                .prio3hist
                                .as_ref()
                                .unwrap()
                                .shard(&measurement, &nonce)
                                .unwrap()
                        }
                        AggFunc::Average => {
                            let measurement = rng.gen::<u16>() as u128;

                            prio3
                                .prio3avg
                                .as_ref()
                                .unwrap()
                                .shard(&measurement, &nonce)
                                .unwrap()
                        }
                    };

                    let alice_id = if i & 1 == 0 { 0 } else { 1 };
                    let bob_id = 1 - alice_id;

                    let alice_input = input_shares[alice_id].clone();
                    let bob_input = input_shares[bob_id].clone();

                    // Using SumVecType here rather than Hist or Avg is fine; doesn't matter
                    let alice_key = VdafKey::<Prio3<SumVecType, XofShake128, SEED_SIZE>> {
                        public_share: public_share.clone(),
                        input_share: alice_input,
                        nonce,
                        agg_id: alice_id,
                    };

                    let bob_key = VdafKey::<Prio3<SumVecType, XofShake128, SEED_SIZE>> {
                        public_share: public_share.clone(),
                        input_share: bob_input,
                        nonce,
                        agg_id: bob_id,
                    };

                    (alice_key.get_encoded(), bob_key.get_encoded())
                })
                .unzip();
            (alice_keys, bob_keys)
        })
        .collect::<Vec<_>>();

    info!("Generated keys");

    for (i, (alice, bob)) in conns.iter().enumerate() {
        let mut alice_idgen = IdGen::new();
        let mut bob_idgen = IdGen::new();
        handles.push(
            alice
                .send_message(alice_idgen.next_send_id(), UseSerde(all_keys[i].0.clone()))
                .unwrap(),
        );
        handles.push(
            bob.send_message(bob_idgen.next_send_id(), UseSerde(all_keys[i].1.clone()))
                .unwrap(),
        );

        info!("sent id {}", i);
    }

    for h in handles {
        h.await.unwrap();
    }
}

#[tokio::main]
pub async fn main() {
    let options = Options::load_from_json("SV2 Client");

    tracing_subscriber::fmt()
        .pretty()
        .with_max_level(options.log_level)
        .init();

    let mut rng = rand::thread_rng();
    let bad_clients = HashSet::<usize>::from_iter(
        (0..options.num_clients)
            .choose_multiple(&mut rng, options.num_bad_clients as usize)
            .iter()
            .cloned(),
    );
    batch_send_measurements(options, bad_clients, &mut rng).await;
}
