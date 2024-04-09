use std::collections::HashSet;
use std::time::{Duration, Instant};

use bin_utils::hhclient::Options;
use bridge::{client_server::batch_meta_clients, id_tracker::IdGen};
use common::utils::transpose_without_clone;
use common::Group;
use futures::stream::FuturesUnordered;
use hhcore::protocol::KeyChain;
use hhcore::utils::bytes_to_bucket;
use hhcore::{bucket::Bucket, protocol::BCSketch};
use rand::distributions::Distribution;
use rand::seq::IteratorRandom;
use rand::Rng;
use serialize::UseSerde;
use sha2::{Digest, Sha256};

use common::group::IntModN;
use hhcore::{get_sign_and_bkt, AggRing};
use tracing::info;
use typenum::Unsigned;
use zipf::ZipfDistribution;

const NUM_ELEMENTS: usize = 10000;

// Specify the population of client strings here.
fn string_population(salt: &[u8]) -> Vec<Vec<u8>> {
    let mut hasher = Sha256::new();
    hasher.update(salt);
    (0u64..NUM_ELEMENTS as u64)
        .map(|i| {
            let mut cur_hasher = hasher.clone();
            cur_hasher.update(i.to_be_bytes());
            cur_hasher.finalize().to_vec()
        })
        .collect()
}

fn sample_bucket<G: Group + Clone + PartialEq + std::fmt::Debug>(
    dist: &ZipfDistribution,
    rep_count: usize,
    population: &[Vec<u8>],
    num_buckets: usize,
    rng: &mut impl Rng,
) -> (Bucket<G>, u16) {
    let client_string_idx = rng.sample(dist);
    let client_string = &population[client_string_idx - 1];
    let (sign, bkt_idx) = get_sign_and_bkt(client_string, num_buckets, rep_count as u16);
    let bkt = bytes_to_bucket::<G>(&client_string, sign);
    (bkt, bkt_idx)
}

fn string_to_bkt_and_idx<G: Group + Clone + PartialEq + std::fmt::Debug>(
    client_string: &[u8],
    rep_count: usize,
    num_buckets: usize,
    _rng: &mut impl Rng,
) -> (Bucket<G>, u16) {
    let (sign, bkt_idx) = get_sign_and_bkt(&client_string, num_buckets, rep_count as u16);
    let bkt = bytes_to_bucket::<G>(&client_string, sign);
    (bkt, bkt_idx)
}

/// Generates num_keys keychain vectors, each of length client_reps.
/// Keys are sampled from dist
fn gen_keys<NumBuckets: Unsigned + std::fmt::Debug>(
    num_keys: usize,
    offset: usize,
    client_reps: usize,
    dist: &ZipfDistribution,
    bad_clients: &HashSet<usize>,
    string_population: &[Vec<u8>],
    _rng: &mut impl Rng,
) -> (Vec<Vec<KeyChain<AggRing>>>, Vec<Vec<KeyChain<AggRing>>>) {
    let mut all_bkt_idx = HashSet::new();
    let num_buckets = NumBuckets::to_usize();
    let (alice_keys, bob_keys): (Vec<_>, Vec<_>) = (0..num_keys)
        .into_iter()
        .map(|id| {
            let mut rng = rand::thread_rng();
            let client_seqnum = offset + id;
            if bad_clients.contains(&client_seqnum) {
                // Create a malformed input by generating keychains for two different locations
                let (alice_result, bob_result): (Vec<_>, Vec<_>) = (0..client_reps)
                    .map(|rep| {
                        let mut bkt_0 = Bucket::zero();
                        let mut bkt_1 = Bucket::zero();
                        let mut bkt_idx_0 = 0;
                        let mut bkt_idx_1 = 0;
                        while bkt_0 == bkt_1 {
                            (bkt_0, bkt_idx_0) = sample_bucket::<AggRing>(
                                &dist,
                                rep,
                                string_population,
                                num_buckets,
                                &mut rng,
                            );
                            (bkt_1, bkt_idx_1) = sample_bucket::<AggRing>(
                                &dist,
                                rep,
                                string_population,
                                num_buckets,
                                &mut rng,
                            );
                        }
                        let (alice_0, _bob_0) = bkt_0.clone().gen_key_chain(
                            client_seqnum as u128,
                            IntModN::<NumBuckets>::from_u16(bkt_idx_0),
                        );
                        let (_alice_1, bob_1) = bkt_1.clone().gen_key_chain(
                            client_seqnum as u128,
                            IntModN::<NumBuckets>::from_u16(bkt_idx_1),
                        );

                        (alice_0, bob_1)
                    })
                    .unzip();
                (alice_result, bob_result)
            } else {
                let client_string = &string_population[dist.sample(&mut rng) - 1];
                let (alice_result, bob_result): (Vec<_>, Vec<_>) = (0..client_reps)
                    .map(|rep| {
                        let (bkt, bkt_idx) = string_to_bkt_and_idx::<AggRing>(
                            client_string,
                            rep,
                            num_buckets,
                            &mut rng,
                        );
                        all_bkt_idx.insert(bkt_idx);
                        bkt.clone().gen_key_chain(
                            client_seqnum as u128,
                            IntModN::<NumBuckets>::from_u16(bkt_idx),
                        )
                    })
                    .unzip();
                (alice_result, bob_result)
            }
        })
        .unzip();
    (alice_keys, bob_keys)
}

async fn batch_send_measurements_per_run<
    NumBuckets: Unsigned + std::fmt::Debug + Send,
    const NUM_CORES: usize,
>(
    options: Options,
    bad_clients: HashSet<usize>,
    salt: &[u8],
    _rng: &mut impl Rng,
) -> (Duration, Duration) {
    let mut alice_idgens = (0..options.client_sockets)
        .map(|_| IdGen::new())
        .collect::<Vec<_>>();
    let mut bob_idgens = (0..options.client_sockets)
        .map(|_| IdGen::new())
        .collect::<Vec<_>>();
    let dist: ZipfDistribution = ZipfDistribution::new(NUM_ELEMENTS, 1.03).unwrap(); // same params as poplar

    let string_population = string_population(salt);
    // key[i][j][k] needs to be i'th run, j'th socket, k'th client
    let now = Instant::now();
    let (alice_keys, bob_keys): (Vec<_>, Vec<_>) = (0..options.client_sockets)
        .into_iter()
        .map(|i| {
            let mut rng = rand::thread_rng();
            let num_to_send = if i == options.client_sockets - 1 {
                options.num_clients
                    - (options.client_sockets - 1) * (options.num_clients / options.client_sockets)
            } else {
                options.num_clients / options.client_sockets
            };
            let (alice_keys, bob_keys) = gen_keys::<NumBuckets>(
                num_to_send,
                i * (options.num_clients / options.client_sockets),
                options.client_reps,
                &dist,
                &bad_clients,
                &string_population,
                &mut rng,
            );
            let alice_keys = transpose_without_clone(alice_keys);
            let bob_keys = transpose_without_clone(bob_keys);
            (alice_keys, bob_keys)
        })
        .unzip();

    let alice_keys = transpose_without_clone(alice_keys);
    let bob_keys = transpose_without_clone(bob_keys);

    let total_gen_time = now.elapsed();
    let conns = batch_meta_clients(options.client_sockets, 0, &options.alice, &options.bob).await;

    let now = Instant::now();
    let handles = FuturesUnordered::new();
    for (alice_rep_keys, bob_rep_keys) in alice_keys.into_iter().zip(bob_keys.into_iter()) {
        for (((i, (alice, bob)), alice_key_chunk), bob_key_chunk) in conns
            .iter()
            .enumerate()
            .zip(alice_rep_keys)
            .zip(bob_rep_keys)
        {
            let alice_idgen = &mut alice_idgens[i];
            let bob_idgen = &mut bob_idgens[i];
            let alice_tosend = (alice_key_chunk).to_vec();
            let bob_tosend = (bob_key_chunk).to_vec();
            handles.push(
                alice
                    .send_message(alice_idgen.next_send_id(), UseSerde(alice_tosend))
                    .unwrap(),
            );
            handles.push(
                bob.send_message(bob_idgen.next_send_id(), UseSerde(bob_tosend))
                    .unwrap(),
            );
        }
    }

    for h in handles {
        h.await.unwrap();
    }
    (total_gen_time, now.elapsed())
}

async fn stream_measurements<NumBuckets: Unsigned + std::fmt::Debug>(
    options: Options,
    bad_clients: HashSet<usize>,
    salt: &[u8],
    rng: &mut impl Rng,
) -> (Duration, Duration) {
    let start = Instant::now();
    let mut gen_time = Duration::ZERO;
    let mut total_sent = 0;
    let mut batch_id = 0;

    let dist = ZipfDistribution::new(NUM_ELEMENTS, 1.03).unwrap();
    let connections =
        batch_meta_clients(options.client_sockets, 0, &options.alice, &options.bob).await;

    let mut alice_idgens = (0..options.client_sockets)
        .map(|_| IdGen::new())
        .collect::<Vec<_>>();
    let mut bob_idgens = (0..options.client_sockets)
        .map(|_| IdGen::new())
        .collect::<Vec<_>>();

    let string_population = string_population(salt);
    while total_sent < options.num_clients {
        batch_id += 1;
        let now = Instant::now();
        let num_to_send = if total_sent == 0 && options.num_clients % options.batch_size != 0 {
            options.num_clients % options.batch_size
        } else {
            options.batch_size
        };
        let (alice_keys, bob_keys) = gen_keys::<NumBuckets>(
            num_to_send,
            total_sent,
            options.client_reps,
            &dist,
            &bad_clients,
            &string_population,
            rng,
        );
        gen_time += now.elapsed();
        info!("Generated {} keys", num_to_send);

        let handles = FuturesUnordered::new();
        for ((i, (alice, bob)), (alice_idgen, bob_idgen)) in connections
            .iter()
            .enumerate()
            .zip(alice_idgens.iter_mut().zip(bob_idgens.iter_mut()))
        {
            let (start, end) = if i == options.client_sockets - 1 {
                (num_to_send / options.client_sockets * i, num_to_send)
            } else {
                (
                    num_to_send / options.client_sockets * i,
                    num_to_send / options.client_sockets * (i + 1),
                )
            };
            let alice_keys_to_send = UseSerde(alice_keys[start..end].to_vec());
            let bob_keys_to_send = UseSerde(bob_keys[start..end].to_vec());
            handles.push(
                alice
                    .send_message(alice_idgen.next_send_id(), alice_keys_to_send)
                    .unwrap(),
            );
            handles.push(
                bob.send_message(bob_idgen.next_send_id(), bob_keys_to_send)
                    .unwrap(),
            );
        }

        total_sent += num_to_send;
        info!("Batch {batch_id} messages sent");
        for h in handles {
            h.await.unwrap();
        }
    }
    (gen_time, start.elapsed() - gen_time)
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

    let salt = vec![1u8; 32];

    println!("bad clients {:?}", bad_clients);
    const NUM_CORES: usize = 32;
    let (gen_time, send_time) = if cfg!(feature = "streaming") {
        match options.num_buckets {
            2048 => {
                stream_measurements::<typenum::U2048>(options, bad_clients, &salt, &mut rng).await
            }
            1024 => {
                stream_measurements::<typenum::U1024>(options, bad_clients, &salt, &mut rng).await
            }
            512 => {
                stream_measurements::<typenum::U512>(options, bad_clients, &salt, &mut rng).await
            }
            256 => {
                stream_measurements::<typenum::U256>(options, bad_clients, &salt, &mut rng).await
            }
            64 => stream_measurements::<typenum::U64>(options, bad_clients, &salt, &mut rng).await,
            32 => stream_measurements::<typenum::U64>(options, bad_clients, &salt, &mut rng).await,
            _ => panic!("unsupported number of buckets"),
        }
    } else {
        match options.num_buckets {
            2048 => {
                batch_send_measurements_per_run::<typenum::U2048, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            1024 => {
                batch_send_measurements_per_run::<typenum::U1024, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            512 => {
                batch_send_measurements_per_run::<typenum::U512, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            256 => {
                batch_send_measurements_per_run::<typenum::U256, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            64 => {
                batch_send_measurements_per_run::<typenum::U64, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            32 => {
                batch_send_measurements_per_run::<typenum::U32, NUM_CORES>(
                    options,
                    bad_clients,
                    &salt,
                    &mut rng,
                )
                .await
            }
            _ => panic!("unsupported number of buckets"),
        }
    };

    println!("gen time, send time");
    println!("{:.2?}, {:.2?}", gen_time, send_time);
}
