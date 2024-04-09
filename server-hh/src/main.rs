use bin_utils::hhserver::Options;
use bridge::{client_server::ClientsPool, id_tracker::IdGen, mpc_conn::MpcConnection};
use common::{
    group::IntModN,
    grouptest::{general_binary_split_test, ClientProofTag},
    prg::Prf,
    utils::transpose_without_clone,
    Group, VERIFY_KEY_SIZE,
};
use core::panic;
use hhcore::{
    bucket::Bucket,
    countsketch::CountSketch,
    protocol::{BCSketch, KeyChain},
    AggRing
};
use rand::Rng;
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serialize::UseSerde;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::Write,
    time::{Duration, Instant},
};
use tokio::net::TcpListener;
use tracing::info;
use typenum::Unsigned;

fn true_heavy_hitters(recovery_threshold: f32, salt: &[u8]) -> HashSet<Vec<u8>> {
    let mut result = HashSet::new();
    let n = match recovery_threshold {
        x if (0.09..0.11).contains(&x) => 1,
        x if (0.009..0.011).contains(&x) => 10,
        x if (0.0009..0.0011).contains(&x) => 100,
        _ => panic!("Unexpected recovery threshold"),
    };
    let mut hasher = Sha256::new();
    hasher.update(salt);
    for i in 0u64..n {
        let mut cur_hasher = hasher.clone();
        cur_hasher.update(i.to_be_bytes());
        let hash = cur_hasher.finalize().to_vec();
        result.insert(hash);
    }
    result
}

async fn aggregate_hhs<NumBuckets: Unsigned + std::fmt::Debug + Send + Sync>(
    client_pfs: Vec<ClientProofTag<String>>,
    client_data: Vec<Vec<Bucket<AggRing>>>,
    verify_key: &[u8; VERIFY_KEY_SIZE],
    peer: &MpcConnection,
    peer_idgen: &mut IdGen,
    num_bad_clients: usize,
) -> CountSketch<AggRing> {
    let (bad_ids, _comm) = general_binary_split_test(
        &client_pfs,
        verify_key,
        peer_idgen,
        peer,
        num_bad_clients,
        32,
    )
    .await;
    let aggregate_sketch;
    if bad_ids.len() == 0 {
        aggregate_sketch = CountSketch::from_bucket_vec(client_data.into_par_iter().reduce(
            || vec![Bucket::<AggRing>::zero(); NumBuckets::to_usize()],
            |mut acc_dat, dat| {
                for i in 0..dat.len() {
                    acc_dat[i].add(&dat[i]);
                }
                acc_dat
            },
        ));
    } else {
        println!("found {} bad indices", bad_ids.len());
        let blank_proof_tag = ClientProofTag::new(0, "".to_owned());
        let (buckets, _) = client_data
            .into_par_iter()
            .zip(client_pfs.into_par_iter())
            .reduce(
                || {
                    (
                        vec![Bucket::<AggRing>::zero(); NumBuckets::to_usize()],
                        blank_proof_tag.clone(),
                    )
                },
                |(mut acc_dat, _pf), (dat, pf_new)| {
                    if bad_ids.contains(&pf_new.testing_id) {
                        (acc_dat, blank_proof_tag.clone())
                    } else {
                        for i in 0..dat.len() {
                            acc_dat[i].add(&dat[i]);
                        }
                        (acc_dat, blank_proof_tag.clone())
                    }
                },
            );
        aggregate_sketch = CountSketch::from_bucket_vec(buckets);
        // println!("end summing");
    }

    aggregate_sketch
}

/// Takes in a small batch of client proofs at a time from the meta client, and validates + aggregates them batch by batch
async fn streaming_collect_and_aggregate<NumBuckets: Unsigned + std::fmt::Debug + Send + Sync>(
    options: &Options,
    verify_key: &[u8; VERIFY_KEY_SIZE],
    listener: &TcpListener,
    peer: &MpcConnection,
    peer_idgen: &mut IdGen,
) -> (Vec<CountSketch<AggRing>>, Duration, Duration, Duration) {
    let mut collection_time = Duration::ZERO;
    let mut expansion_time = Duration::ZERO;
    let mut aggregation_time = Duration::ZERO;
    let mut global_aggregates =
        vec![CountSketch::<AggRing>::new(NumBuckets::to_usize()); options.client_reps];

    let mut total_client_bytes = 0;
    let mut final_batch_time = Duration::ZERO;

    let mut total_clients_seen = 0;

    let mut client_idgen = IdGen::new();
    let clients = ClientsPool::new(options.client_sockets, listener).await; // Set up client connections for the current batch, and accept stuff
    while total_clients_seen < options.num_clients {
        info!("-------------------------------------------------------");
        let cur_collection = Instant::now();

        let client_keys_separated = clients
            .subscribe_and_get::<UseSerde<Vec<Vec<KeyChain<AggRing>>>>>(client_idgen.next_recv_id())
            .await
            .unwrap();

        let client_keys = client_keys_separated
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        total_clients_seen += client_keys.len();
        info!("{} clients seen", total_clients_seen);
        let bad_clients_per_batch = std::cmp::max(
            options.num_bad_clients * client_keys.len() / options.num_clients,
            2,
        );
        total_client_bytes += clients.num_bytes_received_from_all();
        collection_time += cur_collection.elapsed();
        info!("Finished collection");
        let mut cur_expansion_time = Duration::ZERO;
        let mut cur_aggregation_time = Duration::ZERO;
        let client_keys = transpose_without_clone(client_keys);
        let mut batch_aggregates = Vec::with_capacity(options.client_reps);
        for keys in client_keys {
            let now = Instant::now();
            // Eval every key into a sketch and a proof.
            let mut buckets_and_proofs =
                keys.into_par_iter()
                    .map(|key| {
                        let mut prf = Prf::new(verify_key);
                        let testing_id = prf.compute_prf(&key.id.to_le_bytes());
                        let (bucket_vec, tag) =
                            <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<
                                IntModN<NumBuckets>,
                            >(key);
                        (bucket_vec, ClientProofTag::new(testing_id, tag))
                    })
                    .collect::<Vec<_>>();
            buckets_and_proofs.par_sort_by(|a, b| a.1.testing_id.cmp(&b.1.testing_id));
            let (buckets, proofs) = buckets_and_proofs.into_par_iter().unzip();
            cur_expansion_time += now.elapsed();
            let now = Instant::now();
            let aggregate = aggregate_hhs::<NumBuckets>(
                proofs,
                buckets,
                verify_key,
                peer,
                peer_idgen,
                bad_clients_per_batch,
            )
            .await;
            cur_aggregation_time += now.elapsed();
            batch_aggregates.push(aggregate);
        }

        global_aggregates
            .iter_mut()
            .zip(batch_aggregates.into_iter())
            .for_each(|(a, b)| {
                a.insert(&b);
            });

        if total_clients_seen == options.num_clients {
            final_batch_time += cur_aggregation_time + cur_expansion_time;
        }
        aggregation_time += cur_aggregation_time;
        expansion_time += cur_expansion_time;
        info!("Expansion time: {:.2?}", expansion_time);
        info!("Aggregation time: {:.2?}", aggregation_time);
        std::io::stdout().flush().unwrap();
    }
    println!("final batch time: {:?}", final_batch_time);
    println!("Total bytes received from clients: {}", total_client_bytes);
    (
        global_aggregates,
        collection_time,
        expansion_time,
        aggregation_time,
    )
}

/// Accept all client submissions, one run at a time.
async fn batch_collect_and_aggregate_per_run<
    NumBuckets: Unsigned + std::fmt::Debug + Send + Sync,
>(
    options: &Options,
    listener: &TcpListener,
    verify_key: &[u8; VERIFY_KEY_SIZE],
    peer: &MpcConnection,
    peer_idgen: &mut IdGen,
) -> (Vec<CountSketch<AggRing>>, Duration, Duration, Duration) {
    let mut client_idgen = IdGen::new();
    let clients = ClientsPool::new(options.client_sockets, &listener).await;

    let mut collection_time = Duration::ZERO;
    let mut expansion_time = Duration::ZERO;
    let mut aggregation_time = Duration::ZERO;
    info!("Client connection established");
    let mut global_aggregates = Vec::with_capacity(options.client_reps);
    for i in 0..options.client_reps {
        // Each element of this array gets flattened into a `num_clients` long vector of keys
        let now = Instant::now();
        let client_keys = clients
            .subscribe_and_get::<UseSerde<Vec<KeyChain<AggRing>>>>(client_idgen.next_recv_id())
            .await
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        collection_time += now.elapsed();
        info!(
            "Starting expansion for run {}, num keys {}",
            i,
            client_keys.len()
        );

        let now = Instant::now();
        let mut buckets_and_proofs = client_keys
            .into_par_iter()
            .map(|key| {
                let mut prf = Prf::new(verify_key);
                let testing_id = prf.compute_prf(&key.id.to_le_bytes());
                let (bucket_vec, tag) = <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<
                    IntModN<NumBuckets>,
                >(key);
                (bucket_vec, ClientProofTag::new(testing_id, tag))
            })
            .collect::<Vec<_>>();
        buckets_and_proofs.par_sort_by(|a, b| a.1.testing_id.cmp(&b.1.testing_id));
        let (buckets, proofs) = buckets_and_proofs.into_par_iter().unzip();
        expansion_time += now.elapsed();

        info!("Starting aggregation, time is {:.2?}", now.elapsed());
        let now = Instant::now();
        let cur_aggregate = aggregate_hhs::<NumBuckets>(
            proofs,
            buckets,
            verify_key,
            peer,
            peer_idgen,
            options.num_bad_clients,
        )
        .await;
        global_aggregates.push(cur_aggregate);
        aggregation_time += now.elapsed();
    }
    println!(
        "Total bytes received from clients {:.2?}",
        clients.num_bytes_received_from_all()
    );
    (
        global_aggregates,
        collection_time,
        expansion_time,
        aggregation_time,
    )
}

async fn main_with_options<NumBuckets: Unsigned + std::fmt::Debug + Send + Sync>(
    options: &Options,
) {
    tracing_subscriber::fmt()
        .pretty()
        .with_max_level(options.log_level)
        .init();
    // establish connection with the other server
    let peer = if options.is_bob {
        // I'm Bob and need a complete address of alice.
        MpcConnection::new_as_bob(&options.mpc_addr, options.num_mpc_sockets).await
    } else {
        // I'm Alice and I need a port number of alice.
        let mpc_addr =
            u16::from_str_radix(&options.mpc_addr, 10).expect("invalid mpc_addr as port");
        MpcConnection::new_as_alice(mpc_addr, options.num_mpc_sockets).await
    };

    info!("Peer connection set up!");

    // establish connections with the meta client
    info!("My client port: {}", options.client_port);
    let listener = TcpListener::bind(("0.0.0.0", options.client_port))
        .await
        .unwrap();

    let salt = &vec![1u8; 32];

    let mut peer_idgen = IdGen::new();

    let verify_key = if options.is_alice() {
        // Broadcast a random verifier key to the other server
        let mut rng = rand::thread_rng();
        let key = rng.gen::<[u8; VERIFY_KEY_SIZE]>();
        peer.send_message(peer_idgen.next_send_id(), key.to_vec());
        key
    } else {
        let key = peer
            .subscribe_and_get::<UseSerde<Vec<u8>>>(peer_idgen.next_recv_id())
            .await;
        key.unwrap().to_vec().try_into().unwrap()
    };

    let e2etime = Instant::now();

    let collection_time;
    let expansion_time;
    let aggregation_time;
    let mut global_aggregates;

    (
        global_aggregates,
        collection_time,
        expansion_time,
        aggregation_time,
    ) = if cfg!(feature = "streaming") {
        streaming_collect_and_aggregate::<NumBuckets>(
            &options,
            &verify_key,
            &listener,
            &peer,
            &mut peer_idgen,
        )
        .await
    } else {
        batch_collect_and_aggregate_per_run::<NumBuckets>(
            &options,
            &listener,
            &verify_key,
            &peer,
            &mut peer_idgen,
        )
        .await
    };

    // Give the other server time to catch up. Helps make sure both servers terminate together.
    std::thread::sleep(Duration::from_secs(3));
    let now = Instant::now();
    let received_sketches = peer
        .exchange_message(
            peer_idgen.next_exchange_id(),
            UseSerde(global_aggregates.clone()),
        )
        .await
        .unwrap();

    let sketch_recovery_threshold =
        (options.num_clients as f32 * options.recovery_threshold) as AggRing;

    let mut global_recovered = HashSet::new();
    global_aggregates
        .iter_mut()
        .zip(received_sketches.iter())
        .enumerate()
        .for_each(|(i, (my_sketch, received_sketch))| {
            my_sketch.insert(received_sketch);
            let recovered = my_sketch.recover(sketch_recovery_threshold, i);
            global_recovered.extend(recovered);
        });

    let true_hhs = true_heavy_hitters(options.recovery_threshold, salt);
    let mut true_recovered = 0;
    for s in &global_recovered {
        if true_hhs.contains(s) {
            true_recovered += 1;
        }
    }
    let compute_time = now.elapsed();

    println!(
        "# recovered, # false positives, # missed heavy hitters, collection time, expansion time, aggregation time, recover hhs from sketch time, total server compute time, total time including collection, bytes exchanged"
    );
    println!(
        "{}, {}, {}, {:.2?}, {:.2?}, {:.2?}, {:.2?}, {:.2?}, {:.2?}, {}",
        global_recovered.len(),
        global_recovered.len() - true_recovered,
        true_hhs.len() - true_recovered,
        collection_time,
        expansion_time,
        aggregation_time,
        compute_time,
        compute_time + aggregation_time + expansion_time,
        e2etime.elapsed(),
        peer.num_bytes_sent()
    );

    std::thread::sleep(Duration::from_secs(4));
}

#[tokio::main]
pub async fn main() {
    let options = Options::load_from_json("SV2 Server");
    match options.num_buckets {
        2048 => main_with_options::<typenum::U2048>(&options).await,
        1024 => main_with_options::<typenum::U1024>(&options).await,
        512 => main_with_options::<typenum::U512>(&options).await,
        256 => main_with_options::<typenum::U256>(&options).await,
        64 => main_with_options::<typenum::U64>(&options).await,
        32 => main_with_options::<typenum::U32>(&options).await,
        _ => panic!("unsupported number of buckets"),
    }
}
