use std::time::Duration;
use std::{io::Cursor, time::Instant};

use bin_utils::prioserver::Options;
use bin_utils::{AggFunc, Prio3Gadgets, AVG_BITS, F, SEED_SIZE};
use bridge::{client_server::ClientsPool, id_tracker::IdGen, mpc_conn::MpcConnection};
use common::grouptest::{general_binary_split_test, ClientProofTag};
use common::prg::Prf;
use common::VERIFY_KEY_SIZE;
use futures::stream::FuturesUnordered;
use prio::codec::{Encode, ParameterizedDecode};
use prio::field::FieldElement;
use prio::flp::Type;
use prio::vdaf::prio3::Prio3BatchedOutputShare;
use prio::vdaf::xof::Xof;
use prio::vdaf::{
    Aggregatable, AggregateShare, BatchAggregator, Collector, OutputShare, VdafError,
};
use prio::{
    field::Field128,
    vdaf::{prio3::Prio3, VdafBatchedKey},
};
use rand::{thread_rng, Rng};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use serialize::UseSerde;
use tokio::net::TcpListener;
use tracing::info;

const NUM_CORES: usize = 32;

fn prepare_encoded_key<T, P>(
    vdaf: &Prio3<T, P, SEED_SIZE>,
    verify_key: &[u8; SEED_SIZE],
    prf: &mut Prf,
    encoded_vdaf_key: &[u8],
) -> (
    ClientProofTag<Prio3BatchedOutputShare<T::Field>>,
    OutputShare<T::Field>,
)
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    let vdaf_key =
        VdafBatchedKey::decode_with_param(vdaf, &mut Cursor::new(&encoded_vdaf_key)).unwrap();

    let (tag, out_share) = vdaf.prepare_batched(verify_key, &vdaf_key, &()).unwrap();

    let testing_id = prf.compute_prf(&vdaf_key.client_id.to_le_bytes());
    // Instead of checking if the two server's tags are shares of 0, we negate one and check if they're equal.
    if vdaf_key.agg_id == 0 {
        (ClientProofTag::new(testing_id, tag), out_share)
    } else {
        let mut neg_tag = tag;
        neg_tag.output_share *= -T::Field::one();
        (ClientProofTag::new(testing_id, neg_tag), out_share)
    }
}

/// Given some encoded vdaf keys, deserialize them, generate proof tags, and run group testing to find the malicious entries.
/// Then, aggregate all of the well formed entries.
/// Returns (Aggregate output, # clients passed, verification time)
async fn run_vdaf_prepare(
    vdaf: Prio3Gadgets,
    verify_key: [u8; VERIFY_KEY_SIZE],
    vdaf_keys: Vec<Vec<u8>>,
    peer: MpcConnection,
    num_bad_clients: usize,
    mut peer_idgen: IdGen,
    agg_func: AggFunc,
) -> Result<(AggregateShare<F>, usize, Duration), VdafError> {
    let mut prf = Prf::new(&verify_key);
    let mut tags_and_shares = vdaf_keys
        .iter()
        .map(|encoded_vdaf_key| match agg_func {
            AggFunc::SumVec => {
                let vdaf_tmp = vdaf.prio3sv.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, encoded_vdaf_key)
            }
            AggFunc::Histogram => {
                let vdaf_tmp = vdaf.prio3hist.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, encoded_vdaf_key)
            }
            AggFunc::Average => {
                let vdaf_tmp = vdaf.prio3avg.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, encoded_vdaf_key)
            }
        })
        .collect::<Vec<_>>();

    tags_and_shares.sort_by(|a, b| a.0.testing_id.cmp(&b.0.testing_id));
    let (proof_tags, out_shares): (Vec<_>, Vec<_>) = tags_and_shares.into_iter().unzip();
    let now = Instant::now();
    let (bad_set, _split_test_comm) = general_binary_split_test(
        &proof_tags,
        &verify_key,
        &mut peer_idgen,
        &peer,
        num_bad_clients,
        16,
    )
    .await;

    let verif_time = now.elapsed();

    let mut clients_passed: usize = 0;
    let mut aggregate: Option<AggregateShare<Field128>> = None;
    for (share, tag) in out_shares.into_iter().zip(proof_tags.into_iter()) {
        if !bad_set.contains(&tag.testing_id) {
            clients_passed += 1;
            match aggregate {
                Some(ref mut inner) => inner.merge(&share.into()).unwrap(),
                None => aggregate = Some(share.into()),
            }
        }
    }

    Ok((
        aggregate.unwrap(),
        clients_passed,
        verif_time,
    ))
}

/// Same as run_vdaf_prepare, but uses rayon for parallelism
async fn run_vdaf_prepare_rayon(
    vdaf: Prio3Gadgets,
    verify_key: [u8; VERIFY_KEY_SIZE],
    vdaf_keys: Vec<Vec<u8>>,
    peer: MpcConnection,
    num_bad_clients: usize,
    mut peer_idgen: IdGen,
    agg_func: AggFunc,
) -> Result<(AggregateShare<F>, usize, usize, Duration), VdafError> {
    let start_comm = peer.num_bytes_sent();
    let mut tags_and_shares = vdaf_keys
        .into_par_iter()
        .map(|encoded_vdaf_key| {
            let mut prf = Prf::new(&verify_key);
            match agg_func {
            AggFunc::SumVec => {
                let vdaf_tmp = vdaf.prio3sv.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, &encoded_vdaf_key)
            }
            AggFunc::Histogram => {
                let vdaf_tmp = vdaf.prio3hist.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, &encoded_vdaf_key)
            }
            AggFunc::Average => {
                let vdaf_tmp = vdaf.prio3avg.as_ref().unwrap();
                prepare_encoded_key(vdaf_tmp, &verify_key, &mut prf, &encoded_vdaf_key)
            }
        }})
        .collect::<Vec<_>>();

    tags_and_shares.par_sort_by(|a, b| a.0.testing_id.cmp(&b.0.testing_id));
    let (proof_tags, out_shares): (Vec<_>, Vec<_>) = tags_and_shares.into_par_iter().unzip();
    let now = Instant::now();
    let (bad_set, _split_test_comm) = general_binary_split_test(
        &proof_tags,
        &verify_key,
        &mut peer_idgen,
        &peer,
        num_bad_clients,
        16,
    )
    .await;

    let verif_time = now.elapsed();

    let mut clients_passed: usize = 0;
    let mut aggregate: Option<AggregateShare<Field128>> = None;
    for (share, tag) in out_shares.into_iter().zip(proof_tags.into_iter()) {
        if !bad_set.contains(&tag.testing_id) {
            clients_passed += 1;
            match aggregate {
                Some(ref mut inner) => inner.merge(&share.into()).unwrap(),
                None => aggregate = Some(share.into()),
            }
        }
    }

    Ok((
        aggregate.unwrap(),
        peer.num_bytes_sent() - start_comm,
        clients_passed,
        verif_time,
    ))
}


#[tokio::main]
pub async fn main() {
    let options = Options::load_from_json("SV2 Server");

    tracing_subscriber::fmt()
        .pretty()
        .with_max_level(options.log_level)
        .init();

    let prio3_len: usize = options.vec_size as usize;
    let prio3_chunk_len: usize = options.chunk_size as usize;

    // if true, we collect all of the clients tags together and check them together.
    // if false, we run group testing on n/NUM_CORES size groups in parallel
    let single_tag = options.single_tag;

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

    let mut peer_idgen = IdGen::new();

    info!("Peer connection set up!");

    let verify_key = if options.is_alice() {
        // Broadcast a random verifier key to the other server
        let mut rng = thread_rng();
        let key = rng.gen::<[u8; SEED_SIZE]>();
        peer.send_message(peer_idgen.next_send_id(), key.to_vec());
        key
    } else {
        let key = peer
            .subscribe_and_get::<UseSerde<Vec<u8>>>(peer_idgen.next_recv_id())
            .await;
        key.unwrap().to_vec().try_into().unwrap()
    };

    // establish connections with the meta client
    info!("My client port: {}", options.client_port);
    let listener = TcpListener::bind(("0.0.0.0", options.client_port))
        .await
        .unwrap();

    let mut prio3: Prio3Gadgets = Prio3Gadgets {
        prio3sv: None,
        prio3hist: None,
        prio3avg: None,
    };
    match options.agg_fn {
        AggFunc::SumVec => {
            prio3.prio3sv = Some(Prio3::new_sum_vec_256(2, 16, prio3_len, prio3_chunk_len).unwrap())
        }
        AggFunc::Histogram => {
            prio3.prio3hist = Some(Prio3::new_histogram_256(2, prio3_len, prio3_chunk_len).unwrap())
        }
        AggFunc::Average => prio3.prio3avg = Some(Prio3::new_average_256(2, AVG_BITS).unwrap()),
    };

    let num_inputs = options.num_clients;
    let mut now;

    let mut global_aggregate: Option<AggregateShare<Field128>> = None;

    let total_v_comm;
    let mut clients_passed = 0;
    let total_verif_time;
    now = Instant::now();
    let clients = ClientsPool::new(NUM_CORES, &listener).await;
    let mut client_idgen = IdGen::new();

    let client_keys = clients
        .subscribe_and_get::<UseSerde<Vec<Vec<u8>>>>(client_idgen.next_recv_id())
        .await
        .unwrap();

    info!("Key collection: {:?}", now.elapsed());
    info!("Starting aggregation");
    now = Instant::now();
    
    // Each process gets its own set of exchange ids for communicating with the other server.
    // For whisper, we only need to use this to verify tags, so 1million rounds per process should be more than enough.
    let exchange_idgens = (0..NUM_CORES)
        .map(|_| peer_idgen.reserve_rounds(1000000))
        .collect::<Vec<_>>();

    if single_tag {
        let client_keys = client_keys.into_par_iter().flatten().collect::<Vec<_>>();
        let (agg_share, v_comm, passed_count, verif_time)  = run_vdaf_prepare_rayon(
            prio3.clone(),
            verify_key.clone(),
            client_keys,
            peer.clone(),
            options.num_bad_clients,
            exchange_idgens.into_iter().next().unwrap(),
            options.agg_fn.clone(),
        )
        .await
        .unwrap();
        total_v_comm = v_comm;
        global_aggregate = Some(agg_share);
        clients_passed = passed_count;
        total_verif_time = verif_time;
    } else {
        let old_v_comm = peer.num_bytes_sent();
        let handles = client_keys
            .into_iter()
            .zip(exchange_idgens.into_iter())
            .map(|(client_keys_batch, exchange_idgen)| {
                let tmp_prio3 = prio3.clone();
                let tmp_peer = peer.clone();
                let tmp_agg_func = options.agg_fn.clone();
                tokio::spawn(async move {
                    run_vdaf_prepare(
                        tmp_prio3,
                        verify_key.clone(),
                        client_keys_batch,
                        tmp_peer,
                        options.num_bad_clients / NUM_CORES,
                        exchange_idgen,
                        tmp_agg_func,
                    )
                    .await
                })
            })
            .collect::<FuturesUnordered<_>>();

        for handle in handles {
            let (agg_share, passed_count, _verif_time) = handle.await.unwrap().unwrap();
            clients_passed += passed_count;
            if let Some(ref mut inner) = global_aggregate {
                inner.merge(&agg_share).unwrap();
            } else {
                global_aggregate = Some(agg_share);
            }
        }
        total_verif_time = now.elapsed();
        total_v_comm = peer.num_bytes_sent() - old_v_comm;
    }

    info!("Verification comm: {:?}", total_v_comm);
    info!("Verif time: {:?}", total_verif_time);
    match options.agg_fn {
        AggFunc::SumVec => {
            let other_aggregate = AggregateShare::<Field128>::get_decoded_with_param(
                &(prio3.prio3sv.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        global_aggregate.clone().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let _result = prio3
                .prio3sv
                .unwrap()
                .unshard(
                    &(),
                    vec![global_aggregate.clone().unwrap(), other_aggregate],
                    num_inputs,
                )
                .unwrap();
        }
        AggFunc::Histogram => {
            let other_aggregate = AggregateShare::<Field128>::get_decoded_with_param(
                &(prio3.prio3hist.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        global_aggregate.clone().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let _result = prio3
                .prio3hist
                .unwrap()
                .unshard(
                    &(),
                    vec![global_aggregate.clone().unwrap(), other_aggregate],
                    num_inputs,
                )
                .unwrap();
        }
        AggFunc::Average => {
            let other_aggregate = AggregateShare::<Field128>::get_decoded_with_param(
                &(prio3.prio3avg.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        global_aggregate.clone().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let _result = prio3
                .prio3avg
                .unwrap()
                .unshard(
                    &(),
                    vec![global_aggregate.clone().unwrap(), other_aggregate],
                    num_inputs,
                )
                .unwrap();
        }
    }

    let aggregation_time = now.elapsed();
    info!("Finished aggregation, {:?}", aggregation_time);
    info!(
        "Aggregation comm: {:?}",
        global_aggregate.unwrap().get_encoded().len()
    );
    info!("Aggregation function used {:?}", options.agg_fn);
    info!("Clients passed: {:?}", clients_passed);
    //info!("aggregate: {:?}", _result);
}
