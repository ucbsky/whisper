use std::{io::Cursor, time::Instant};

use bin_utils::prioserver::Options;
use bin_utils::{AggFunc, Prio3Gadgets, F};
use bridge::id_tracker::ExchangeId;
use bridge::{client_server::ClientsPool, id_tracker::IdGen, mpc_conn::MpcConnection};
use prio::codec::{Encode, ParameterizedDecode};

use futures::stream::FuturesUnordered;
use itertools::Itertools;
use prio::vdaf::Aggregator;
use prio::vdaf::{
    Aggregatable, AggregateShare, Client, Collector, PrepareTransition, VdafError, VdafKey,
};
use rand::{thread_rng, Rng};
use serialize::{Communicate, UseSerde};
use tokio::net::TcpListener;
use tracing::info;

const NONCE_SIZE: usize = 16;
const VERIFY_KEY_SIZE: usize = 16;
const NUM_CORES: usize = 32;

// Prepares all the [`vdaf_keys`], turning them into output shares.
// Uses the provided [`ExchangeId`] to exchange one message with the other server.
async fn run_vdaf_prepare<V, const VERIFY_KEY_SIZE: usize>(
    vdaf: V,
    verify_key: [u8; VERIFY_KEY_SIZE],
    agg_param: V::AggregationParam,
    vdaf_keys: Vec<Vec<u8>>,
    peer: MpcConnection,
    exchange_id: ExchangeId,
) -> (Result<Vec<V::OutputShare>, VdafError>, usize)
where
    V: Client<NONCE_SIZE> + Aggregator<VERIFY_KEY_SIZE, NONCE_SIZE> + Collector + Sync,
    V::AggregationParam: Sync,
    V::PrepareMessage: Send,
    V::OutputShare: Send + Sync,
{
    let mut verif_comm = 0;
    let (states, prepare_shares): (Vec<_>, Vec<_>) = vdaf_keys
        .iter()
        .enumerate()
        .map(|(_id, encoded_vdaf_key)| {
            let vdaf_key =
                VdafKey::decode_with_param(&vdaf, &mut Cursor::new(encoded_vdaf_key)).unwrap();

            let (state, prepare_share) = vdaf
                .prepare_init(
                    &verify_key,
                    vdaf_key.agg_id,
                    &agg_param,
                    &vdaf_key.nonce,
                    &vdaf_key.public_share,
                    &vdaf_key.input_share,
                )
                .unwrap();
            let prep_sh_enc = prepare_share.get_encoded();
            ((state, vdaf_key.agg_id), prep_sh_enc)
        })
        .unzip();

    let my_msg = UseSerde(prepare_shares.clone());
    verif_comm += my_msg.size_in_bytes();

    let other_prepare_shares_encoded = &peer.exchange_message(exchange_id, my_msg).await.unwrap();

    let out_shares = prepare_shares
        .iter()
        .zip(other_prepare_shares_encoded.iter())
        .zip(states.iter())
        .map(
            |((prepare_share_encoded, other_prepare_share_encoded), (state, agg_id))| {
                let other_prepare_share =
                    V::PrepareShare::get_decoded_with_param(state, other_prepare_share_encoded)
                        .unwrap();
                let prepare_share =
                    V::PrepareShare::get_decoded_with_param(state, &prepare_share_encoded).unwrap();
                let inputs = if *agg_id == 1 {
                    vec![other_prepare_share, prepare_share]
                } else {
                    vec![prepare_share, other_prepare_share]
                };
                let prepare_message = vdaf
                    .prepare_shares_to_prepare_message(&agg_param, inputs)
                    .unwrap();
                let transition = vdaf
                    .prepare_next(state.clone(), prepare_message.clone())
                    .unwrap();

                let outshare = match transition {
                    PrepareTransition::Continue(_, _) => {
                        panic!("Unexpected PrepareTransition::Continue in prio3")
                    }
                    PrepareTransition::Finish(transition_out_share) => transition_out_share,
                };
                outshare
            },
        )
        .collect_vec();

    (Ok(out_shares), verif_comm)
}

async fn main_with_options(options: Options) {
    tracing_subscriber::fmt()
        .pretty()
        .with_max_level(options.log_level)
        .init();

    let prio3_len = options.vec_size as usize;
    let prio3_chunk_len = options.chunk_size as usize;

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
        let key = rng.gen::<[u8; VERIFY_KEY_SIZE]>();
        peer.send_message(peer_idgen.next_send_id(), key.to_vec());
        key
    } else {
        let key = peer
            .subscribe_and_get::<UseSerde<Vec<u8>>>(peer_idgen.next_recv_id())
            .await;
        key.unwrap().to_vec().try_into().unwrap()
    };

    // Establish connections with the meta client
    info!("My client port: {}", options.client_port);
    let listener = TcpListener::bind(("0.0.0.0", options.client_port))
        .await
        .unwrap();

    // Total number of client messages
    let num_inputs = options.num_clients;
    let mut now;
    let prio3: Prio3Gadgets = Prio3Gadgets::new(&options.agg_fn, prio3_len, prio3_chunk_len);

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

    let exchange_ids = (0..NUM_CORES)
        .map(|_| peer_idgen.next_exchange_id())
        .collect::<Vec<_>>();
    let client_keys = client_keys;

    let mut output_shares = Vec::with_capacity(options.num_clients);

    // At this point, `client_keys` is chunked up into `NUM_CORES` batches.
    // Spawn a process to prepare each batch.
    let handles = client_keys
        .into_iter()
        .zip(exchange_ids.into_iter())
        .map(|(client_keys_batch, exchange_id)| {
            let tmp_prio3 = prio3.clone();
            let tmp_peer = peer.clone();
            match options.agg_fn {
                AggFunc::SumVec => tokio::spawn(async move {
                    let vdaf_tmp = tmp_prio3.prio3sv.unwrap();
                    run_vdaf_prepare(
                        vdaf_tmp,
                        verify_key.clone(),
                        (),
                        client_keys_batch,
                        tmp_peer,
                        exchange_id,
                    )
                    .await
                }),
                AggFunc::Histogram => tokio::spawn(async move {
                    let vdaf_tmp = tmp_prio3.prio3hist.unwrap();
                    run_vdaf_prepare(
                        vdaf_tmp,
                        verify_key.clone(),
                        (),
                        client_keys_batch,
                        tmp_peer,
                        exchange_id,
                    )
                    .await
                }),
                AggFunc::Average => tokio::spawn(async move {
                    let vdaf_tmp = tmp_prio3.prio3avg.unwrap();
                    run_vdaf_prepare(
                        vdaf_tmp,
                        verify_key.clone(),
                        (),
                        client_keys_batch,
                        tmp_peer,
                        exchange_id,
                    )
                    .await
                }),
            }
        })
        .collect::<FuturesUnordered<_>>();

    let mut verif_comm = 0;
    for handle in handles {
        let res = handle.await.unwrap();
        verif_comm += res.1;
        let mut batch_out_shares = res.0.unwrap();
        output_shares.append(&mut batch_out_shares);
    }

    info!("Verification communication: {} bytes", verif_comm);

    let prepare_time = now.elapsed();
    info!("Finished preparation, {:?}", prepare_time);

    // Aggregate all output shares.
    let mut aggregate: Option<AggregateShare<F>> = None;
    for out_share in output_shares {
        let this_agg_share = out_share.into();
        if let Some(ref mut inner) = aggregate {
            inner.merge(&this_agg_share).unwrap();
        } else {
            aggregate = Some(this_agg_share);
        }
    }

    match options.agg_fn {
        AggFunc::SumVec => {
            let other_aggregate = AggregateShare::<F>::get_decoded_with_param(
                &(prio3.prio3sv.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        aggregate.as_ref().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let result = prio3
                .prio3sv
                .unwrap()
                .unshard(&(), vec![aggregate.unwrap(), other_aggregate], num_inputs)
                .unwrap();
            info!("Aggregation comm: {:?}", 16 * result.len());
        }
        AggFunc::Histogram => {
            let other_aggregate = AggregateShare::<F>::get_decoded_with_param(
                &(prio3.prio3hist.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        aggregate.as_ref().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let result = prio3
                .prio3hist
                .unwrap()
                .unshard(&(), vec![aggregate.unwrap(), other_aggregate], num_inputs)
                .unwrap();
            info!("Aggregation comm: {:?}", 16 * result.len());
        }
        AggFunc::Average => {
            let other_aggregate = AggregateShare::<F>::get_decoded_with_param(
                &(prio3.prio3avg.as_ref().unwrap(), &()),
                &peer
                    .exchange_message(
                        peer_idgen.next_exchange_id(),
                        aggregate.as_ref().unwrap().get_encoded(),
                    )
                    .await
                    .unwrap(),
            )
            .unwrap();

            let _result = prio3
                .prio3avg
                .unwrap()
                .unshard(&(), vec![aggregate.unwrap(), other_aggregate], num_inputs)
                .unwrap();
            info!("Aggregation comm: {:?}", 8);
        }
    }

    let aggregation_time = now.elapsed();
    info!(
        "Finished aggregation, {:?}",
        aggregation_time - prepare_time
    );
    info!("Total time: {:?}", aggregation_time);
}

#[tokio::main]
pub async fn main() {
    let options = Options::load_from_json("SV2 Server");
    main_with_options(options).await;
}
