use bridge::{id_tracker::IdGen, mpc_conn::MpcConnection};
use prio::{codec::Encode, field::Field128, vdaf::prio3::Prio3BatchedOutputShare};
use serde::{Deserialize, Serialize};
use serialize::UseSerde;
use sha2::{Digest, Sha256};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};
use std::collections::HashSet;

use crate::VERIFY_KEY_SIZE;

pub trait ToBytes: Sized {
    fn to_bytes(&self) -> Vec<u8>;
}

impl ToBytes for String {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClientProofTag<P> {
    pub testing_id: u128,
    pub tag: P,
}

impl<P> ClientProofTag<P> {
    pub fn new(testing_id: u128, tag: P) -> Self {
        Self { testing_id, tag }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
enum BatchProofTag {
    Tag16([u8; 16]),
    Tag32([u8; 32]),
}

impl<P: ToBytes + PartialEq + std::fmt::Debug + Send> ClientProofTag<P> {
    fn batch_proof(
        proofs: &[Self],
        verify_key: &[u8; VERIFY_KEY_SIZE],
        upper: u128,
        tag_size: usize,
    ) -> (BatchProofTag, usize) {
        match tag_size {
            16 => {
                let result = Self::batch_proof_shake128(proofs, verify_key, upper);
                (BatchProofTag::Tag16(result.0), result.1)
            }
            32 => {
                let result = Self::batch_proof_sha256(proofs, verify_key, upper);
                (BatchProofTag::Tag32(result.0), result.1)
            }
            _ => panic!("Unexpected output size"),
        }
    }

    fn batch_proof_sha256(
        proofs: &[Self],
        verify_key: &[u8; VERIFY_KEY_SIZE],
        upper: u128,
    ) -> ([u8; 32], usize) {
        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, verify_key);
        let mut split_idx = proofs.len();
        for (i, proof) in proofs.iter().enumerate() {
            if proof.testing_id > upper {
                split_idx = i;
                break;
            }
            Digest::update(&mut hasher, &proof.tag.to_bytes());
        }
        (hasher.finalize().into(), split_idx)
    }

    fn batch_proof_shake128(
        proofs: &[Self],
        verify_key: &[u8; VERIFY_KEY_SIZE],
        upper: u128,
    ) -> ([u8; 16], usize) {
        let mut hasher = Shake128::default();
        hasher.update(verify_key);
        let mut split_idx = proofs.len();
        for (i, proof) in proofs.iter().enumerate() {
            if proof.testing_id > upper {
                split_idx = i;
                break;
            }
            hasher.update(&proof.tag.to_bytes());
        }
        let mut reader = hasher.finalize_xof();
        let mut res1 = [0u8; 16];
        reader.read(&mut res1);
        (res1, split_idx)
    }
}

impl ToBytes for Field128 {
    fn to_bytes(&self) -> Vec<u8> {
        self.get_encoded()
    }
}

impl ToBytes for Prio3BatchedOutputShare<Field128> {
    fn to_bytes(&self) -> Vec<u8> {
        self.get_encoded()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GroupTestMessage<P: ToBytes + PartialEq + std::fmt::Debug + Clone + Send> {
    range: (u128, u128),
    singleton: Option<ClientProofTag<P>>,
    batch_proof: Option<BatchProofTag>,
}

/// Performs a general binary split test
///
pub async fn general_binary_split_test<
    P: ToBytes
        + PartialEq
        + std::fmt::Debug
        + Clone
        + Serialize
        + for<'a> Deserialize<'a>
        + Send
        + Sync
        + 'static,
>(
    proofs: &[ClientProofTag<P>],
    verify_key: &[u8; VERIFY_KEY_SIZE],
    idgen: &mut IdGen,
    peer: &MpcConnection,
    d: usize,
    message_size: usize,
) -> (HashSet<u128>, usize) {
    // First, do a complete batch proof, to see if we need to to group testing at all.
    let mut comm: usize = 0;
    let (batch_proof, _) = ClientProofTag::batch_proof(proofs, verify_key, u128::MAX, message_size);
    let received_batch_proof = peer
        .exchange_message(idgen.next_exchange_id(), UseSerde(batch_proof))
        .await
        .unwrap();
    comm += 1;
    if batch_proof == received_batch_proof {
        return (HashSet::new(), comm);
    }
    // Uses a similar idea to Hwang's group testing algorithm, but with lower round complexity.
    // Split up the proofs into n/d batches, expecting each batch to have about one defect.
    let d = std::cmp::max(d, 2);
    let quotient = u128::MAX / d as u128;
    let remainder = u128::MAX % d as u128;

    let ranges_to_check = (0..d)
        .map(|i| {
            if (i as u128) < remainder {
                (quotient + 1) * (i as u128 + 1)
            } else {
                remainder * (quotient + 1) + (i as u128 + 1 - remainder) * (quotient)
            }
        })
        .collect::<Vec<_>>();

    // What I need to send for the next round
    let mut my_msgs = Vec::with_capacity(d);

    // In the case where we send a singleton and the other party has a large batch, we might not know if
    // our singleton was correct until after the other party checks. In this case, if the other party
    // finds an error, it's sent over as a special correction.
    let mut my_corrections = Vec::new();

    // Keeps track of the proofs in each range.
    let mut cur_slices = Vec::with_capacity(d);
    let mut slice_so_far = proofs;
    let mut lower = 0;

    // Get the initial batch proofs, for d different sections.
    for i in 0..d as usize {
        let upper = ranges_to_check[i];
        let (pf, slice_idx) =
            ClientProofTag::batch_proof(slice_so_far, verify_key, upper, message_size);
        if slice_idx == 0 {
            my_msgs.push(GroupTestMessage::<P> {
                range: (lower, upper),
                singleton: None,
                batch_proof: None,
            });
        } else if slice_idx == 1 {
            my_msgs.push(GroupTestMessage::<P> {
                range: (lower, upper),
                singleton: Some(slice_so_far[0].clone()),
                batch_proof: None,
            });
        } else {
            my_msgs.push(GroupTestMessage::<P> {
                range: (lower, upper),
                singleton: None,
                batch_proof: Some(pf),
            });
        }
        cur_slices.push(slice_so_far);
        slice_so_far = &slice_so_far[slice_idx..];
        lower = upper;
    }
    let mut result = HashSet::new();

    while !(my_msgs.is_empty() && my_corrections.is_empty()) {
        comm += 1;
        let (received_msgs, received_corrections) = peer
            .exchange_message(
                idgen.next_exchange_id(),
                UseSerde((my_msgs.clone(), my_corrections.clone())),
            )
            .await
            .unwrap();

        // First, see if there's anything we missed in the last round
        for correction in received_corrections {
            result.insert(correction);
        }

        let mut new_msgs = Vec::new();
        let mut new_slices = Vec::new();
        let mut new_corrections = Vec::new();
        my_msgs
            .into_iter()
            .zip(received_msgs.into_iter())
            .zip(cur_slices.into_iter())
            .for_each(|((my_msg, received_msg), slice)| {
                assert!(received_msg.range == my_msg.range);
                match my_msg {
                    GroupTestMessage {
                        range: _,
                        singleton: Some(singleton),
                        batch_proof: None,
                    } => {
                        match received_msg {
                            GroupTestMessage {
                                range: _,
                                singleton: None,
                                batch_proof: None,
                            } => {
                                // My singleton is invalid
                                result.insert(singleton.testing_id);
                            }
                            GroupTestMessage {
                                range: _,
                                singleton: Some(other_singleton),
                                batch_proof: None,
                            } => {
                                // If our singletons match, then they're both good.
                                // Otherwise, they're both bad, so I'll mark mine.
                                if other_singleton != singleton {
                                    result.insert(singleton.testing_id);
                                }
                            }
                            _ => {}
                        }
                    }

                    // We only need to do something if we had multiple proofs in this range, and
                    GroupTestMessage {
                        range,
                        singleton: None,
                        batch_proof: Some(my_batch_pf),
                    } => {
                        match received_msg {
                            GroupTestMessage {
                                range: _,
                                singleton: Some(singleton),
                                batch_proof: None,
                            } => {
                                for proof in slice {
                                    // need both the testing_id and the batchableproof to match
                                    if singleton.testing_id == proof.testing_id {
                                        if singleton.tag == proof.tag {
                                            continue;
                                        } else {
                                            new_corrections.push(proof.testing_id);
                                        }
                                    }
                                    result.insert(proof.testing_id);
                                }
                            }
                            GroupTestMessage {
                                range: _,
                                singleton: None,
                                batch_proof: None,
                            } => {
                                // The other party didn't receive anything in this range.
                                // Everything in this slice was asymmetrically delivered.
                                for proof in slice {
                                    result.insert(proof.testing_id);
                                }
                            }
                            GroupTestMessage {
                                range: _,
                                singleton: None,
                                batch_proof: Some(other_batch_pf),
                            } => {
                                if my_batch_pf != other_batch_pf {
                                    let midpt = range.0 + ((range.1 - range.0) >> 1);
                                    let (left_pf, mid_idx) = ClientProofTag::batch_proof(
                                        slice,
                                        verify_key,
                                        midpt,
                                        message_size,
                                    );
                                    let (right_pf, end_idx) = ClientProofTag::batch_proof(
                                        &slice[mid_idx..],
                                        verify_key,
                                        range.1,
                                        message_size,
                                    );
                                    let left_msg = match mid_idx {
                                        0 => GroupTestMessage {
                                            range: (range.0, midpt),
                                            singleton: None,
                                            batch_proof: None,
                                        },
                                        1 => GroupTestMessage {
                                            range: (range.0, midpt),
                                            singleton: Some(slice[0].clone()),
                                            batch_proof: None,
                                        },
                                        _ => GroupTestMessage {
                                            range: (range.0, midpt),
                                            singleton: None,
                                            batch_proof: Some(left_pf),
                                        },
                                    };

                                    let right_msg = match end_idx {
                                        0 => GroupTestMessage {
                                            range: (midpt, range.1),
                                            singleton: None,
                                            batch_proof: None,
                                        },
                                        1 => GroupTestMessage {
                                            range: (midpt, range.1),
                                            singleton: Some(slice[mid_idx].clone()),
                                            batch_proof: None,
                                        },
                                        _ => GroupTestMessage {
                                            range: (midpt, range.1),
                                            singleton: None,
                                            batch_proof: Some(right_pf),
                                        },
                                    };
                                    new_msgs.push(left_msg);
                                    new_msgs.push(right_msg);
                                    new_slices.push(&slice[..mid_idx]);
                                    new_slices.push(&slice[mid_idx..]);
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            });
        my_msgs = new_msgs;
        my_corrections = new_corrections;
        cur_slices = new_slices;
    }
    (result, comm)
}

#[tokio::test]
async fn test_general_binary_split() {
    use bridge::mpc_conn::mpc_localhost_pair;
    use rand::thread_rng;
    use rand::RngCore;

    let (peer0, peer1) = mpc_localhost_pair(6665, 16).await;

    const NUM_STRS: usize = 300;
    const MSG_SIZE: usize = 32;
    let expected = 5;

    let tags0 = (0..NUM_STRS).map(|i| i.to_string()).collect::<Vec<_>>();
    let mut tags1 = tags0.clone();
    let mut idgen0 = IdGen::new();
    let mut idgen1 = IdGen::new();

    let error_locs = [20, 68, 100];
    for i in error_locs {
        tags1[i] = "INCONSISTENCY".to_owned();
    }

    let mut testing_ids = (0..NUM_STRS)
        .map(|_| {
            let mut buf = [0u8; 16];
            thread_rng().fill_bytes(&mut buf);
            u128::from_le_bytes(buf)
        })
        .collect::<Vec<_>>();
    testing_ids.sort_by(|a, b| a.cmp(b));

    let (proofs0, mut proofs1): (Vec<_>, Vec<_>) = testing_ids
        .into_iter()
        .zip(tags0.into_iter())
        .zip(tags1.into_iter())
        .map(|((id, tag0), tag1)| (ClientProofTag::new(id, tag0), ClientProofTag::new(id, tag1)))
        .unzip();

    let deletion_locs = [200, 4];
    for i in deletion_locs {
        proofs1.remove(i);
    }

    let verify_key = [1u8; 16];

    let handle0 = general_binary_split_test(
        &proofs0,
        &verify_key,
        &mut idgen0,
        &peer0,
        expected,
        MSG_SIZE,
    );
    let handle1 = general_binary_split_test(
        &proofs1,
        &verify_key,
        &mut idgen1,
        &peer1,
        expected,
        MSG_SIZE,
    );
    let (result0, result1) = futures::join!(handle0, handle1);
    println!("{:?}, {:?}", result0, result1);
    assert!(result0.0.len() == error_locs.len() + deletion_locs.len());
    assert!(result1.0.len() == error_locs.len());

    for error_id in result1.0 {
        for proof in proofs1.iter() {
            if proof.testing_id == error_id {
                assert!(proof.tag == "INCONSISTENCY".to_owned());
            }
        }
    }
    println!("{:?} ", peer0.num_bytes_sent());
}
