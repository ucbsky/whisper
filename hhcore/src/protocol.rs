use common::{group::BatchedSampling, prg, BitDecomposable, Group};
use serde::{Deserialize, Serialize};
use sha256::digest;

use crate::{
    bucket::{Bucket, STRING_SIZE},
    vdpf::VDPFKey,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct KeyChain<G> {
    pub id: u128,
    pub main_key: VDPFKey<Bucket<G>>,
    pub(crate) support_keys: Vec<VDPFKey<G>>,
}

pub trait BCSketch<G>: Group {
    fn gen_key_chain<GIn>(self, id: u128, alpha: GIn) -> (KeyChain<G>, KeyChain<G>)
    where
        G: prg::FromRng + Clone + Group + std::fmt::Debug + BatchedSampling,
        GIn: prg::FromRng + Clone + std::fmt::Debug + BitDecomposable;

    fn eval_and_check<GIn>(key: KeyChain<G>) -> (Vec<Self>, String)
    where
        G: prg::FromRng + Clone + Group + std::fmt::Debug + BatchedSampling + BitDecomposable,
        GIn: prg::FromRng + Clone + std::fmt::Debug + BitDecomposable,
        Self: Sized;
}

impl<G: Group + Clone + std::cmp::PartialEq + std::fmt::Debug> BCSketch<G> for Bucket<G> {
    fn gen_key_chain<GIn>(self, id: u128, alpha: GIn) -> (KeyChain<G>, KeyChain<G>)
    where
        G: prg::FromRng + Clone + Group + std::fmt::Debug + BatchedSampling,
        GIn: prg::FromRng + Clone + std::fmt::Debug + BitDecomposable,
    {
        let (main_key0, main_key1) = VDPFKey::gen::<GIn>(alpha, self.clone());
        let mut support_keys0 = Vec::new();
        let mut support_keys1 = Vec::new();

        for i in 0..self.data.len() {
            let (sk0, sk1);
            if self.data[i] == G::one() {
                (sk0, sk1) = VDPFKey::gen::<bool>(true, G::one());
            } else if self.data[i] == G::minusone() {
                (sk0, sk1) = VDPFKey::gen::<bool>(false, G::one());
            } else {
                panic!("Data is not -1 or 1");
            }
            support_keys0.push(sk0);
            support_keys1.push(sk1);
        }

        let (sk0, sk1);
        if self.sign_data == G::one() {
            (sk0, sk1) = VDPFKey::gen::<bool>(true, G::one());
        } else if self.sign_data == G::minusone() {
            (sk0, sk1) = VDPFKey::gen::<bool>(false, G::one());
        } else {
            panic!("Data is not -1 or 1");
        }

        support_keys0.push(sk0);
        support_keys1.push(sk1);
        (
            KeyChain {
                id,
                main_key: main_key0,
                support_keys: support_keys0,
            },
            KeyChain {
                id,
                main_key: main_key1,
                support_keys: support_keys1,
            },
        )
    }

    fn eval_and_check<GIn>(key: KeyChain<G>) -> (Vec<Self>, String)
    where
        G: prg::FromRng + Clone + Group + std::fmt::Debug + BatchedSampling + BitDecomposable,
        GIn: prg::FromRng + Clone + std::fmt::Debug + BitDecomposable,
        Self: Sized,
    {
        let mut buck = Self::zero();
        let mut proof_string = String::new();
        let mut support_vec = Vec::new();
        let (dat, proof) = key.main_key.eval_all::<GIn>();

        for dat_bkt in &dat {
            buck.add(dat_bkt);
        }

        proof_string.push_str(&proof.proof);

        assert!(key.support_keys.len() == STRING_SIZE + 1);
        for i in 0..key.support_keys.len() {
            let (dat_small, proof_small) = key.support_keys[i].eval_all::<bool>();

            let dat_left = dat_small[0].clone();
            let mut dat_right = dat_small[1].clone();

            dat_right.sub(&dat_left);

            // To later check that left and right sum to 1 (ensures that the vote was correct & indeed 1)
            let mut dat_tmp = G::zero();
            dat_tmp.add(&dat_small[0]);
            dat_tmp.add(&dat_small[1]);
            support_vec.push(dat_tmp);

            if i == key.support_keys.len() - 1 {
                // sign data
                buck.sign_data.sub(&dat_right);
            } else {
                // data
                buck.data[i].sub(&dat_right);
            }

            proof_string.push_str(&proof_small.proof.to_string());
        }

        // At this point, buck as a vector should represent shares of 0 except ctr field
        if key.main_key.key_id {
            // remove 1 from buck's ctr
            buck.ctr.sub(&G::one());
            buck.negate();
        }
        // Shares of 0 have now become equal shares

        // Convert buck to bytes and sha_256 it
        const BUCK_SIZE: usize = STRING_SIZE + 2;
        const BUCK_ONES_SIZE: usize = STRING_SIZE + 1; // ctr being 1 has already been ensured above
        let mut buck_bytes = Vec::<u8>::new();

        for i in 0..buck.data.len() {
            buck_bytes.append(&mut buck.data[i].as_bytes());
        }

        buck_bytes.append(&mut buck.sign_data.as_bytes());

        buck_bytes.append(&mut buck.ctr.as_bytes());

        // Now checking that all support keys were of the form: [0, 1] or [1, 0]
        if key.main_key.key_id {
            for i in 0..support_vec.len() {
                support_vec[i].sub(&G::one());
                support_vec[i].negate();
            }
        }

        // Now support_vec should be exactly the same for both servers
        let mut buck_ones_bytes = Vec::<u8>::new();

        for i in 0..support_vec.len() {
            buck_ones_bytes.append(&mut support_vec[i].as_bytes());
        }

        if (G::bitsize() == 64) || (G::bitsize() == 32) || (G::bitsize() == 16) {
            let blow_factor = G::bitsize() / 8;
            // First digest the proof that support keys match the main VDPF
            //let mut bytes_buf = [0u8; BUCK_SIZE * blow_factor];
            let mut bytes_buf = vec![0u8; BUCK_SIZE * blow_factor];
            bytes_buf.copy_from_slice(&buck_bytes);
            let mut hash = digest(&bytes_buf[..]);
            proof_string.push_str(&hash);
            // Second digest the proof that support keys are of the form [0, 1] or [1, 0]
            //bytes_buf = [0u8; BUCK_ONES_SIZE * blow_factor];
            bytes_buf = vec![0u8; BUCK_ONES_SIZE * blow_factor];
            bytes_buf.copy_from_slice(&buck_ones_bytes);
            hash = digest(&bytes_buf[..]);
            proof_string.push_str(&hash);
        } else {
            unimplemented!("Only implemented for 64/32/16 bit groups");
        }

        (dat, digest(proof_string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bucket::Bucket;
    use colored::Colorize;
    use common::group::IntModN;
    use common::prg::FromRng;
    use rayon::prelude::*;
    use std::time::Instant;
    use typenum::Unsigned;

    #[test]
    fn test_optimized_core_single() {
        type N = typenum::U1024;
        type AggRing = u32;
        //println!("Testing optimized core for M = {}", M::to_u64());

        let mut rng = rand::thread_rng();
        let mut alpha: IntModN<N> = IntModN::zero();
        alpha.from_rng(&mut rng);
        println!("alpha: {}", alpha.val);
        let alpha_clone = alpha.clone();
        println!("Number of buckets: {}", (1 << IntModN::<N>::bitsize()));

        println!("Randomly sampling a valid beta/bucket...");
        let beta = Bucket::<AggRing>::rand();
        let beta_clone = beta.clone();

        println!("Generating key chain [@clients]...");
        let (key0, key1) = beta.gen_key_chain(0u128, alpha);

        println!("Evaluating and checking [@servers]...");
        println!("Eval [@server0]...");
        let (mut dat0, proof0) =
            <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(key0);
        println!("Eval [@server1]...");
        let (dat1, proof1) =
            <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(key1);

        println!("Checking if proofs are equal [@servers with comm.]...");
        assert_eq!(proof0, proof1);

        println!("Reconstructing the sketch [@servers with comm.]...");
        dat0.iter_mut().zip(dat1.iter()).for_each(|(x, y)| {
            x.add(y);
        });

        println!(
            "Checking if reconstructed sketch is equal to original sketch [for testing only]..."
        );
        assert_eq!(dat0[alpha_clone.val as usize].data, beta_clone.data);
    }

    #[ignore]
    #[test]
    fn bench_optimized_core_batch() {
        type N = typenum::U1024;
        type AggRing = u16;
        let clients = 1e4 as usize;
        let mode = 2; // -1 for multithreaded with collect (no fold or reduce), 0 for single-threaded (fold), 1 for multi-threaded with rayon (fold), 2 for multi-threaded with rayon (reduce)
                      // Fold vs reduce (https://docs.rs/rayon/1.5.0/rayon/iter/trait.ParallelIterator.html#combining-fold-with-other-operations): reduce() requires that the identity function has the same type as the things you are iterating over, and it fully reduces the list of items into a single item, while with fold(), the identity function does not have to have the same type as the things you are iterating over, and you potentially get back many results.
        println!("Number of buckets: {}", (1 << IntModN::<N>::bitsize()));
        println!("Number of clients: {}", clients);

        println!("Generating random sketches for each client [@clients]...");
        let mut now = Instant::now();

        let raw_data = (0..clients)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut alpha: IntModN<N> = IntModN::zero();
                alpha.from_rng(&mut rng);
                let beta = Bucket::<AggRing>::rand();
                (alpha, beta)
            })
            .collect::<Vec<(IntModN<N>, Bucket<AggRing>)>>();

        let mut elapsed = now.elapsed();
        println!(
            "Time elapsed in random sketch generation (all clients combined): {:.2?}",
            elapsed
        );

        println!("Generating key chains [@clients]...");
        now = Instant::now();

        let key_pack = raw_data
            .into_par_iter()
            .map(|(alpha, beta)| {
                let (key0, key1) = beta.gen_key_chain(0u128, alpha);
                (key0, key1)
            })
            .collect::<Vec<(KeyChain<AggRing>, KeyChain<AggRing>)>>();

        elapsed = now.elapsed();
        println!(
            "Time elapsed in key generation (all clients combined): {:.2?}",
            elapsed
        );

        println!("Evaluating and checking [@servers]...");
        now = Instant::now();

        if mode == -1 {
            let _server0_work = key_pack
                .into_par_iter()
                .map(|(k0, _)| {
                    let (dat0, proof0) =
                        <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(k0);
                    (dat0, proof0)
                })
                .collect::<Vec<(Vec<Bucket<AggRing>>, String)>>();
        } else if mode == 0 {
            let (_server0_acc_dat, server0_acc_proof) = key_pack
                .into_iter()
                .map(|(k0, _)| {
                    let (dat0, proof0) =
                        <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(k0);
                    (dat0, proof0)
                })
                .fold(
                    (
                        vec![Bucket::<AggRing>::zero(); N::to_usize()],
                        String::new(),
                    ),
                    |(mut acc_dat, mut acc_proof), (dat, proof)| {
                        for i in 0..dat.len() {
                            acc_dat[i].add(&dat[i]);
                        }
                        acc_proof.push_str(&proof);
                        (acc_dat, acc_proof)
                    },
                );
            let _acc_proof_digest = digest(&server0_acc_proof[..]);
        } else if mode == 1 {
            let (server0_acc_dat, _server0_acc_proof) = key_pack
                .into_par_iter()
                .map(|(k0, _)| {
                    let (dat0, proof0) =
                        <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(k0);
                    (dat0, proof0)
                })
                .fold(
                    || {
                        (
                            vec![Bucket::<AggRing>::zero(); N::to_usize()],
                            String::new(),
                        )
                    },
                    |(mut acc_dat, mut acc_proof), (dat, proof)| {
                        for i in 0..dat.len() {
                            acc_dat[i].add(&dat[i]);
                        }
                        acc_proof.push_str(&proof);
                        (acc_dat, acc_proof)
                    },
                )
                .collect::<(Vec<Vec<Bucket<AggRing>>>, Vec<String>)>();
            println!(
                "Rayon fold returned {} partial folds (typically roughly matches #cores)",
                server0_acc_dat.len().to_string().yellow()
            );
        } else {
            // mode = 2
            let (_server0_acc_dat, _server0_acc_proof) = key_pack
                .into_par_iter()
                .map(|(k0, _)| {
                    let (dat0, proof0) =
                        <Bucket<AggRing> as BCSketch<AggRing>>::eval_and_check::<IntModN<N>>(k0);
                    (dat0, proof0)
                })
                .reduce(
                    || {
                        (
                            vec![Bucket::<AggRing>::zero(); N::to_usize()],
                            String::new(),
                        )
                    },
                    |(mut acc_dat, mut acc_proof), (dat, proof)| {
                        for i in 0..dat.len() {
                            acc_dat[i].add(&dat[i]);
                        }
                        acc_proof.push_str(&proof);
                        (acc_dat, acc_proof)
                    },
                );
        }

        elapsed = now.elapsed();
        println!(
            "Time elapsed in VDPF EvalAll (all clients combined): {:.2?}",
            elapsed
        );
        println!(
            "Time elapsed in 17 runs is estimated to be: {:.2?}",
            elapsed * 17
        );
    }
}
