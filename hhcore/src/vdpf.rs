use common::{
    prg::{self, HashOutput},
    BitDecomposable, Group,
};
use serde::{Deserialize, Serialize};
use sha256::digest;

use crate::dpf::{
    gen_correction, gen_output_correction, CorrectionWord, EvalState, OutputCorrection, TupleUtil,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionProof {
    correction: HashOutput,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProofString {
    pub proof: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VDPFKey<GOut> {
    pub(crate) key_id: bool,
    root_seed: prg::PrgSeed,
    correction_words: Vec<CorrectionWord>,
    correction_proof: CorrectionProof,
    output_correction: OutputCorrection<GOut>,
}

impl<GOut> VDPFKey<GOut>
where
    GOut: prg::FromRng + Clone + Group + std::fmt::Debug,
{
    pub fn zero() -> Self {
        Self {
            key_id: false,
            root_seed: prg::PrgSeed::zero(),
            correction_words: Vec::new(),
            correction_proof: CorrectionProof {
                correction: HashOutput {
                    seeds: (
                        prg::PrgSeed::zero(),
                        prg::PrgSeed::zero(),
                        prg::PrgSeed::zero(),
                        prg::PrgSeed::zero(),
                    ),
                },
            },
            output_correction: OutputCorrection::<GOut> { word: GOut::zero() },
        }
    }

    pub fn gen<GIn>(alpha: GIn, beta: GOut) -> (VDPFKey<GOut>, VDPFKey<GOut>)
    where
        GIn: prg::FromRng + Clone + BitDecomposable + std::fmt::Debug,
    {
        let mut sample_success = false;
        let mut keys = (VDPFKey::<GOut>::zero(), VDPFKey::<GOut>::zero());

        while !sample_success {
            let alpha_bits = alpha.decompose();

            let root_seeds = (prg::PrgSeed::random(), prg::PrgSeed::random());
            let root_bits = (false, true);

            let mut seeds = root_seeds.clone();
            let mut bits = root_bits;

            let mut correction_words: Vec<CorrectionWord> = Vec::new();

            for (_i, &cur_bit) in alpha_bits.iter().enumerate() {
                correction_words.push(gen_correction(cur_bit, &mut bits, &mut seeds));
            }

            let proof_0 = seeds.0.mmo_hash2to4(&alpha);
            let proof_1 = seeds.1.mmo_hash2to4(&alpha);

            // 'cs' in the paper
            let correction_proof = CorrectionProof {
                correction: &proof_0 ^ &proof_1,
            };

            bits = (seeds.0.get_lsb(), seeds.1.get_lsb());

            if bits.0 == bits.1 {
                //println!("LSB of seed 0 and seed 1 matched. Running key generation again...");
                continue;
            } else {
                sample_success = true;
            }

            let output_correction = gen_output_correction(&beta, &bits, &seeds);

            keys = (
                VDPFKey::<GOut> {
                    key_id: false,
                    root_seed: root_seeds.0,
                    correction_words: correction_words.clone(),
                    correction_proof: correction_proof.clone(),
                    output_correction: output_correction.clone(),
                },
                VDPFKey::<GOut> {
                    key_id: true,
                    root_seed: root_seeds.1,
                    correction_words,
                    correction_proof,
                    output_correction,
                },
            )
        }
        keys
    }

    pub fn eval_init(&self) -> EvalState {
        EvalState {
            level: 0,
            seed: self.root_seed.clone(),
            bit: self.key_id,
        }
    }

    pub fn eval_all<GIn>(&self) -> (Vec<GOut>, ProofString)
    where
        GIn: prg::FromRng + Clone + BitDecomposable + std::fmt::Debug,
    {
        let depth = GIn::bitsize();
        let mut res = Vec::new();
        // Storing all inner nodes minimizes the PRG calls compared to if they do in-order traversal of the tree using recursion or stack.
        let node_count = 2 * (1 << depth) - 1;
        let mut nodes: Vec<EvalState> = Vec::new();
        // pi in the paper; this step is pi <- cs
        let mut pi = self.correction_proof.correction.clone();
        let mut proof_string = ProofString {
            proof: String::new(),
        };

        nodes.push(self.eval_init());

        for i in 0..((1 << depth) - 1) {
            let state = nodes[i].clone();

            let expanded = state.seed.expand_direction(true, true);

            let mut left_seed = expanded.seeds.get(false).clone();
            let mut left_bit = *expanded.bits.get(false);
            let mut right_seed = expanded.seeds.get(true).clone();
            let mut right_bit = *expanded.bits.get(true);

            // Tau computation
            if state.bit {
                left_seed = &left_seed ^ &self.correction_words[state.level].seed;
                left_bit ^= self.correction_words[state.level].bits.get(false);
                right_seed = &right_seed ^ &self.correction_words[state.level].seed;
                right_bit ^= self.correction_words[state.level].bits.get(true);
            }

            nodes.push(EvalState {
                level: state.level + 1,
                seed: left_seed,
                bit: left_bit,
            });

            nodes.push(EvalState {
                level: state.level + 1,
                seed: right_seed,
                bit: right_bit,
            });
        }

        let mut x = GIn::zero();
        // Handle proof and output level
        for i in ((1 << depth) - 1)..node_count {
            let state = nodes[i].clone();

            // tilde{pi} in the paper
            let tilde_pi = state.seed.mmo_hash2to4(&x);
            let bit = state.seed.get_lsb();

            let converted = state.seed.convert::<GOut>();

            let mut word = converted.word;
            // correct(tilde{pi}, cs, t) in the paper
            let tilde_pi_corrected = if bit {
                word.add(&self.output_correction.word);
                &tilde_pi ^ &self.correction_proof.correction
            } else {
                tilde_pi
            };

            if self.key_id {
                word.negate();
            }

            res.push(word);

            let h_prime_in = &pi ^ &tilde_pi_corrected;
            let h_prime_out = h_prime_in.mmo_hash4to4();

            pi = &pi ^ &h_prime_out;

            x.add(&GIn::one());
        }

        let mut pi_bytes = [0u8; 64];
        pi_bytes[..16].copy_from_slice(&pi.seeds.0.key);
        pi_bytes[16..32].copy_from_slice(&pi.seeds.1.key);
        pi_bytes[32..48].copy_from_slice(&pi.seeds.2.key);
        pi_bytes[48..64].copy_from_slice(&pi.seeds.3.key);
        // Compute SHA 256 of pi
        proof_string.proof = digest(&pi_bytes);

        (res, proof_string)
    }
}

#[cfg(test)]
mod tests {
    use common::group::IntModN;

    use super::*;
    use crate::bucket::Bucket;
    use std::time::Instant;

    #[test]
    fn test_vdpf() {
        let alpha = 190u16;
        let beta = 99910u64;
        let (key0, key1) = VDPFKey::gen(alpha, beta);

        let now = Instant::now();
        let (res0, resproof0) = key0.eval_all::<u16>();
        let (res1, resproof1) = key1.eval_all::<u16>();
        let elapsed = now.elapsed();
        println!("Time elapsed in EvalAll: {:.2?}", elapsed);

        println!("Checking if proof is valid...");
        assert_eq!(resproof0, resproof1);
        println!("Proofs are: {}, {}", resproof0.proof, resproof1.proof);
        println!("Passed!");

        (0..u16::MAX).for_each(|idx| {
            if idx == alpha {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == beta);
            } else {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == 0);
            }
        });
    }

    #[test]
    fn test_vdpf_small_gin() {
        use common::group::IntModN;

        let alpha = IntModN::<typenum::U1024>::from_u16(19u16);
        let beta = 99910u64;
        let (key0, key1) = VDPFKey::gen(alpha, beta);

        let now = Instant::now();
        let (res0, resproof0) = key0.eval_all::<IntModN<typenum::U1024>>();
        let (res1, resproof1) = key1.eval_all::<IntModN<typenum::U1024>>();
        let elapsed = now.elapsed();
        println!("Time elapsed in EvalAll: {:.2?}", elapsed);

        println!("Checking if proof is valid...");
        assert_eq!(resproof0, resproof1);
        println!("Proofs are: {}, {}", resproof0.proof, resproof1.proof);
        println!("Passed!");

        (0..1024).for_each(|idx| {
            if idx == alpha.val {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == beta);
            } else {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == 0);
            }
        });
    }

    #[test]
    fn test_vdpf_large_gout() {
        use common::Share;
        let alpha = IntModN::<typenum::U64>::from_u16(19u16);
        let beta = Bucket::<u64>::random();
        let zero = Bucket::<u64>::zero();
        let (key0, key1) = VDPFKey::gen(alpha, beta.clone());

        let now = Instant::now();
        let (mut res0, resproof0) = key0.eval_all::<IntModN<typenum::U64>>();
        let (res1, resproof1) = key1.eval_all::<IntModN<typenum::U64>>();
        let elapsed = now.elapsed();
        println!("Time elapsed in EvalAll: {:.2?}", elapsed);

        println!("Checking if proof is valid...");
        assert_eq!(resproof0, resproof1);
        println!("Passed!");

        (0..64).for_each(|idx| {
            res0[idx as usize].add(&res1[idx as usize]);
            if IntModN::<typenum::U64>::from_u16(idx) == alpha {
                assert_eq!(res0[idx as usize], beta);
            } else {
                assert_eq!(res0[idx as usize], zero);
            }
        });
    }
}
