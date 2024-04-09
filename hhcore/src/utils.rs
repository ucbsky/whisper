use common::Group;
use rand::Rng;

use crate::bucket::{Bucket, STRING_SIZE};

impl<G: Group + Clone + std::fmt::Debug> Bucket<G> {
    pub fn sub_one(&mut self) {
        for i in 0..self.data.len() {
            self.data[i].sub(&G::one());
        }
        self.sign_data.sub(&G::one());
        self.ctr.sub(&G::one());
    }

    /// Generate a random bucket of the BC sketch with the right distribution of each counter
    pub fn rand() -> Self {
        let mut rng = rand::thread_rng();
        let rand_bits: Vec<bool> = (0..STRING_SIZE).map(|_| rng.gen()).collect();

        let data = (0..STRING_SIZE)
            .map(|i| {
                if rand_bits[i] {
                    G::one()
                } else {
                    G::minusone()
                }
            })
            .collect();
        let sign_data = if rng.gen::<bool>() {
            G::one()
        } else {
            G::minusone()
        };

        Self {
            data,
            sign_data,
            ctr: G::one(),
        }
    }
}

pub fn bytes_to_bucket<G: Group + Clone + std::cmp::PartialEq + std::fmt::Debug>(
    s: &[u8],
    sign: bool,
) -> Bucket<G> {
    let bits = common::bytes_to_bits(s);
    let signed_one = if sign { G::one() } else { G::minusone() };
    let signed_minusone = if sign { G::minusone() } else { G::one() };
    let bc_sketch_bkt = Bucket::<G> {
        data: bits
            .iter()
            .map(|bit| {
                if *bit {
                    signed_one.clone()
                } else {
                    signed_minusone.clone()
                }
            })
            .collect(),
        sign_data: signed_one,
        ctr: G::one(),
    };
    bc_sketch_bkt
}
