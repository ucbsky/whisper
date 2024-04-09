use std::collections::HashSet;

use crate::bucket::Bucket;
use common::{bits_to_bytes, Group};
use serde::{Deserialize, Serialize};

/// Defines functionality for CountSketch, the data structure used for recovering heavy hitters.
///
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CountSketch<G: Group> {
    pub buckets: Vec<Bucket<G>>,
    pub ctr: usize,
}

impl<G: Group + Clone + std::fmt::Debug + common::Ordered + std::cmp::PartialOrd> CountSketch<G> {
    pub fn new(num_buckets: usize) -> Self {
        Self {
            buckets: vec![Bucket::<G>::zero(); num_buckets],
            ctr: 0,
        }
    }
    pub fn insert(&mut self, input: &CountSketch<G>) {
        self.buckets
            .iter_mut()
            .zip(input.buckets.iter())
            .for_each(|(cur_bkt, new_bkt)| cur_bkt.add(new_bkt));
        self.ctr += input.ctr;
    }

    pub fn recover(&self, threshold: G, rep_count: usize) -> HashSet<Vec<u8>> {
        let mut result = HashSet::new();
        for (i, b) in self.buckets.as_slice().iter().enumerate() {
            if b.ctr > threshold {
                let potential_hh_bytes = bits_to_bytes(&b.recover());
                let (_, recovered_bkt_idx) = crate::get_sign_and_bkt(
                    &potential_hh_bytes,
                    self.buckets.len(),
                    rep_count as u16,
                );
                if recovered_bkt_idx == i as u16 {
                    result.insert(potential_hh_bytes);
                }
            }
        }
        result
    }

    pub fn from_bucket_vec(buckets: Vec<Bucket<G>>) -> Self {
        CountSketch { buckets, ctr: 1 }
    }
}

#[cfg(test)]
mod tests {
    use sha2::Digest;
    use sha2::Sha256;

    use crate::bucket::Bucket;
    use crate::countsketch::CountSketch;
    use std::collections::HashSet;

    fn true_heavy_hitters(recovery_threshold: f32, salt: &[u8]) -> HashSet<Vec<u8>> {
        let mut result = HashSet::new();
        let n = match recovery_threshold {
            x if (0.009..0.011).contains(&x) => 10,
            x if (0.0009..0.0011).contains(&x) => 100,
            _ => panic!("Unexpected recovery threshold"),
        };
        let mut hasher = Sha256::new();
        hasher.update(salt);
        for i in 1usize..n + 1 {
            let mut cur_hasher = hasher.clone();
            cur_hasher.update(i.to_string().as_bytes());
            let hash = cur_hasher.finalize().to_vec();
            result.insert(hash);
        }
        result
    }

    #[test]
    fn test_countsketch() {
        use crate::get_sign_and_bkt;
        use crate::utils::bytes_to_bucket;
        use common::Group;
        use rand::distributions::Distribution;
        use rand::thread_rng;
        use zipf::ZipfDistribution;

        const NUM_REPS: usize = 14;
        const NUM_SAMPLES: usize = 10000;
        type AggRing = u64;
        let zipf = ZipfDistribution::new(10000, 1.03).unwrap();
        let num_buckets = 256usize;
        let mut accumulators = vec![CountSketch::<AggRing>::new(num_buckets); NUM_REPS];

        let mut rng = thread_rng();
        let salt = [47u8; 32];

        let mut hasher = Sha256::new();
        hasher.update(salt);

        (0..NUM_SAMPLES).into_iter().for_each(|_| {
            let mut new_hasher = hasher.clone();
            let sample = zipf.sample(&mut rng);
            new_hasher.update(sample.to_string().as_bytes());
            let sample_str = new_hasher.finalize().to_vec();
            for (i, accumulator) in accumulators.iter_mut().enumerate() {
                let (sign, idx) = get_sign_and_bkt(&sample_str, num_buckets, i as u16);
                let bkt = bytes_to_bucket(&sample_str, sign);
                let mut bkt_vec = vec![Bucket::<AggRing>::zero(); num_buckets];
                bkt_vec[idx as usize] = bkt;
                let sketch = CountSketch::from_bucket_vec(bkt_vec);
                accumulator.insert(&sketch);
            }
        });
        let thresholds = [0.01, 0.001];
        for threshold in thresholds {
            let mut all_hhs = HashSet::new();
            for (i, accumulator) in accumulators.iter().enumerate() {
                let recovered = accumulator.recover((threshold * NUM_SAMPLES as f32) as AggRing, i);
                all_hhs.extend(recovered);
            }
            let true_hhs = true_heavy_hitters(threshold, &salt);
            let mut true_recovered = 0;
            for s in &all_hhs {
                if true_hhs.contains(s) {
                    true_recovered += 1;
                }
            }
            println!("Threshold: {}", threshold);
            println!("total recovered, correct recovered, false positives, num missed");
            println!(
                "{}, {}, {}, {}",
                all_hhs.len(),
                true_recovered,
                all_hhs.len() - true_recovered,
                true_hhs.len() - true_recovered
            );
        }
    }
}
