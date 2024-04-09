use sha2::{Digest, Sha256};

pub mod bucket;
pub mod countsketch;
pub mod dpf;
pub mod protocol;
pub mod utils;
pub mod vdpf;

// The type of each individual bit counter in the sketch.
// If the sketch needs to hold fewer than 2^16 strings, use u16 for increased performance and
// smaller keys. 
pub type AggRing = u32;

#[inline]
pub fn get_sign_and_bkt(v: &[u8], num_buckets: usize, rep_count: u16) -> (bool, u16) {
    let mut hasher = Sha256::new();
    hasher.update(v);
    hasher.update(rep_count.to_le_bytes());
    let hash = hasher.finalize();
    let hash_u64 = u64::from_le_bytes(hash[3..11].try_into().unwrap());
    let sign = hash[0] & 1 == 1;
    (sign, (hash_u64 % num_buckets as u64) as u16)
}
