use common::{group::BatchedSampling, prg::FromRng, Group, Ordered, Share, SizedGroup};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bucket<G> {
    pub(crate) data: Vec<G>,
    pub(crate) sign_data: G,
    pub(crate) ctr: G,
}

pub const STRING_SIZE: usize = 256;
impl<G: Group + Clone> SizedGroup for Bucket<G> {
    #[inline]
    fn size() -> usize {
        STRING_SIZE + 2
    }
    #[inline]
    fn core_size() -> usize {
        STRING_SIZE
    }
    #[inline]
    fn padded_size() -> usize {
        Self::size() + 8 - (Self::size() % 8)
    }
}

impl<G: Group + Clone> Group for Bucket<G> {
    #[inline]
    fn zero() -> Self {
        Self {
            data: vec![G::zero(); STRING_SIZE],
            sign_data: G::zero(),
            ctr: G::zero(),
        }
    }

    #[inline]
    fn one() -> Self {
        Self {
            data: vec![G::one(); STRING_SIZE],
            sign_data: G::one(),
            ctr: G::one(),
        }
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        for i in 0..self.data.len() {
            self.data[i].add(&other.data[i]);
        }
        self.sign_data.add(&other.sign_data);
        self.ctr.add(&other.ctr);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        // This is hadamard product
        assert_eq!(self.data.len(), other.data.len());
        for i in 0..self.data.len() {
            self.data[i].mul(&other.data[i]);
        }
        self.sign_data.mul(&other.sign_data);
        self.ctr.mul(&other.ctr);
    }

    #[inline]
    fn negate(&mut self) {
        for i in 0..self.data.len() {
            self.data[i].negate();
        }
        self.sign_data.negate();
        self.ctr.negate();
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        for i in 0..self.data.len() {
            self.data[i].sub(&other.data[i]);
        }
        self.sign_data.sub(&other.sign_data);
        self.ctr.sub(&other.ctr);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl<G: Group + Clone + FromRng + BatchedSampling> FromRng for Bucket<G> {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        // This can be inefficient because it calls one prg for each element
        // for i in 0..self.size {
        //     self.data[i].from_rng(rng);
        // }

        // Optimized version
        let rnd_data = G::get_random_batch(self.data.len() + 2, rng);
        for i in 0..self.data.len() {
            self.data[i] = rnd_data[i].clone();
        }
        self.sign_data = rnd_data[self.data.len()].clone();
        self.ctr = rnd_data[self.data.len() + 1].clone();
    }
}

impl<G: Group + Ordered> Bucket<G> {
    pub fn recover(&self) -> Vec<bool> {
        if self.sign_data.positive() {
            self.data.iter().map(|a| a.positive()).collect()
        } else {
            self.data.iter().map(|a| !a.positive()).collect()
        }
    }
}

impl<G: Group + Clone + FromRng + BatchedSampling> Share for Bucket<G> {}

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

mod tests {
    #[cfg(test)]
    use crate::bucket::Bucket;

    #[cfg(test)]
    use common::{Group, Share};

    #[test]
    fn share_bucket() {
        let val = Bucket::<u64>::random();
        println!("val: {:?}", val);
        let (s0, s1) = val.share();
        let mut out = Bucket::<u64>::zero();
        out.add(&s0);
        out.add(&s1);
        assert_eq!(out, val);
    }

    #[test]
    fn test_bucket_recover() {
        use crate::utils::bytes_to_bucket;
        use rand::Rng;

        let mut b = Bucket::<u64>::zero();
        let s1 = "hello my name is curious george!"; // sign: 1, 0
        let s2 = "asdfiej;lj2938psh,mcs932shdfpqw8"; // sign: 0, 1
        let b1 = bytes_to_bucket::<u64>(s1.as_bytes(), false);
        let b2 = bytes_to_bucket::<u64>(s2.as_bytes(), true);

        println!("{:?} {:?}", b1.sign_data, b2.sign_data);
        for _ in 0..100 {
            // 100 b1's
            b.add(&b1);
        }
        for _ in 0..60 {
            // 60 b2's
            b.add(&b2);
        }
        for _ in 0..30 {
            // 30 random's
            let rand_str = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(32)
                .map(char::from)
                .collect::<String>();
            b.add(&bytes_to_bucket::<u64>(
                rand_str.as_bytes(),
                rand::thread_rng().gen(),
            ));
        }

        assert_eq!(common::bits_to_string(&b.recover()).unwrap(), s1);

        for _ in 0..160 {
            // 220 b2's
            b.add(&b2);
        }

        for _ in 0..80 {
            // 80 random's
            let rand_str: String = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(32)
                .map(char::from)
                .collect();
            b.add(&bytes_to_bucket::<u64>(
                rand_str.as_bytes(),
                rand::thread_rng().gen(),
            ));
        }

        assert_eq!(common::bits_to_string(&b.recover()).unwrap(), s2);
    }
}
