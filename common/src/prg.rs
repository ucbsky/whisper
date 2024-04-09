use core::arch::x86_64::{
    __m128i, _mm_add_epi64, _mm_loadu_si128, _mm_set_epi64x, _mm_storeu_si128,
};

use aes::cipher::{generic_array::GenericArray, Block, BlockEncrypt, KeyInit};
use aes::cipher::{BlockEncryptMut, KeyIvInit};
use aes::Aes128;
use ctr::cipher::StreamCipher;
use ecb::Encryptor;

use rand::Rng;

use serde::Deserialize;
use serde::Serialize;
use std::ops;

// AES key size in bytes. We always use AES-128,
// which has 16-byte keys.
const AES_KEY_SIZE: usize = 16;

// AES block size in bytes. Always 16 bytes.
pub const AES_BLOCK_SIZE: usize = 16;

// XXX Todo try using 8-way parallelism
// This stream uses block cipher and is used for DPF
pub struct FixedKeyPrgStream {
    aes: Aes128,
    ctr: __m128i,
    buf: [u8; AES_BLOCK_SIZE * 8],
    have: usize,
    buf_ptr: usize,
    count: usize,
}

/// Implements H' hash function from the paper but the output is 4\lambda bits (not 2\lambda); therefore, in the protocol, we need to output a SHA hash over this to make output 2\lambda bits. Uses MMO one-way compression and fixed-key AES (block cipher mode).
pub struct HasherStream {
    aes: Aes128,
    in_blocks: [__m128i; 4],
    buf: [u8; AES_BLOCK_SIZE * 4],
}

use std::cell::RefCell;

use crate::BitDecomposable;

// This global stream keeps getting updated with a new key whenever we call set_key. A handle to a pointer/reference of this key is obtained using "with" function.
thread_local!(static FIXED_KEY_STREAM: RefCell<FixedKeyPrgStream> = RefCell::new(FixedKeyPrgStream::new()));
thread_local!(static HASHER_STREAM: RefCell<HasherStream> = RefCell::new(HasherStream::new()));

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PrgSeed {
    pub key: [u8; AES_KEY_SIZE],
}

pub trait FromRng {
    fn from_rng(&mut self, stream: &mut (impl rand::Rng + rand_core::RngCore));

    /// Uses thread_rng as the randomness stream. Change to another stream if needed.
    fn randomize(&mut self) {
        self.from_rng(&mut rand::thread_rng());
    }
}

// This stream uses AES in ctr mode and is not used in DPF. It is rather used when we want to return a RNG from a PRG seed. That is the only place it is used.
pub struct PrgStream {
    stream: ctr::Ctr128LE<Aes128>,
}

pub struct PrgOutput {
    pub bits: (bool, bool),
    pub seeds: (PrgSeed, PrgSeed),
}

pub struct PrgOutputBits {
    pub bits: Vec<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HashOutput {
    pub seeds: (PrgSeed, PrgSeed, PrgSeed, PrgSeed),
}

pub struct ConvertOutput<T: FromRng> {
    pub seed: PrgSeed,
    pub word: T,
}

impl ops::BitXor for &PrgSeed {
    type Output = PrgSeed;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut out = PrgSeed::zero();

        for ((out, left), right) in out.key.iter_mut().zip(&self.key).zip(&rhs.key) {
            *out = left ^ right;
        }

        out
    }
}

impl ops::BitXor for &HashOutput {
    type Output = HashOutput;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::Output {
            seeds: (
                &self.seeds.0 ^ &rhs.seeds.0,
                &self.seeds.1 ^ &rhs.seeds.1,
                &self.seeds.2 ^ &rhs.seeds.2,
                &self.seeds.3 ^ &rhs.seeds.3,
            ),
        }
    }
}

impl PrgSeed {
    pub fn to_rng(&self) -> PrgStream {
        let iv: [u8; AES_BLOCK_SIZE] = [0; AES_BLOCK_SIZE];

        let key = GenericArray::from_slice(&self.key);
        let nonce = GenericArray::from_slice(&iv);
        PrgStream {
            stream: ctr::Ctr128LE::new(key, nonce),
        }
    }

    pub fn get_lsb(&self) -> bool {
        self.key[0] & 0x1 == 0x1
    }

    /// Expand PRG seed into two new seeds if the corresponding bit is set. If left is set, left side is expanded, if right is set, right side is expanded, and if both, then both expanded.
    pub fn expand_direction(self: &PrgSeed, left: bool, right: bool) -> PrgOutput {
        FIXED_KEY_STREAM.with(|s_in| {
            let mut key_short = self.key;

            // Zero out first (from LSB) two bits and use for output
            let bits_from_seed = ((key_short[0] & 0x1) == 0, (key_short[0] & 0x2) == 0);
            key_short[0] &= 0xFC;

            let mut s = s_in.borrow_mut();
            s.set_key(&key_short);

            let mut out = PrgOutput {
                bits: bits_from_seed,
                seeds: (PrgSeed::zero(), PrgSeed::zero()),
            };

            if left {
                if left != right {
                    // Because here fill bytes will be called only once
                    s.fill_bytes_fast(&mut out.seeds.0.key);
                } else {
                    // Because here fill bytes will be called twice; happens for some parts of keygen only
                    //s.fill_bytes(&mut out.seeds.0.key);
                    s.fill_bytes_fast2(&mut out.seeds.0.key, &mut out.seeds.1.key);
                }
            } else {
                s.skip_block();
            }

            if right {
                if left != right {
                    s.fill_bytes_fast(&mut out.seeds.1.key);
                } else {
                    //s.fill_bytes(&mut out.seeds.1.key); Already taken care of in the left case
                }
            } else {
                s.skip_block();
            }

            out
        })
    }

    pub fn expand(self: &PrgSeed) -> PrgOutput {
        self.expand_direction(true, true)
    }

    pub fn convert<T: FromRng + crate::Group>(self: &PrgSeed) -> ConvertOutput<T> {
        let mut out = ConvertOutput {
            seed: PrgSeed::zero(),
            word: T::zero(),
        };

        FIXED_KEY_STREAM.with(|s_in| {
            let mut s = s_in.borrow_mut();
            s.set_key(&self.key);
            // We don't need this extra call because we only use the "word" from convert output unlike IDPF from Poplar
            // s.fill_bytes(&mut out.seed.key);
            unsafe {
                let sp = s_in.as_ptr();
                out.word.from_rng(&mut *sp);
            }
        });

        out
    }

    // Ref https://github.com/sachaservan/vdpf/blob/main/src/mmo.c
    /// This emulates the hash function H as defined in the paper https://eprint.iacr.org/2021/580.pdf; this is used to generate the proof of correctness of the DPF.
    pub fn mmo_hash2to4<GIn: FromRng + crate::Group + BitDecomposable>(
        self: &PrgSeed,
        x: &GIn,
    ) -> HashOutput {
        FIXED_KEY_STREAM.with(|s_in| {
            let mut out = HashOutput {
                seeds: (
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                ),
            };

            // First block is generated from MMO of the input index x
            let mut x_bytes: [u8; AES_BLOCK_SIZE] = [0; AES_BLOCK_SIZE];
            let tmp = x.as_bytes();
            assert!(tmp.len() <= x_bytes.len());
            x_bytes[..tmp.len()].copy_from_slice(&tmp);

            let mut s = s_in.borrow_mut();

            s.set_key(&x_bytes);
            s.fill_bytes_fast(&mut out.seeds.0.key);

            // Second block is generated from MMO of the self PRG seed
            s.set_key(&self.key);
            s.fill_bytes_fast(&mut out.seeds.1.key);

            // Third block is generated from MMO of the first block
            s.set_key(&out.seeds.0.key);
            s.fill_bytes_fast(&mut out.seeds.2.key);

            // Fourth block is generated from MMO of the second block
            s.set_key(&out.seeds.1.key);
            s.fill_bytes_fast(&mut out.seeds.3.key);

            out
        })
    }

    pub fn zero() -> PrgSeed {
        PrgSeed {
            key: [0; AES_KEY_SIZE],
        }
    }

    pub fn random() -> PrgSeed {
        let mut key: [u8; AES_KEY_SIZE] = [0; AES_KEY_SIZE];
        rand::thread_rng().fill(&mut key);

        PrgSeed { key }
    }
}

impl rand::RngCore for PrgStream {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for v in dest.iter() {
            debug_assert_eq!(*v, 0u8);
        }

        self.stream.apply_keystream(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl FixedKeyPrgStream {
    fn new() -> Self {
        // Fixed key for AES
        let key = GenericArray::from_slice(&[0; AES_KEY_SIZE]);

        let ctr_init = FixedKeyPrgStream::load(&[0; AES_BLOCK_SIZE]);
        FixedKeyPrgStream {
            aes: Aes128::new(key),
            ctr: ctr_init,
            buf: [0; AES_BLOCK_SIZE * 8],
            buf_ptr: AES_BLOCK_SIZE,
            have: AES_BLOCK_SIZE,
            count: 0,
        }
    }

    fn set_key(&mut self, key: &[u8; 16]) {
        // This is fixed key because when we call set_key, it doesn't change the key but rather changes the counter to be the key we wanted it to be set
        self.ctr = FixedKeyPrgStream::load(key);
        self.buf_ptr = AES_BLOCK_SIZE;
        self.have = AES_BLOCK_SIZE;
    }

    fn skip_block(&mut self) {
        // Only allow skipping a block on a block boundary.
        debug_assert_eq!(self.have % AES_BLOCK_SIZE, 0);
        debug_assert_eq!(self.buf_ptr, AES_BLOCK_SIZE);
        self.ctr = FixedKeyPrgStream::inc_be(self.ctr);
    }

    // Only use when you are sure that refill_fast will be called only once for each key
    fn refill_fast(&mut self) {
        //println!("Refill");
        debug_assert_eq!(self.buf_ptr, AES_BLOCK_SIZE);

        // Write counter into buffer.
        FixedKeyPrgStream::store(self.ctr, &mut self.buf[0..AES_BLOCK_SIZE]);

        let count_bytes = self.buf;
        //println!("count_bytes when set: {:?}", count_bytes);
        let gen = GenericArray::from_mut_slice(&mut self.buf[0..AES_BLOCK_SIZE]);
        //println!("Gen before enc: {:?}", gen);
        self.aes.encrypt_block(gen);
        //println!("Gen after enc: {:?}", gen);
        //println!("count_bytes after enc: {:?}", count_bytes);

        // Compute:   AES_0000(ctr) XOR ctr
        self.buf
            .iter_mut()
            .zip(count_bytes.iter())
            .for_each(|(x1, x2)| *x1 ^= *x2);
    }

    // Only use when you are sure that fill_bytes_fast will be called only once for each key
    fn fill_bytes_fast(&mut self, dest: &mut [u8]) {
        self.refill_fast();
        dest.copy_from_slice(&self.buf[..dest.len()]);
    }

    fn refill(&mut self) {
        //println!("Refill");
        debug_assert_eq!(self.buf_ptr, AES_BLOCK_SIZE);

        self.have = AES_BLOCK_SIZE;
        self.buf_ptr = 0;

        // Write counter into buffer.
        FixedKeyPrgStream::store(self.ctr, &mut self.buf[0..AES_BLOCK_SIZE]);

        let count_bytes = self.buf;
        //println!("count_bytes when set: {:?}", count_bytes);
        let gen = GenericArray::from_mut_slice(&mut self.buf[0..AES_BLOCK_SIZE]);
        //println!("Gen before enc: {:?}", gen);
        self.aes.encrypt_block(gen);
        //println!("Gen after enc: {:?}", gen);
        //println!("count_bytes after enc: {:?}", count_bytes);

        // Compute:   AES_0000(ctr) XOR ctr
        self.buf
            .iter_mut()
            .zip(count_bytes.iter())
            .for_each(|(x1, x2)| *x1 ^= *x2);

        self.ctr = FixedKeyPrgStream::inc_be(self.ctr);
        self.count += AES_BLOCK_SIZE;
    }

    // This is only called if Gout size is large. It is never called for the PRG expansion because our PRG outputs are just size 2\lambda.
    fn refill8(&mut self) {
        self.have = 8 * AES_BLOCK_SIZE;
        self.buf_ptr = 0;

        //let block = GenericArray::clone_from_slice(&[0u8; 16]);
        //let mut block8 = GenericArray::clone_from_slice(&[block; 8]);
        let mut block8 = [GenericArray::clone_from_slice(&[0u8; 16]); 8];

        let mut cnts = [[0u8; AES_BLOCK_SIZE]; 8];
        for i in 0..8 {
            // Write counter into buffer
            FixedKeyPrgStream::store(self.ctr, &mut block8[i]);
            FixedKeyPrgStream::store(self.ctr, &mut cnts[i]);
            self.ctr = FixedKeyPrgStream::inc_be(self.ctr);
        }

        self.aes.encrypt_blocks(&mut block8);

        for i in 0..8 {
            // Compute:   AES_0000(ctr) XOR ctr
            block8[i]
                .iter_mut()
                .zip(cnts[i].iter())
                .for_each(|(x1, x2)| *x1 ^= *x2);
        }

        for i in 0..8 {
            self.buf[i * AES_BLOCK_SIZE..(i + 1) * AES_BLOCK_SIZE].copy_from_slice(&block8[i]);
        }

        self.count += 8 * AES_BLOCK_SIZE;

        //println!("Blocks: {:?}", self.buf[0]);
        //println!("Blocks: {:?}", self.buf[1]);
        //println!("Blocks: {:?}", self.buf[2]);
    }

    fn refill2(&mut self) {
        self.have = 2 * AES_BLOCK_SIZE;
        self.buf_ptr = 0;

        //let block = GenericArray::clone_from_slice(&[0u8; 16]);
        //let mut block8 = GenericArray::clone_from_slice(&[block; 8]);
        let mut block2 = [GenericArray::clone_from_slice(&[0u8; 16]); 2];

        let mut cnts = [[0u8; AES_BLOCK_SIZE]; 2];
        for i in 0..2 {
            // Write counter into buffer
            FixedKeyPrgStream::store(self.ctr, &mut block2[i]);
            FixedKeyPrgStream::store(self.ctr, &mut cnts[i]);
            self.ctr = FixedKeyPrgStream::inc_be(self.ctr);
        }

        self.aes.encrypt_blocks(&mut block2);

        for i in 0..2 {
            // Compute:   AES_0000(ctr) XOR ctr
            block2[i]
                .iter_mut()
                .zip(cnts[i].iter())
                .for_each(|(x1, x2)| *x1 ^= *x2);
        }

        for i in 0..2 {
            self.buf[i * AES_BLOCK_SIZE..(i + 1) * AES_BLOCK_SIZE].copy_from_slice(&block2[i]);
        }

        self.count += 2 * AES_BLOCK_SIZE;

        //println!("Blocks: {:?}", self.buf[0]);
        //println!("Blocks: {:?}", self.buf[1]);
        //println!("Blocks: {:?}", self.buf[2]);
    }

    fn fill_bytes_fast2(&mut self, dest1: &mut [u8], dest2: &mut [u8]) {
        self.refill2();
        dest1.copy_from_slice(&self.buf[..dest1.len()]);
        dest2.copy_from_slice(&self.buf[dest1.len()..dest1.len() + dest2.len()]);
    }

    // From RustCrypto aesni crate
    #[inline(always)]
    fn inc_be(v: __m128i) -> __m128i {
        unsafe { _mm_add_epi64(v, _mm_set_epi64x(1, 0)) }
    }

    #[inline(always)]
    fn store(val: __m128i, at: &mut [u8]) {
        debug_assert_eq!(at.len(), AES_BLOCK_SIZE);

        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            _mm_storeu_si128(at.as_mut_ptr() as *mut __m128i, val)
        }
    }

    // Modified from RustCrypto aesni crate
    #[inline(always)]
    fn load(key: &[u8; 16]) -> __m128i {
        let val = Block::<Aes128>::from_slice(key);

        // Safety: `loadu` supports unaligned loads
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            _mm_loadu_si128(val.as_ptr() as *const __m128i)
        }
    }
}

impl rand::RngCore for FixedKeyPrgStream {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    // Can be slow because of all the unncessary arithmetic it does when we already know the key will be refreshed after one call to fill_bytes because we use a key to expand at most 2 blocks (gout is taken care by a separate function, so ignore that for this argument).
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        //println!("Fill bytes: {}", dest.len());
        let mut dest_ptr = 0;
        while dest_ptr < dest.len() {
            if self.buf_ptr == self.have {
                if dest.len() > 4 * AES_BLOCK_SIZE {
                    self.refill8();
                    //println!("Refill8");
                    //self.refill();
                } else {
                    self.refill();
                    //println!("Refill");
                }
            }

            let to_copy = std::cmp::min(self.have - self.buf_ptr, dest.len() - dest_ptr);
            dest[dest_ptr..dest_ptr + to_copy]
                .copy_from_slice(&self.buf[self.buf_ptr..self.buf_ptr + to_copy]);

            self.buf_ptr += to_copy;
            dest_ptr += to_copy;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl HasherStream {
    fn new() -> Self {
        // Fixed key for AES
        let key = GenericArray::from_slice(&[1; AES_KEY_SIZE]);

        let blocks_init = HasherStream::load4(&[[0; AES_BLOCK_SIZE]; 4]);
        HasherStream {
            aes: Aes128::new(key),
            in_blocks: blocks_init,
            buf: [0; AES_BLOCK_SIZE * 4],
        }
    }

    // Modified from RustCrypto aesni crate
    #[inline(always)]
    fn load4(input: &[[u8; 16]; 4]) -> [__m128i; 4] {
        let vals: Vec<_> = (0..4)
            .map(|i| Block::<Aes128>::from_slice(&input[i]))
            .collect();
        //let val = Block::<Aes128>::from_slice(input);

        // Safety: `loadu` supports unaligned loads
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            [
                _mm_loadu_si128(vals[0].as_ptr() as *const __m128i),
                _mm_loadu_si128(vals[1].as_ptr() as *const __m128i),
                _mm_loadu_si128(vals[2].as_ptr() as *const __m128i),
                _mm_loadu_si128(vals[3].as_ptr() as *const __m128i),
            ]
        }
    }

    #[inline(always)]
    fn store(val: __m128i, at: &mut [u8]) {
        debug_assert_eq!(at.len(), AES_BLOCK_SIZE);

        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            _mm_storeu_si128(at.as_mut_ptr() as *mut __m128i, val);
        }
    }

    fn set_input_blocks(&mut self, bl: &[[u8; 16]; 4]) {
        self.in_blocks = HasherStream::load4(bl);
    }

    fn set_input_blocks_from_seeds(&mut self, seeds: &(PrgSeed, PrgSeed, PrgSeed, PrgSeed)) {
        let bl = [seeds.0.key, seeds.1.key, seeds.2.key, seeds.3.key];
        self.set_input_blocks(&bl);
    }

    fn mmo4(&mut self) {
        let mut block4 = [GenericArray::clone_from_slice(&[0u8; 16]); 4];

        let mut in_blocks = [[0u8; AES_BLOCK_SIZE]; 4];
        for i in 0..4 {
            // Write input blocks into buffer
            HasherStream::store(self.in_blocks[i], &mut block4[i]);
            HasherStream::store(self.in_blocks[i], &mut in_blocks[i]);
        }

        // Batched AES call on 4 blocks
        self.aes.encrypt_blocks(&mut block4);

        for i in 0..4 {
            // Compute:   AES_k(in) XOR in
            block4[i]
                .iter_mut()
                .zip(in_blocks[i].iter())
                .for_each(|(x1, x2)| *x1 ^= *x2);
        }

        for i in 0..4 {
            self.buf[i * AES_BLOCK_SIZE..(i + 1) * AES_BLOCK_SIZE].copy_from_slice(&block4[i]);
        }
    }
}

impl HashOutput {
    // Ref https://github.com/sachaservan/vdpf/blob/main/src/mmo.c
    /// This emulates the hash function H' as defined in the paper https://eprint.iacr.org/2021/580.pdf.
    pub fn mmo_hash4to4(self: &HashOutput) -> HashOutput {
        HASHER_STREAM.with(|s_in| {
            let mut out = HashOutput {
                seeds: (
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                    PrgSeed::zero(),
                ),
            };

            let mut s = s_in.borrow_mut();

            s.set_input_blocks_from_seeds(&self.seeds);
            s.mmo4();

            out.seeds.0.key.copy_from_slice(&s.buf[0..AES_BLOCK_SIZE]);
            out.seeds
                .1
                .key
                .copy_from_slice(&s.buf[AES_BLOCK_SIZE..2 * AES_BLOCK_SIZE]);
            out.seeds
                .2
                .key
                .copy_from_slice(&s.buf[2 * AES_BLOCK_SIZE..3 * AES_BLOCK_SIZE]);
            out.seeds
                .3
                .key
                .copy_from_slice(&s.buf[3 * AES_BLOCK_SIZE..4 * AES_BLOCK_SIZE]);

            out
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PrfKey {
    pub key: [u8; AES_KEY_SIZE],
}

pub struct Prf {
    aes: Encryptor<Aes128>,
}

impl Prf {
    pub fn new(key: &[u8; AES_KEY_SIZE]) -> Self {
        Self {
            aes: Encryptor::<Aes128>::new(key.into()),
        }
    }

    pub fn compute_prf(&mut self, input: &[u8; AES_BLOCK_SIZE]) -> u128 {
        let mut output = [0u8; AES_BLOCK_SIZE].into();
        self.aes.encrypt_block_b2b_mut(input.into(), &mut output);

        u128::from_le_bytes(output.into())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        let zero = PrgSeed::zero();
        assert_eq!(zero.key.len(), AES_KEY_SIZE);
        for i in 0..AES_KEY_SIZE {
            assert_eq!(zero.key[i], 0u8);
        }
    }

    #[test]
    fn xor_zero() {
        let zero = PrgSeed::zero();
        let rand = PrgSeed::random();
        assert_ne!(rand.key, zero.key);

        let out = &zero ^ &rand;
        assert_eq!(out.key, rand.key);

        let out = &rand ^ &rand;
        assert_eq!(out.key, zero.key);
    }

    #[test]
    fn from_stream() {
        let rand = PrgSeed::random();
        let zero = PrgSeed::zero();
        let out = rand.expand();

        assert_ne!(out.seeds.0.key, zero.key);
        assert_ne!(out.seeds.1.key, zero.key);
        assert_ne!(out.seeds.0.key, out.seeds.1.key);
    }

    // #[test]
    // fn test_prf() {
    //     let key = [2u8; AES_KEY_SIZE];
    //     let mut prf = Prf::new(&key);

    //     let output = prf.compute_prf(&[1u8; AES_BLOCK_SIZE]);
    //     println !("out: {:?}", output);
    // }
}
