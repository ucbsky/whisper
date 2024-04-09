use std::{marker::PhantomData, mem};

use crate::prg::{self};
use prio::field::{Field128, Field64, FieldElement};
use typenum::Unsigned;

impl crate::Group for u64 {
    #[inline]
    fn zero() -> Self {
        0u64
    }

    #[inline]
    fn one() -> Self {
        1u64
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self = u64::wrapping_add(*self, *other);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self = u64::wrapping_mul(*self, *other);
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self = u64::wrapping_sub(*self, *other);
    }

    #[inline]
    fn negate(&mut self) {
        *self = u64::wrapping_neg(*self);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl crate::prg::FromRng for u64 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        *self = rng.next_u64();
    }
}

impl crate::Share for u64 {}

impl crate::Ordered for u64 {
    fn greater(&self, other: &Self) -> bool {
        if self > &(1 << 32) {
            if other < &(1 << 32) {
                false
            } else {
                self > other
            }
        } else if other > &(1 << 32) {
            true
        } else {
            self > other
        }
    }

    #[inline]
    fn positive(&self) -> bool {
        self < &(1 << 32)
    }
}

// =================

impl crate::Group for u32 {
    #[inline]
    fn zero() -> Self {
        0u32
    }

    #[inline]
    fn one() -> Self {
        1u32
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self = u32::wrapping_add(*self, *other);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self = u32::wrapping_mul(*self, *other);
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self = u32::wrapping_sub(*self, *other);
    }

    #[inline]
    fn negate(&mut self) {
        *self = u32::wrapping_neg(*self);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl crate::prg::FromRng for u32 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        *self = rng.next_u32();
    }
}

impl crate::Share for u32 {}

impl crate::Ordered for u32 {
    fn greater(&self, other: &Self) -> bool {
        if self > &(1 << 16) {
            if other < &(1 << 16) {
                false
            } else {
                self > other
            }
        } else if other > &(1 << 16) {
            true
        } else {
            self > other
        }
    }

    #[inline]
    fn positive(&self) -> bool {
        self < &(1 << 16)
    }
}
// ================

impl crate::Group for u16 {
    #[inline]
    fn zero() -> Self {
        0u16
    }

    #[inline]
    fn one() -> Self {
        1u16
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self = u16::wrapping_add(*self, *other);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self = u16::wrapping_mul(*self, *other);
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self = u16::wrapping_sub(*self, *other);
    }

    #[inline]
    fn negate(&mut self) {
        *self = u16::wrapping_neg(*self);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl crate::prg::FromRng for u16 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        *self = rng.next_u32() as u16;
    }
}

impl crate::Share for u16 {}

impl crate::Ordered for u16 {
    fn greater(&self, other: &Self) -> bool {
        if self > &(1 << 8) {
            if other < &(1 << 8) {
                false
            } else {
                self > other
            }
        } else if other > &(1 << 8) {
            true
        } else {
            self > other
        }
    }

    #[inline]
    fn positive(&self) -> bool {
        self < &(1 << 8)
    }
}

// ================

impl crate::Group for u8 {
    #[inline]
    fn zero() -> Self {
        0u8
    }

    #[inline]
    fn one() -> Self {
        1u8
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self = u8::wrapping_add(*self, *other);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self = u8::wrapping_mul(*self, *other);
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self = u8::wrapping_sub(*self, *other);
    }

    #[inline]
    fn negate(&mut self) {
        *self = u8::wrapping_neg(*self);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl crate::prg::FromRng for u8 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        *self = rng.next_u32() as u8;
    }
}

// ================

pub trait BatchedSampling {
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized;
}

impl BatchedSampling for u64 {
    /// Note the transmute function might not be safe because of alighnment and endianess,
    /// but we don't care since we use it only for generating random numbers
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized,
    {
        const BLOWUP_U8: usize = mem::size_of::<u64>() / mem::size_of::<u8>();
        let mut buf = vec![0u8; size * BLOWUP_U8];
        rng.fill_bytes(&mut buf);
        let mut out = vec![0u64; size];
        out.iter_mut()
            .zip(buf.chunks(BLOWUP_U8))
            .for_each(|(x, y)| {
                *x = unsafe {
                    mem::transmute::<[u8; BLOWUP_U8], u64>(
                        y.try_into()
                            .expect("Something went wrong with the chunk size"),
                    )
                };
            });
        out
    }
}

impl BatchedSampling for u32 {
    /// Note the transmute function might not be safe because of alighnment and endianess,
    /// but we don't care since we use it only for generating random numbers
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized,
    {
        const BLOWUP_U8: usize = mem::size_of::<u32>() / mem::size_of::<u8>();
        let mut buf = vec![0u8; size * BLOWUP_U8];
        rng.fill_bytes(&mut buf);
        let mut out = vec![0u32; size];
        out.iter_mut()
            .zip(buf.chunks(BLOWUP_U8))
            .for_each(|(x, y)| {
                *x = unsafe {
                    mem::transmute::<[u8; BLOWUP_U8], u32>(
                        y.try_into()
                            .expect("Something went wrong with the chunk size"),
                    )
                };
            });
        out
    }
}

impl BatchedSampling for u16 {
    /// Note the transmute function might not be safe because of alighnment and endianess,
    /// but we don't care since we use it only for generating random numbers
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized,
    {
        const BLOWUP_U8: usize = mem::size_of::<u16>() / mem::size_of::<u8>();
        let mut buf = vec![0u8; size * BLOWUP_U8];
        rng.fill_bytes(&mut buf);
        let mut out = vec![0u16; size];
        out.iter_mut()
            .zip(buf.chunks(BLOWUP_U8))
            .for_each(|(x, y)| {
                *x = unsafe {
                    mem::transmute::<[u8; BLOWUP_U8], u16>(
                        y.try_into()
                            .expect("Something went wrong with the chunk size"),
                    )
                };
            });
        out
    }
}

impl BatchedSampling for Field128 {
    /// Note the transmute function might not be safe because of alighnment and endianess,
    /// but we don't care since we use it only for generating random numbers
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut base_bytes = vec![0u8; size << 4];
        rng.fill_bytes(&mut base_bytes);
        Field128::byte_slice_into_vec(&base_bytes).unwrap()
    }
}

impl BatchedSampling for Field64 {
    /// Note the transmute function might not be safe because of alighnment and endianess,
    /// but we don't care since we use it only for generating random numbers
    fn get_random_batch(size: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut base_bytes = vec![0u8; size << 3];
        rng.fill_bytes(&mut base_bytes);
        Field64::byte_slice_into_vec(&base_bytes).unwrap()
    }
}
// ===================

impl crate::Group for bool {
    #[inline]
    fn zero() -> Self {
        false
    }

    #[inline]
    fn one() -> Self {
        true
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self ^= *other;
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self &= *other;
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self ^= *other;
    }

    #[inline]
    fn negate(&mut self) {}

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl crate::prg::FromRng for bool {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        // This is inefficient, but I don't think this is ever called, so this is fine
        *self = (rng.next_u32() & 1) == 1;
    }
}

impl crate::Share for bool {}

// ===================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntModN<N: Unsigned> {
    pub val: u16,
    phantom: PhantomData<N>,
}

impl<N> IntModN<N>
where
    N: Unsigned,
{
    pub fn from_u16(input: u16) -> Self {
        IntModN {
            val: input % N::to_u16(),
            phantom: PhantomData::<N>,
        }
    }
}

impl<N> crate::Group for IntModN<N>
where
    N: Unsigned,
{
    fn zero() -> Self {
        Self {
            val: 0,
            phantom: PhantomData::<N>,
        }
    }

    fn one() -> Self {
        Self {
            val: 0,
            phantom: PhantomData::<N>,
        }
    }

    fn add(&mut self, other: &Self) {
        self.val = (self.val + other.val) % (N::to_u16());
    }

    fn mul(&mut self, other: &Self) {
        self.val = (self.val * other.val) % (N::to_u16());
    }

    fn negate(&mut self) {
        self.val = N::to_u16() - self.val;
    }

    fn sub(&mut self, other: &Self) {
        self.val = self.val.wrapping_sub(other.val) % (N::to_u16());
    }

    fn minusone() -> Self {
        Self {
            val: N::to_u16() - 1,
            phantom: PhantomData::<N>,
        }
    }
}

impl<N> crate::prg::FromRng for IntModN<N>
where
    N: Unsigned,
{
    fn from_rng(&mut self, stream: &mut (impl rand::Rng + rand_core::RngCore)) {
        self.val = stream.gen_range(0..N::to_u16());
    }
}

impl<N> crate::BitDecomposable for IntModN<N>
where
    N: Unsigned,
{
    fn decompose(&self) -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..IntModN::<N>::bitsize() {
            let bit = (self.val & (1 << i)) != 0;
            out.push(bit);
        }
        out.reverse();
        out
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for i in 0..2 {
            let byte = (self.val & (0xff << (8 * i))) >> (8 * i);
            out.push(byte as u8);
        }
        out
    }

    fn bitsize() -> usize {
        assert_ne!(N::to_u16(), 0);
        (16 - N::to_u16().leading_zeros() - 1) as usize
        // magic number 16: because we never have more than 65536 buckets
    }
}

// ===================

impl crate::Group for Field128 {
    #[inline]
    fn zero() -> Self {
        <Field128 as FieldElement>::zero()
    }

    #[inline]
    fn one() -> Self {
        <Field128 as FieldElement>::one()
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self += *other;
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self *= *other;
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self -= *other;
    }

    #[inline]
    fn negate(&mut self) {
        use std::ops::Neg;
        *self = self.neg();
    }

    #[inline]
    fn minusone() -> Self {
        use std::ops::Neg;
        <Field128 as FieldElement>::one().neg()
    }
}

impl crate::prg::FromRng for Field128 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        // Optimized version
        *self = Field128::try_from_random(rng.gen::<[u8; 16]>().as_slice()).unwrap();
    }
}

impl crate::Share for Field128 {}

impl crate::Group for Field64 {
    #[inline]
    fn zero() -> Self {
        <Field64 as FieldElement>::zero()
    }

    #[inline]
    fn one() -> Self {
        <Field64 as FieldElement>::one()
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        *self += *other;
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        *self *= *other;
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        *self -= *other;
    }

    #[inline]
    fn negate(&mut self) {
        use std::ops::Neg;
        *self = self.neg();
    }

    #[inline]
    fn minusone() -> Self {
        use std::ops::Neg;
        <Field64 as FieldElement>::one().neg()
    }
}

impl crate::prg::FromRng for Field64 {
    fn from_rng(&mut self, rng: &mut impl rand::Rng) {
        // Optimized version
        *self = Field64::try_from_random(rng.gen::<[u8; 8]>().as_slice()).unwrap();
    }
}

impl crate::Share for Field64 {}
// ===================

impl<G: crate::Group + Clone> crate::Group for (G, G) {
    #[inline]
    fn zero() -> Self {
        (G::zero(), G::zero())
    }

    #[inline]
    fn one() -> Self {
        (G::one(), G::one())
    }

    #[inline]
    fn add(&mut self, other: &Self) {
        self.0.add(&other.0);
        self.1.add(&other.1);
    }

    #[inline]
    fn mul(&mut self, other: &Self) {
        self.0.mul(&other.0);
        self.1.mul(&other.1);
    }

    #[inline]
    fn negate(&mut self) {
        self.0.negate();
        self.1.negate();
    }

    #[inline]
    fn sub(&mut self, other: &Self) {
        self.0.sub(&other.0);
        self.1.sub(&other.1);
    }

    #[inline]
    fn minusone() -> Self {
        let mut out = Self::one();
        out.negate();
        out
    }
}

impl<G: crate::Group + Clone + prg::FromRng> crate::prg::FromRng for (G, G) {
    fn from_rng(&mut self, rng: &mut (impl rand::Rng + rand_core::RngCore)) {
        self.0.from_rng(rng);
        self.1.from_rng(rng);
    }
}

impl<G: crate::Group + Clone + prg::FromRng> crate::Share for (G, G) {}
