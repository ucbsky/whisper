use std::str::Utf8Error;

pub mod group;
pub mod grouptest;
pub mod prg;
pub mod utils;

pub const VERIFY_KEY_SIZE: usize = 16;

// Additive (Commutative) group
pub trait Group {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&mut self, other: &Self);
    fn mul(&mut self, other: &Self);
    fn negate(&mut self);
    fn sub(&mut self, other: &Self);
    fn minusone() -> Self;
}

pub trait SizedGroup {
    fn size() -> usize;
    fn core_size() -> usize;
    fn padded_size() -> usize;
}

pub trait Ordered {
    fn greater(&self, other: &Self) -> bool;
    fn positive(&self) -> bool;
}

pub trait Share: Group + prg::FromRng + Clone {
    fn random() -> Self {
        let mut out = Self::zero();
        out.randomize();
        out
    }

    fn share(&self) -> (Self, Self) {
        let s0 = Self::random();
        let mut s1 = self.clone();
        s1.sub(&s0);
        (s0, s1)
    }

    fn share_random() -> (Self, Self) {
        (Self::random(), Self::random())
    }
}

pub trait BitDecomposable: Group {
    /// MSB to LSB
    fn decompose(&self) -> Vec<bool>;
    fn as_bytes(&self) -> Vec<u8>;
    fn bitsize() -> usize;
}

impl BitDecomposable for u64 {
    fn decompose(&self) -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..64 {
            let bit = (*self & (1 << i)) != 0;
            out.push(bit);
        }
        out.reverse();
        out
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for i in 0..8 {
            let byte = (*self & (0xff << (8 * i))) >> (8 * i);
            out.push(byte as u8);
        }
        out
    }

    fn bitsize() -> usize {
        64
    }
}

impl BitDecomposable for u32 {
    fn decompose(&self) -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..32 {
            let bit = (*self & (1 << i)) != 0;
            out.push(bit);
        }
        out.reverse();
        out
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for i in 0..4 {
            let byte = (*self & (0xff << (8 * i))) >> (8 * i);
            out.push(byte as u8);
        }
        out
    }

    fn bitsize() -> usize {
        32
    }
}

impl BitDecomposable for u16 {
    fn decompose(&self) -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..16 {
            let bit = (*self & (1 << i)) != 0;
            out.push(bit);
        }
        out.reverse();
        out
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for i in 0..2 {
            let byte = (*self & (0xff << (8 * i))) >> (8 * i);
            out.push(byte as u8);
        }
        out
    }

    fn bitsize() -> usize {
        16
    }
}

impl BitDecomposable for u8 {
    fn decompose(&self) -> Vec<bool> {
        let mut out: Vec<bool> = Vec::new();
        for i in 0..8 {
            let bit = (*self & (1 << i)) != 0;
            out.push(bit);
        }
        out.reverse();
        out
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for i in 0..1 {
            let byte = (*self & (0xff << (8 * i))) >> (8 * i);
            out.push(byte);
        }
        out
    }

    fn bitsize() -> usize {
        8
    }
}

impl BitDecomposable for bool {
    fn decompose(&self) -> Vec<bool> {
        vec![*self]
    }

    fn as_bytes(&self) -> Vec<u8> {
        vec![*self as u8]
    }

    fn bitsize() -> usize {
        1
    }
}

pub(crate) fn u32_to_bits(nbits: u8, input: u32) -> Vec<bool> {
    assert!(nbits <= 32);

    let mut out: Vec<bool> = Vec::new();
    for i in 0..nbits {
        let bit = (input & (1 << i)) != 0;
        out.push(bit);
    }
    out
}

fn bits_to_u8(bits: &[bool]) -> u8 {
    assert_eq!(bits.len(), 8);
    let mut out = 0u8;
    for i in 0..8 {
        let b8: u8 = bits[i].into();
        out |= b8 << i;
    }

    out
}

pub fn bits_to_string(bits: &[bool]) -> Result<String, Utf8Error> {
    assert!(bits.len() % 8 == 0);

    let mut out: String = "".to_string();
    let byte_len = bits.len() / 8;
    for b in 0..byte_len {
        let byte = &bits[8 * b..8 * (b + 1)];
        let ubyte = bits_to_u8(byte);
        out.push_str(std::str::from_utf8(&[ubyte])?);
    }

    Ok(out)
}

pub fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let byte_len = bits.len() / 8;
    let mut out = Vec::with_capacity(byte_len);
    for b in 0..byte_len {
        let byte = &bits[8 * b..8 * (b + 1)];
        let ubyte = bits_to_u8(byte);
        out.push(ubyte);
    }
    out
}

pub fn bytes_to_bits(input: &[u8]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(input.len() << 3);
    for byte in input {
        let mut b = crate::u32_to_bits(8, (*byte).into());
        bits.append(&mut b);
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share() {
        let val = u64::random();
        println!("val: {}", val);
        //let vall = val as u16;
        //println!("val as u16: {}", vall);
        let (s0, s1) = val.share();
        let mut out = u64::zero();
        out.add(&s0);
        out.add(&s1);
        assert_eq!(out, val);
    }

    #[test]
    fn to_bits() {
        let empty: Vec<bool> = vec![];
        assert_eq!(u32_to_bits(0, 7), empty);
        assert_eq!(u32_to_bits(1, 0), vec![false]);
        assert_eq!(u32_to_bits(2, 0), vec![false, false]);
        assert_eq!(u32_to_bits(2, 3), vec![true, true]);
        assert_eq!(u32_to_bits(2, 1), vec![true, false]);
        assert_eq!(u32_to_bits(12, 65535), vec![true; 12]);
    }
}
