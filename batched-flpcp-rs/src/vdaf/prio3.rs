// SPDX-License-Identifier: MPL-2.0

//! Implementation of the Prio3 VDAF [[draft-irtf-cfrg-vdaf-07]].
//!
//! **WARNING:** This code has not undergone significant security analysis. Use at your own risk.
//!
//! Prio3 is based on the Prio system desigend by Dan Boneh and Henry Corrigan-Gibbs and presented
//! at NSDI 2017 [[CGB17]]. However, it incorporates a few techniques from Boneh et al., CRYPTO
//! 2019 [[BBCG+19]], that lead to substantial improvements in terms of run time and communication
//! cost. The security of the construction was analyzed in [[DPRS23]].
//!
//! Prio3 is a transformation of a Fully Linear Proof (FLP) system [[draft-irtf-cfrg-vdaf-07]] into
//! a VDAF. The base type, [`Prio3`], supports a wide variety of aggregation functions, some of
//! which are instantiated here:
//!
//! - [`Prio3Count`] for aggregating a counter (*)
//! - [`Prio3Sum`] for copmputing the sum of integers (*)
//! - [`Prio3SumVec`] for aggregating a vector of integers
//! - [`Prio3Histogram`] for estimating a distribution via a histogram (*)
//!
//! Additional types can be constructed from [`Prio3`] as needed.
//!
//! (*) denotes that the type is specified in [[draft-irtf-cfrg-vdaf-07]].
//!
//! [BBCG+19]: https://ia.cr/2019/188
//! [CGB17]: https://crypto.stanford.edu/prio/
//! [DPRS23]: https://ia.cr/2023/130
//! [draft-irtf-cfrg-vdaf-07]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/07/

use super::xof::XofShake128;
use super::xof::XofShake256;
#[cfg(feature = "experimental")]
use super::AggregatorWithNoise;
use super::BatchAggregator;
use super::BatchClient;
use super::VdafBatchedKey;
use crate::codec::{CodecError, Decode, Encode, ParameterizedDecode};
#[cfg(feature = "experimental")]
use crate::dp::DifferentialPrivacyStrategy;
use crate::field::Field128;
use crate::field::Field64;
use crate::field::{decode_fieldvec, FftFriendlyFieldElement, FieldElement};
#[cfg(feature = "multithreaded")]
use crate::flp::gadgets::ParallelSumMultithreaded;
#[cfg(feature = "experimental")]
use crate::flp::gadgets::PolyEval;
use crate::flp::gadgets::{Mul, ParallelSum};
#[cfg(feature = "experimental")]
use crate::flp::types::fixedpoint_l2::FixedPointBoundedL2VecSum;
use crate::flp::types::{Average, Count, Histogram, Sum, SumVec};
use crate::flp::Type;
#[cfg(feature = "experimental")]
use crate::flp::TypeWithNoise;
use crate::prng::Prng;
use crate::vdaf::xof::{IntoFieldVec, Seed, Xof};
use crate::vdaf::{
    Aggregatable, AggregateShare, Aggregator, Client, Collector, OutputShare, PrepareTransition,
    Share, ShareDecodingParameter, Vdaf, VdafError,
};

use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::io::Cursor;
use std::iter::{self, IntoIterator};
use std::marker::PhantomData;
use subtle::{Choice, ConstantTimeEq};

const DST_MEASUREMENT_SHARE: u16 = 1;
const DST_PROOF_SHARE: u16 = 2;
const DST_JOINT_RANDOMNESS: u16 = 3;
const DST_PROVE_RANDOMNESS: u16 = 4;
const DST_QUERY_RANDOMNESS: u16 = 5;
const DST_JOINT_RAND_SEED: u16 = 6;
const DST_JOINT_RAND_PART: u16 = 7;
const DST_QUERY_RAND_PART: u16 = 8;
const DST_RLC_RANDOMNESS: u16 = 9;
const DST_HASH_PART: u16 = 10;

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate result is the sum.
pub type Prio3Count = Prio3<Count<Field64>, XofShake128, 16>;

/// The count type. Each measurement is an integer (represented in a 128 bit field) in `[0,2)` and the aggregate result is the sum.
pub type Prio3Count128 = Prio3<Count<Field128>, XofShake128, 16>;

/// The count type. Each measurement is an integer (represented in a 128 bit field) in `[0,2)` and the aggregate result is the sum.
/// Uses 2 128 bit fields to achieve 128 bit security with Whisper
pub type Prio3Count256 = Prio3<Count<Field128>, XofShake256, 32>;

impl Prio3Count {
    /// Construct an instance of Prio3Count with the given number of aggregators.
    pub fn new_count(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Count::new())
    }
}

impl Prio3Count128 {
    /// Construct an instance of Prio3Count with the given number of aggregators.
    pub fn new_count_relevant(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Count::new())
    }
}

impl Prio3Count256 {
    /// Construct an instance of Prio3Count with the given number of aggregators.
    pub fn new_count_256(num_aggregators: u8) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Count::new())
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2^bits)` and the
/// aggregate is the element-wise sum.
pub type Prio3SumVec = Prio3<SumVec<Field64, ParallelSum<Field64, Mul<Field64>>>, XofShake128, 16>;

impl Prio3SumVec {
    /// Construct an instance of Prio3SumVec with the given number of aggregators. `bits` defines
    /// the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_sum_vec(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len, chunk_length)?)
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2^bits)` and the
/// aggregate is the element-wise sum, in 128 bit field
pub type Prio3SumVec128 =
    Prio3<SumVec<Field128, ParallelSum<Field128, Mul<Field128>>>, XofShake128, 16>;

impl Prio3SumVec128 {
    /// Construct an instance of Prio3SumVec with the given number of aggregators. `bits` defines
    /// the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_sum_vec_128(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len, chunk_length)?)
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2^bits)` and the
/// aggregate is the element-wise sum, in 128 bit field, ran 2x for 256 bit security
pub type Prio3SumVec256 =
    Prio3<SumVec<Field128, ParallelSum<Field128, Mul<Field128>>>, XofShake128, 16>;

impl Prio3SumVec256 {
    /// Construct an instance of Prio3SumVec with the given number of aggregators. `bits` defines
    /// the bit width of each summand of the measurement; `len` defines the length of the
    /// measurement vector.
    pub fn new_sum_vec_256(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len, chunk_length)?)
    }
}

/// Like [`Prio3SumVec`] except this type uses multithreading to improve sharding and preparation
/// time. Note that the improvement is only noticeable for very large input lengths.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3SumVecMultithreaded =
    Prio3<SumVec<Field64, ParallelSumMultithreaded<Field64, Mul<Field64>>>, XofShake128, 16>;

#[cfg(feature = "multithreaded")]
impl Prio3SumVecMultithreaded {
    /// Construct an instance of Prio3SumVecMultithreaded with the given number of
    /// aggregators. `bits` defines the bit width of each summand of the measurement; `len` defines
    /// the length of the measurement vector.
    pub fn new_sum_vec_multithreaded(
        num_aggregators: u8,
        bits: usize,
        len: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, SumVec::new(bits, len, chunk_length)?)
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
pub type Prio3Sum = Prio3<Sum<Field64>, XofShake128, 16>;

impl Prio3Sum {
    /// Construct an instance of Prio3Sum with the given number of aggregators and required bit
    /// length. The bit length must not exceed 64.
    pub fn new_sum(num_aggregators: u8, bits: usize) -> Result<Self, VdafError> {
        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Prio3::new(num_aggregators, Sum::new(bits)?)
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
pub type Prio3Sum256 = Prio3<Sum<Field128>, XofShake256, 32>;

impl Prio3Sum256 {
    /// Construct an instance of Prio3Sum with the given number of aggregators and required bit
    /// length. The bit length must not exceed 64.
    pub fn new_sum_256(num_aggregators: u8, bits: usize) -> Result<Self, VdafError> {
        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Prio3::new(num_aggregators, Sum::new(bits)?)
    }
}

/// The fixed point vector sum type. Each measurement is a vector of fixed point numbers
/// and the aggregate is the sum represented as 64-bit floats. The preparation phase
/// ensures the L2 norm of the input vector is < 1.
///
/// This is useful for aggregating gradients in a federated version of
/// [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) with
/// [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy),
/// useful, e.g., for [differentially private deep learning](https://arxiv.org/pdf/1607.00133.pdf).
/// The bound on input norms is required for differential privacy. The fixed point representation
/// allows an easy conversion to the integer type used in internal computation, while leaving
/// conversion to the client. The model itself will have floating point parameters, so the output
/// sum has that type as well.
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub type Prio3FixedPointBoundedL2VecSum<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        ParallelSum<Field64, PolyEval<Field64>>,
        ParallelSum<Field64, Mul<Field64>>,
    >,
    XofShake128,
    16,
>;

/// The fixed point vector sum type. Each measurement is a vector of fixed point numbers
/// and the aggregate is the sum represented as 64-bit floats. The verification function
/// ensures the L2 norm of the input vector is < 1.
#[cfg(all(feature = "experimental", feature = "multithreaded"))]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "experimental", feature = "multithreaded")))
)]
pub type Prio3FixedPointBoundedL2VecSumMultithreaded<Fx> = Prio3<
    FixedPointBoundedL2VecSum<
        Fx,
        ParallelSumMultithreaded<Field64, PolyEval<Field64>>,
        ParallelSumMultithreaded<Field64, Mul<Field64>>,
    >,
    XofShake128,
    16,
>;

#[cfg(all(feature = "experimental", feature = "multithreaded"))]
impl<Fx: Fixed + CompatibleFloat> Prio3FixedPointBoundedL2VecSumMultithreaded<Fx> {
    /// Construct an instance of this VDAF with the given number of aggregators and number of
    /// vector entries.
    pub fn new_fixedpoint_boundedl2_vec_sum_multithreaded(
        num_aggregators: u8,
        entries: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Prio3::new(num_aggregators, FixedPointBoundedL2VecSum::new(entries)?)
    }
}

/// The histogram type. Each measurement is an integer in `[0, length)` and the result is a
/// histogram counting the number of occurrences of each measurement.
pub type Prio3Histogram =
    Prio3<Histogram<Field64, ParallelSum<Field64, Mul<Field64>>>, XofShake128, 16>;

impl Prio3Histogram {
    /// Constructs an instance of Prio3Histogram with the given number of aggregators,
    /// number of buckets, and parallel sum gadget chunk length.
    pub fn new_histogram(
        num_aggregators: u8,
        length: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Histogram::new(length, chunk_length)?)
    }
}

/// The histogram type. Each measurement is an integer in `[0, length)` and the result is a
/// histogram counting the number of occurrences of each measurement.
/// Uses 2 parallel 128 bit runs to achieve 128 bit security with Whisper
pub type Prio3Histogram256 =
    Prio3<Histogram<Field128, ParallelSum<Field128, Mul<Field128>>>, XofShake128, 16>;

impl Prio3Histogram256 {
    /// Constructs an instance of Prio3Histogram with the given number of aggregators,
    /// number of buckets, and parallel sum gadget chunk length.
    pub fn new_histogram_256(
        num_aggregators: u8,
        length: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Histogram::new(length, chunk_length)?)
    }
}

/// Like [`Prio3Histogram`] except this type uses multithreading to improve sharding and preparation
/// time. Note that this improvement is only noticeable for very large input lengths.
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3HistogramMultithreaded =
    Prio3<Histogram<Field64, ParallelSumMultithreaded<Field64, Mul<Field64>>>, XofShake128, 16>;

#[cfg(feature = "multithreaded")]
impl Prio3HistogramMultithreaded {
    /// Construct an instance of Prio3HistogramMultithreaded with the given number of aggregators,
    /// number of buckets, and parallel sum gadget chunk length.
    pub fn new_histogram_multithreaded(
        num_aggregators: u8,
        length: usize,
        chunk_length: usize,
    ) -> Result<Self, VdafError> {
        Prio3::new(num_aggregators, Histogram::new(length, chunk_length)?)
    }
}

/// The average type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and
/// the aggregate is the arithmetic average.
pub type Prio3Average = Prio3<Average<Field64>, XofShake128, 16>;

impl Prio3Average {
    /// Construct an instance of Prio3Average with the given number of aggregators and required bit
    /// length. The bit length must not exceed 64.
    pub fn new_average(num_aggregators: u8, bits: usize) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Ok(Prio3 {
            num_aggregators,
            typ: Average::new(bits)?,
            phantom: PhantomData,
        })
    }
}

/// The average type. Uses two parallel runs with 128 bit field, to provide 128 bit security.
pub type Prio3Average256 = Prio3<Average<Field128>, XofShake128, 16>;

impl Prio3Average256 {
    /// Construct an instance of Prio3Average with the given number of aggregators and required bit
    /// length. The bit length must not exceed 64.
    pub fn new_average_256(num_aggregators: u8, bits: usize) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({bits}) exceeds limit for aggregate type (64)"
            )));
        }

        Ok(Prio3 {
            num_aggregators,
            typ: Average::new(bits)?,
            phantom: PhantomData,
        })
    }
}

/// The base type for Prio3.
///
/// An instance of Prio3 is determined by:
///
/// - a [`Type`] that defines the set of valid input measurements; and
/// - a [`Xof`] for deriving vectors of field elements from seeds.
///
/// New instances can be defined by aliasing the base type. For example, [`Prio3Count`] is an alias
/// for `Prio3<Count<Field64>, XofShake128, 16>`.
///
/// ```
/// use prio::vdaf::{
///     Aggregator, Client, Collector, PrepareTransition,
///     prio3::Prio3,
/// };
/// use rand::prelude::*;
///
/// let num_shares = 2;
/// let vdaf = Prio3::new_count(num_shares).unwrap();
///
/// let mut out_shares = vec![vec![]; num_shares.into()];
/// let mut rng = thread_rng();
/// let verify_key = rng.gen();
/// let measurements = [0, 1, 1, 1, 0];
/// for measurement in measurements {
///     // Shard
///     let nonce = rng.gen::<[u8; 16]>();
///     let (public_share, input_shares) = vdaf.shard(&measurement, &nonce).unwrap();
///
///     // Prepare
///     let mut prep_states = vec![];
///     let mut prep_shares = vec![];
///     for (agg_id, input_share) in input_shares.iter().enumerate() {
///         let (state, share) = vdaf.prepare_init(
///             &verify_key,
///             agg_id,
///             &(),
///             &nonce,
///             &public_share,
///             input_share
///         ).unwrap();
///         prep_states.push(state);
///         prep_shares.push(share);
///     }
///     let prep_msg = vdaf.prepare_shares_to_prepare_message(&(), prep_shares).unwrap();
///
///     for (agg_id, state) in prep_states.into_iter().enumerate() {
///         let out_share = match vdaf.prepare_step(state, prep_msg.clone()).unwrap() {
///             PrepareTransition::Finish(out_share) => out_share,
///             _ => panic!("unexpected transition"),
///         };
///         out_shares[agg_id].push(out_share);
///     }
/// }
///
/// // Aggregate
/// let agg_shares = out_shares.into_iter()
///     .map(|o| vdaf.aggregate(&(), o).unwrap());
///
/// // Unshard
/// let agg_res = vdaf.unshard(&(), agg_shares, measurements.len()).unwrap();
/// assert_eq!(agg_res, 3);
/// ```
#[derive(Clone, Debug)]
pub struct Prio3<T, P, const SEED_SIZE: usize>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    num_aggregators: u8,
    typ: T,
    phantom: PhantomData<P>,
}

impl<T, P, const SEED_SIZE: usize> Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Construct an instance of this Prio3 VDAF with the given number of aggregators and the
    /// underlying type.
    pub fn new(num_aggregators: u8, typ: T) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;
        Ok(Self {
            num_aggregators,
            typ,
            phantom: PhantomData,
        })
    }

    /// The output length of the underlying FLP.
    pub fn output_len(&self) -> usize {
        self.typ.output_len()
    }

    /// The verifier length of the underlying FLP.
    pub fn verifier_len(&self) -> usize {
        self.typ.verifier_len()
    }

    fn derive_joint_rand_seed<'a>(
        parts: impl Iterator<Item = &'a Seed<SEED_SIZE>>,
    ) -> Seed<SEED_SIZE> {
        let mut xof = P::init(
            &[0; SEED_SIZE],
            &Self::domain_separation_tag(DST_JOINT_RAND_SEED),
        );
        for part in parts {
            xof.update(part.as_ref());
        }
        xof.into_seed()
    }

    fn random_size(&self) -> usize {
        if self.typ.joint_rand_len() == 0 {
            // Two seeds per helper for measurement and proof shares, plus one seed for proving
            // randomness.
            (usize::from(self.num_aggregators - 1) * 2 + 1 + usize::from(self.num_aggregators))
                * SEED_SIZE
        } else {
            (
                // Two seeds per helper for measurement and proof shares
                usize::from(self.num_aggregators - 1) * 2
                // One seed for proving randomness
                + 1
                // One seed per aggregator for joint randomness blinds
                + usize::from(self.num_aggregators)
                // One seed per aggregator for query randomness blinds
                + usize::from(self.num_aggregators)
            ) * SEED_SIZE
        }
    }

    fn random_size_double_field(&self) -> usize {
        if self.typ.joint_rand_len() == 0 {
            // Two seeds per helper for measurement and proof shares, plus two seeds for proving
            // randomness (one for each parallel fused run).
            (usize::from(self.num_aggregators - 1) * 3 + 1 + 1 + usize::from(self.num_aggregators))
                * SEED_SIZE
        } else {
            (
                // Three seeds per helper for measurement and 2x proof shares
                usize::from(self.num_aggregators - 1) * 3
                // two seeds for proving randomness (one for each parallel fused run)
                + 1 + 1
                // One seed per aggregator for joint randomness blinds
                + usize::from(self.num_aggregators)
                // One seed per aggregator for query randomness blinds
                + usize::from(self.num_aggregators)
            ) * SEED_SIZE
        }
    }

    /// When 256-bit isn't natively supported, we use a tuple of two 128-bit fields to get same soundess.
    #[allow(clippy::type_complexity)]
    pub(crate) fn shard_with_random_new<const N: usize>(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; N],
        random: &[u8],
    ) -> Result<
        (
            Prio3PublicShare<SEED_SIZE>,
            Vec<Prio3InputShare<T::Field, SEED_SIZE>>,
            Vec<Prio3ProofShare<T::Field, SEED_SIZE>>,
            Prio3PublicShare<SEED_SIZE>,
            Prio3PublicProof<T::Field, SEED_SIZE>,
            Prio3PublicProof<T::Field, SEED_SIZE>,
            Vec<Prio3Blinds<SEED_SIZE>>,
        ),
        VdafError,
    > {
        if random.len() != self.random_size_double_field() {
            return Err(VdafError::Uncategorized(
                "incorrect random input length".to_string(),
            ));
        }
        let mut random_seeds = random.chunks_exact(SEED_SIZE);
        let num_aggregators = self.num_aggregators;
        let encoded_measurement = self.typ.encode_measurement(measurement)?;
        let enc_measurement = encoded_measurement.clone();

        // Generate the measurement shares and compute the joint randomness.
        let mut helper_shares_first = Vec::with_capacity(num_aggregators as usize - 1);
        let mut helper_shares_second = Vec::with_capacity(num_aggregators as usize - 1);
        let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        // println!("Is 3 move? {}", self.typ.joint_rand_len());
        let mut leader_measurement_share = encoded_measurement.clone();
        for agg_id in 1..num_aggregators {
            // The Option from the ChunksExact iterator is okay to unwrap because we checked that
            // the randomness slice is long enough for this VDAF. The slice-to-array conversion
            // Result is okay to unwrap because the ChunksExact iterator always returns slices of
            // the correct length.
            let measurement_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let proof_share_seed_first = random_seeds.next().unwrap().try_into().unwrap();
            let proof_share_seed_second = random_seeds.next().unwrap().try_into().unwrap();
            let measurement_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &Seed(measurement_share_seed),
                &Self::domain_separation_tag(DST_MEASUREMENT_SHARE),
                &[agg_id],
            ));
            let joint_rand_blind =
                if let Some(helper_joint_rand_parts) = helper_joint_rand_parts.as_mut() {
                    let joint_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                    // 1st thing inside RO_i is the ith blind
                    let mut joint_rand_part_xof = P::init(
                        &joint_rand_blind,
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    // 2nd thing inside RO_i is the ith aggregator ID
                    joint_rand_part_xof.update(&[agg_id]); // Aggregator ID
                                                           // 3rd thing inside RO_i is the nonce
                    joint_rand_part_xof.update(nonce);

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                        y.encode(&mut encoding_buffer);
                        // 4th thing inside RO_i is the encoded ith measurement share
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Compute the RO_i output from the 4 things that were fed into it (see above)
                    helper_joint_rand_parts.push(joint_rand_part_xof.into_seed());

                    Some(joint_rand_blind)
                } else {
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                    }
                    None
                };
            let helper_first = HelperShare::from_seeds(
                measurement_share_seed,
                proof_share_seed_first,
                joint_rand_blind,
            );
            helper_shares_first.push(helper_first);
            // Input should be same for both runs
            let helper_second = HelperShare::from_seeds(
                measurement_share_seed,
                proof_share_seed_second,
                joint_rand_blind,
            );
            helper_shares_second.push(helper_second);
        }

        //let mut helper_shares_second = helper_shares_first.clone();

        let mut leader_blind_opt = None;
        let public_share = Prio3PublicShare {
            joint_rand_parts: helper_joint_rand_parts
                .as_ref()
                .map(|helper_joint_rand_parts| {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut joint_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    joint_rand_part_xof.update(&[0]); // Aggregator ID
                    joint_rand_part_xof.update(nonce);
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_measurement_share.iter() {
                        x.encode(&mut encoding_buffer);
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }
                    leader_blind_opt = Some(leader_blind);

                    let leader_joint_rand_seed_part = joint_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    // Put all the RO_i results (joint randomness shares) into a vector - will be used to derive the joint randomness
                    vec.push(leader_joint_rand_seed_part);
                    vec.extend(helper_joint_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Compute the joint randomness.
        let joint_rand_fused: Vec<T::Field> = public_share
            .joint_rand_parts
            .as_ref()
            .map(|joint_rand_parts| {
                // Combine all RO_i results to the main RO result
                let joint_rand_seed = Self::derive_joint_rand_seed(joint_rand_parts.iter());
                P::seed_stream(
                    &joint_rand_seed,
                    &Self::domain_separation_tag(DST_JOINT_RANDOMNESS),
                    &[],
                )
                .into_field_vec(2 * self.typ.joint_rand_len()) // This factor of 2 is for soundness to hold; see it as an alternative to using a large 256-bit field. We are emulating the same soundess as a 256-bit field by using two 128-bit fields. Each RO output can be seen as tuple of size two with each element being a 128-bit field.
            })
            .unwrap_or_default();

        let joint_rand_split;
        if self.typ.joint_rand_len() > 0 {
            joint_rand_split = joint_rand_fused
                .chunks_exact(self.typ.joint_rand_len())
                .map(|x| x.to_vec())
                .collect::<Vec<_>>();

            assert_eq!(joint_rand_split.len(), 2);
        } else {
            // Placeholder
            joint_rand_split = vec![joint_rand_fused.clone(); 2];
        }

        // Run the proof-generation algorithm.
        // First run
        let prove_rand_seed_first = random_seeds.next().unwrap().try_into().unwrap();
        let prove_rand_first = P::seed_stream(
            &Seed::from_bytes(prove_rand_seed_first),
            &Self::domain_separation_tag(DST_PROVE_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len());
        let mut leader_proof_share_first = self.typ.prove(
            &encoded_measurement,
            &prove_rand_first,
            &joint_rand_split[0],
        )?;

        let proof_first = leader_proof_share_first.clone();

        // Generate the proof shares and distribute the joint randomness seed hints.
        for (j, helper) in helper_shares_first.iter_mut().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            for (x, y) in leader_proof_share_first
                .iter_mut()
                .zip(proof_share_prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }
        }

        // Prep the output messages.
        let mut out_first = Vec::with_capacity(num_aggregators as usize);
        out_first.push(Prio3InputShare {
            measurement_share: Share::Leader(leader_measurement_share.clone()),
            proof_share: Share::Leader(leader_proof_share_first.clone()),
            joint_rand_blind: leader_blind_opt.clone(),
        });

        for helper in helper_shares_first.clone().into_iter() {
            out_first.push(Prio3InputShare {
                measurement_share: Share::Helper(helper.measurement_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_blind: helper.joint_rand_blind,
            });
        }

        // Second run
        let prove_rand_seed_second = random_seeds.next().unwrap().try_into().unwrap();
        let prove_rand_second = P::seed_stream(
            &Seed::from_bytes(prove_rand_seed_second),
            &Self::domain_separation_tag(DST_PROVE_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len());
        let mut leader_proof_share_second = self.typ.prove(
            &encoded_measurement,
            &prove_rand_second,
            &joint_rand_split[1],
        )?;

        let proof_second = leader_proof_share_second.clone();

        // Generate the proof shares and distribute the joint randomness seed hints.
        for (j, helper) in helper_shares_second.iter_mut().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            for (x, y) in leader_proof_share_second
                .iter_mut()
                .zip(proof_share_prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }
        }

        // Prep the output messages
        let mut out_second = Vec::with_capacity(num_aggregators as usize);
        out_second.push(Prio3ProofShare {
            proof_share: Share::Leader(leader_proof_share_second.clone()),
        });

        for helper in helper_shares_second.clone().into_iter() {
            out_second.push(Prio3ProofShare {
                proof_share: Share::Helper(helper.proof_share),
            });
        }

        // Common to both runs
        // Deriving query_rand from Fiat-Shamir
        let mut helper_query_rand_parts = if self.typ.query_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        let mut helper_query_rand_blinds = Vec::with_capacity(num_aggregators as usize - 1);

        // Helper's RO_i queries
        for (j, (helper_first, helper_second)) in helper_shares_first
            .into_iter()
            .zip(helper_shares_second.into_iter())
            .enumerate()
        {
            let proof_share_prng_first: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper_first.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            let proof_share_prng_second: Prng<T::Field, _> =
                Prng::from_seed_stream(P::seed_stream(
                    &helper_second.proof_share,
                    &Self::domain_separation_tag(DST_PROOF_SHARE),
                    &[j as u8 + 1],
                ));

            // There will be only one set of blinds because blinds are per RO call which shared by both runs
            let query_rand_blind =
                if let Some(helper_query_rand_parts) = helper_query_rand_parts.as_mut() {
                    let query_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                    // 1st thing inside RO_i is the ith blind
                    let mut query_rand_part_xof = P::init(
                        &query_rand_blind,
                        &Self::domain_separation_tag(DST_QUERY_RAND_PART),
                    );

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in joint_rand_fused.iter() {
                        x.encode(&mut encoding_buffer);
                        // 2nd thing inside RO_i is the previous round joint randomness (captures transcript till previous round)
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (_x, y) in leader_proof_share_first
                        .iter()
                        .zip(proof_share_prng_first)
                        .take(self.typ.proof_len())
                    {
                        y.encode(&mut encoding_buffer);
                        // 3rd thing inside RO_i is the encoded ith proof share of the first proof
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (_x, y) in leader_proof_share_second
                        .iter()
                        .zip(proof_share_prng_second)
                        .take(self.typ.proof_len())
                    {
                        y.encode(&mut encoding_buffer);
                        // 4th thing inside RO_i is the encoded ith proof share of the second proof
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Compute the RO_i output from the 4 things that were fed into it (see above)
                    helper_query_rand_parts.push(query_rand_part_xof.into_seed());

                    Some(query_rand_blind)
                } else {
                    None
                };

            helper_query_rand_blinds.push(query_rand_blind);
        }

        // Leader's RO_i queries
        let mut query_leader_blind_opt = None;
        let public_share_second = Prio3PublicShare {
            // Repurposing the joint_rand_parts field to store the query_rand_parts
            joint_rand_parts: helper_query_rand_parts
                .as_ref()
                .map(|helper_query_rand_parts| {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut query_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &Self::domain_separation_tag(DST_QUERY_RAND_PART),
                    );

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in joint_rand_fused.iter() {
                        x.encode(&mut encoding_buffer);
                        // 2nd thing inside RO_i is the previous round joint randomness (captures transcript till previous round)
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // First run included
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_proof_share_first.iter() {
                        x.encode(&mut encoding_buffer);
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Second run included
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_proof_share_second.iter() {
                        x.encode(&mut encoding_buffer);
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    query_leader_blind_opt = Some(leader_blind);

                    let leader_query_rand_seed_part = query_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    // Put all the RO_i results (joint randomness shares) into a vector - will be used to derive the joint randomness
                    vec.push(leader_query_rand_seed_part);
                    vec.extend(helper_query_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Compute the query randomness.
        let query_rand_fused: Vec<T::Field> = public_share_second
            .joint_rand_parts
            .as_ref()
            .map(|query_rand_parts| {
                // Combine all RO_i results to the main RO result
                let query_rand_seed = Self::derive_joint_rand_seed(query_rand_parts.iter());
                P::seed_stream(
                    &query_rand_seed,
                    &Self::domain_separation_tag(DST_QUERY_RANDOMNESS),
                    &[],
                )
                .into_field_vec(2 * self.typ.query_rand_len())
            })
            .unwrap_or_default();

        let query_rand_split = query_rand_fused
            .chunks_exact(self.typ.query_rand_len())
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        assert_eq!(query_rand_split.len(), 2);

        // Run the query algorithm
        let public_proof_run1 = Prio3PublicProof {
            query_answers: self.typ.query(
                &enc_measurement,
                &proof_first,
                &query_rand_split[0],
                &joint_rand_split[0],
                1,
            )?,
        };

        let public_proof_run2 = Prio3PublicProof {
            query_answers: self.typ.query(
                &enc_measurement,
                &proof_second,
                &query_rand_split[1],
                &joint_rand_split[1],
                1,
            )?,
        };

        // Prep the query blind messages.
        let mut query_blind_out = Vec::with_capacity(num_aggregators as usize);
        query_blind_out.push(Prio3Blinds {
            query_rand_blind: query_leader_blind_opt,
        });
        for helper_blind in helper_query_rand_blinds.into_iter() {
            query_blind_out.push(Prio3Blinds {
                query_rand_blind: helper_blind.map(Seed::from_bytes),
            });
        }

        Ok((
            public_share,
            out_first,
            out_second,
            public_share_second,
            public_proof_run1,
            public_proof_run2,
            query_blind_out,
        ))
    }

    #[allow(clippy::type_complexity)]
    #[allow(dead_code)]
    pub(crate) fn shard_with_random_new_single_field<const N: usize>(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; N],
        random: &[u8],
    ) -> Result<
        (
            Prio3PublicShare<SEED_SIZE>,
            Vec<Prio3InputShare<T::Field, SEED_SIZE>>,
            Prio3PublicShare<SEED_SIZE>,
            Prio3PublicProof<T::Field, SEED_SIZE>,
            Vec<Prio3Blinds<SEED_SIZE>>,
        ),
        VdafError,
    > {
        if random.len() != self.random_size() {
            return Err(VdafError::Uncategorized(
                "incorrect random input length".to_string(),
            ));
        }
        let mut random_seeds = random.chunks_exact(SEED_SIZE);
        let num_aggregators = self.num_aggregators;
        let encoded_measurement = self.typ.encode_measurement(measurement)?;
        let enc_measurement = encoded_measurement.clone();

        // Generate the measurement shares and compute the joint randomness.
        let mut helper_shares = Vec::with_capacity(num_aggregators as usize - 1);
        let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        // println!("Is 3 move? {}", self.typ.joint_rand_len());
        let mut leader_measurement_share = encoded_measurement.clone();
        for agg_id in 1..num_aggregators {
            // The Option from the ChunksExact iterator is okay to unwrap because we checked that
            // the randomness slice is long enough for this VDAF. The slice-to-array conversion
            // Result is okay to unwrap because the ChunksExact iterator always returns slices of
            // the correct length.
            let measurement_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let proof_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let measurement_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &Seed(measurement_share_seed),
                &Self::domain_separation_tag(DST_MEASUREMENT_SHARE),
                &[agg_id],
            ));
            let joint_rand_blind =
                if let Some(helper_joint_rand_parts) = helper_joint_rand_parts.as_mut() {
                    let joint_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                    // 1st thing inside RO_i is the ith blind
                    let mut joint_rand_part_xof = P::init(
                        &joint_rand_blind,
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    // 2nd thing inside RO_i is the ith aggregator ID
                    joint_rand_part_xof.update(&[agg_id]); // Aggregator ID
                                                           // 3rd thing inside RO_i is the nonce
                    joint_rand_part_xof.update(nonce);

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                        y.encode(&mut encoding_buffer);
                        // 4th thing inside RO_i is the encoded ith measurement share
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Compute the RO_i output from the 4 things that were fed into it (see above)
                    helper_joint_rand_parts.push(joint_rand_part_xof.into_seed());

                    Some(joint_rand_blind)
                } else {
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                    }
                    None
                };
            let helper =
                HelperShare::from_seeds(measurement_share_seed, proof_share_seed, joint_rand_blind);
            helper_shares.push(helper);
        }

        let mut leader_blind_opt = None;
        let public_share = Prio3PublicShare {
            joint_rand_parts: helper_joint_rand_parts
                .as_ref()
                .map(|helper_joint_rand_parts| {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut joint_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    joint_rand_part_xof.update(&[0]); // Aggregator ID
                    joint_rand_part_xof.update(nonce);
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_measurement_share.iter() {
                        x.encode(&mut encoding_buffer);
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }
                    leader_blind_opt = Some(leader_blind);

                    let leader_joint_rand_seed_part = joint_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    // Put all the RO_i results (joint randomness shares) into a vector - will be used to derive the joint randomness
                    vec.push(leader_joint_rand_seed_part);
                    vec.extend(helper_joint_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Compute the joint randomness.
        let joint_rand: Vec<T::Field> = public_share
            .joint_rand_parts
            .as_ref()
            .map(|joint_rand_parts| {
                // Combine all RO_i results to the main RO result
                let joint_rand_seed = Self::derive_joint_rand_seed(joint_rand_parts.iter());
                P::seed_stream(
                    &joint_rand_seed,
                    &Self::domain_separation_tag(DST_JOINT_RANDOMNESS),
                    &[],
                )
                .into_field_vec(self.typ.joint_rand_len())
            })
            .unwrap_or_default();

        // Run the proof-generation algorithm.
        let prove_rand_seed = random_seeds.next().unwrap().try_into().unwrap();
        let prove_rand = P::seed_stream(
            &Seed::from_bytes(prove_rand_seed),
            &Self::domain_separation_tag(DST_PROVE_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len());
        let mut leader_proof_share =
            self.typ
                .prove(&encoded_measurement, &prove_rand, &joint_rand)?;

        let proof = leader_proof_share.clone();

        // Generate the proof shares and distribute the joint randomness seed hints.
        for (j, helper) in helper_shares.iter_mut().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            for (x, y) in leader_proof_share
                .iter_mut()
                .zip(proof_share_prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }
        }

        // Prep the output messages.
        let mut out = Vec::with_capacity(num_aggregators as usize);
        out.push(Prio3InputShare {
            measurement_share: Share::Leader(leader_measurement_share),
            proof_share: Share::Leader(leader_proof_share.clone()),
            joint_rand_blind: leader_blind_opt,
        });

        for helper in helper_shares.clone().into_iter() {
            out.push(Prio3InputShare {
                measurement_share: Share::Helper(helper.measurement_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_blind: helper.joint_rand_blind,
            });
        }

        // Deriving query_rand from Fiat-Shamir
        let mut helper_query_rand_parts = if self.typ.query_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        let mut helper_query_rand_blinds = Vec::with_capacity(num_aggregators as usize - 1);

        for (j, helper) in helper_shares.into_iter().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            let query_rand_blind =
                if let Some(helper_query_rand_parts) = helper_query_rand_parts.as_mut() {
                    let query_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                    // 1st thing inside RO_i is the ith blind
                    let mut query_rand_part_xof = P::init(
                        &query_rand_blind,
                        &Self::domain_separation_tag(DST_QUERY_RAND_PART),
                    );

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in joint_rand.iter() {
                        x.encode(&mut encoding_buffer);
                        // 2nd thing inside RO_i is the previous round joint randomness (captures transcript till previous round)
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (_x, y) in leader_proof_share
                        .iter()
                        .zip(proof_share_prng)
                        .take(self.typ.proof_len())
                    {
                        y.encode(&mut encoding_buffer);
                        // 3rd thing inside RO_i is the encoded ith proof share
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Compute the RO_i output from the 3 things that were fed into it (see above)
                    helper_query_rand_parts.push(query_rand_part_xof.into_seed());

                    Some(query_rand_blind)
                } else {
                    None
                };

            helper_query_rand_blinds.push(query_rand_blind);
        }

        let mut query_leader_blind_opt = None;
        let public_share_second = Prio3PublicShare {
            // Repurposing the joint_rand_parts field to store the query_rand_parts
            joint_rand_parts: helper_query_rand_parts
                .as_ref()
                .map(|helper_query_rand_parts| {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut query_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &Self::domain_separation_tag(DST_QUERY_RAND_PART),
                    );

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in joint_rand.iter() {
                        x.encode(&mut encoding_buffer);
                        // 2nd thing inside RO_i is the previous round joint randomness (captures transcript till previous round)
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_proof_share.iter() {
                        x.encode(&mut encoding_buffer);
                        query_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }
                    query_leader_blind_opt = Some(leader_blind);

                    let leader_query_rand_seed_part = query_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    // Put all the RO_i results (joint randomness shares) into a vector - will be used to derive the joint randomness
                    vec.push(leader_query_rand_seed_part);
                    vec.extend(helper_query_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Compute the query randomness.
        let query_rand: Vec<T::Field> = public_share_second
            .joint_rand_parts
            .as_ref()
            .map(|query_rand_parts| {
                // Combine all RO_i results to the main RO result
                let query_rand_seed = Self::derive_joint_rand_seed(query_rand_parts.iter());
                P::seed_stream(
                    &query_rand_seed,
                    &Self::domain_separation_tag(DST_QUERY_RANDOMNESS),
                    &[],
                )
                .into_field_vec(self.typ.query_rand_len())
            })
            .unwrap_or_default();

        // Run the query algorithm
        let public_proof = Prio3PublicProof {
            query_answers: self
                .typ
                .query(&enc_measurement, &proof, &query_rand, &joint_rand, 1)?,
        };

        // Prep the query blind messages.
        let mut query_blind_out = Vec::with_capacity(num_aggregators as usize);
        query_blind_out.push(Prio3Blinds {
            query_rand_blind: query_leader_blind_opt,
        });
        for helper_blind in helper_query_rand_blinds.into_iter() {
            query_blind_out.push(Prio3Blinds {
                query_rand_blind: helper_blind.map(Seed::from_bytes),
            });
        }

        Ok((
            public_share,
            out,
            public_share_second,
            public_proof,
            query_blind_out,
        ))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn shard_with_random<const N: usize>(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; N],
        random: &[u8],
    ) -> Result<
        (
            Prio3PublicShare<SEED_SIZE>,
            Vec<Prio3InputShare<T::Field, SEED_SIZE>>,
        ),
        VdafError,
    > {
        if random.len() != self.random_size() {
            return Err(VdafError::Uncategorized(
                "incorrect random input length".to_string(),
            ));
        }
        let mut random_seeds = random.chunks_exact(SEED_SIZE);
        let num_aggregators = self.num_aggregators;
        let encoded_measurement = self.typ.encode_measurement(measurement)?;

        // Generate the measurement shares and compute the joint randomness.
        let mut helper_shares = Vec::with_capacity(num_aggregators as usize - 1);
        let mut helper_joint_rand_parts = if self.typ.joint_rand_len() > 0 {
            Some(Vec::with_capacity(num_aggregators as usize - 1))
        } else {
            None
        };
        // println!("Is 3 move? {}", self.typ.joint_rand_len());
        let mut leader_measurement_share = encoded_measurement.clone();
        for agg_id in 1..num_aggregators {
            // The Option from the ChunksExact iterator is okay to unwrap because we checked that
            // the randomness slice is long enough for this VDAF. The slice-to-array conversion
            // Result is okay to unwrap because the ChunksExact iterator always returns slices of
            // the correct length.
            let measurement_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let proof_share_seed = random_seeds.next().unwrap().try_into().unwrap();
            let measurement_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &Seed(measurement_share_seed),
                &Self::domain_separation_tag(DST_MEASUREMENT_SHARE),
                &[agg_id],
            ));
            let joint_rand_blind =
                if let Some(helper_joint_rand_parts) = helper_joint_rand_parts.as_mut() {
                    let joint_rand_blind = random_seeds.next().unwrap().try_into().unwrap();
                    // 1st thing inside RO_i is the ith blind
                    let mut joint_rand_part_xof = P::init(
                        &joint_rand_blind,
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    // 2nd thing inside RO_i is the ith aggregator ID
                    joint_rand_part_xof.update(&[agg_id]); // Aggregator ID
                                                           // 3rd thing inside RO_i is the nonce
                    joint_rand_part_xof.update(nonce);

                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                        y.encode(&mut encoding_buffer);
                        // 4th thing inside RO_i is the encoded ith measurement share
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }

                    // Compute the RO_i output from the 4 things that were fed into it (see above)
                    helper_joint_rand_parts.push(joint_rand_part_xof.into_seed());

                    Some(joint_rand_blind)
                } else {
                    for (x, y) in leader_measurement_share
                        .iter_mut()
                        .zip(measurement_share_prng)
                    {
                        *x -= y;
                    }
                    None
                };
            let helper =
                HelperShare::from_seeds(measurement_share_seed, proof_share_seed, joint_rand_blind);
            helper_shares.push(helper);
        }

        let mut leader_blind_opt = None;
        let public_share = Prio3PublicShare {
            joint_rand_parts: helper_joint_rand_parts
                .as_ref()
                .map(|helper_joint_rand_parts| {
                    let leader_blind_bytes = random_seeds.next().unwrap().try_into().unwrap();
                    let leader_blind = Seed::from_bytes(leader_blind_bytes);

                    let mut joint_rand_part_xof = P::init(
                        leader_blind.as_ref(),
                        &Self::domain_separation_tag(DST_JOINT_RAND_PART),
                    );
                    joint_rand_part_xof.update(&[0]); // Aggregator ID
                    joint_rand_part_xof.update(nonce);
                    let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
                    for x in leader_measurement_share.iter() {
                        x.encode(&mut encoding_buffer);
                        joint_rand_part_xof.update(&encoding_buffer);
                        encoding_buffer.clear();
                    }
                    leader_blind_opt = Some(leader_blind);

                    let leader_joint_rand_seed_part = joint_rand_part_xof.into_seed();

                    let mut vec = Vec::with_capacity(self.num_aggregators());
                    // Put all the RO_i results (joint randomness shares) into a vector - will be used to derive the joint randomness
                    vec.push(leader_joint_rand_seed_part);
                    vec.extend(helper_joint_rand_parts.iter().cloned());
                    vec
                }),
        };

        // Compute the joint randomness.
        let joint_rand: Vec<T::Field> = public_share
            .joint_rand_parts
            .as_ref()
            .map(|joint_rand_parts| {
                // Combine all RO_i results to the main RO result
                let joint_rand_seed = Self::derive_joint_rand_seed(joint_rand_parts.iter());
                P::seed_stream(
                    &joint_rand_seed,
                    &Self::domain_separation_tag(DST_JOINT_RANDOMNESS),
                    &[],
                )
                .into_field_vec(self.typ.joint_rand_len())
            })
            .unwrap_or_default();

        // Run the proof-generation algorithm.
        let prove_rand_seed = random_seeds.next().unwrap().try_into().unwrap();
        let prove_rand = P::seed_stream(
            &Seed::from_bytes(prove_rand_seed),
            &Self::domain_separation_tag(DST_PROVE_RANDOMNESS),
            &[],
        )
        .into_field_vec(self.typ.prove_rand_len());
        let mut leader_proof_share =
            self.typ
                .prove(&encoded_measurement, &prove_rand, &joint_rand)?;

        // Generate the proof shares and distribute the joint randomness seed hints.
        for (j, helper) in helper_shares.iter_mut().enumerate() {
            let proof_share_prng: Prng<T::Field, _> = Prng::from_seed_stream(P::seed_stream(
                &helper.proof_share,
                &Self::domain_separation_tag(DST_PROOF_SHARE),
                &[j as u8 + 1],
            ));
            for (x, y) in leader_proof_share
                .iter_mut()
                .zip(proof_share_prng)
                .take(self.typ.proof_len())
            {
                *x -= y;
            }
        }

        // Prep the output messages.
        let mut out = Vec::with_capacity(num_aggregators as usize);
        out.push(Prio3InputShare {
            measurement_share: Share::Leader(leader_measurement_share),
            proof_share: Share::Leader(leader_proof_share),
            joint_rand_blind: leader_blind_opt,
        });

        for helper in helper_shares.into_iter() {
            out.push(Prio3InputShare {
                measurement_share: Share::Helper(helper.measurement_share),
                proof_share: Share::Helper(helper.proof_share),
                joint_rand_blind: helper.joint_rand_blind,
            });
        }

        Ok((public_share, out))
    }

    fn role_try_from(&self, agg_id: usize) -> Result<u8, VdafError> {
        if agg_id >= self.num_aggregators as usize {
            return Err(VdafError::Uncategorized("unexpected aggregator id".into()));
        }
        Ok(u8::try_from(agg_id).unwrap())
    }
}

impl<T, P, const SEED_SIZE: usize> Vdaf for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    const ID: u32 = T::ID;
    type Measurement = T::Measurement;
    type AggregateResult = T::AggregateResult;
    type AggregationParam = ();
    type PublicShare = Prio3PublicShare<SEED_SIZE>;
    type InputShare = Prio3InputShare<T::Field, SEED_SIZE>;
    type ProofShare = Prio3ProofShare<T::Field, SEED_SIZE>;
    type OutputShare = OutputShare<T::Field>;
    type AggregateShare = AggregateShare<T::Field>;
    type PublicProof = Prio3PublicProof<T::Field, SEED_SIZE>;
    type Blinds = Prio3Blinds<SEED_SIZE>;
    type PrepareState = Prio3PrepareState<T::Field, SEED_SIZE>;
    type BatchedOutputShare = Prio3BatchedOutputShare<T::Field>;

    fn num_aggregators(&self) -> usize {
        self.num_aggregators as usize
    }
}

/// Public proof broadcast by the [`Client`] to every [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
/// Message broadcast by each [`Aggregator`] in each round of the Preparation phase.
pub struct Prio3PublicProof<F, const SEED_SIZE: usize> {
    query_answers: Vec<F>,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3PublicProof<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3PublicProof<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3PublicProof<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.query_answers.ct_eq(&other.query_answers)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> Encode for Prio3PublicProof<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        for x in &self.query_answers {
            x.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        // Each element of the verifier has the same size.
        let len = F::ENCODED_SIZE * self.query_answers.len();
        Some(len)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> ParameterizedDecode<usize>
    for Prio3PublicProof<F, SEED_SIZE>
{
    // todo change decoding paramter to usize instead of &usize
    fn decode_with_param(
        decoding_parameter: &usize,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut query_answers = Vec::with_capacity(*decoding_parameter);
        for _ in 0..*decoding_parameter {
            query_answers.push(F::decode(bytes)?);
        }

        Ok(Prio3PublicProof { query_answers })
    }
}

/// Message broadcast by the [`Client`] to every [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
pub struct Prio3PublicShare<const SEED_SIZE: usize> {
    /// Contributions to the joint randomness from every aggregator's share.
    joint_rand_parts: Option<Vec<Seed<SEED_SIZE>>>,
}

impl<const SEED_SIZE: usize> Encode for Prio3PublicShare<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if let Some(joint_rand_parts) = self.joint_rand_parts.as_ref() {
            for part in joint_rand_parts.iter() {
                part.encode(bytes);
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        if let Some(joint_rand_parts) = self.joint_rand_parts.as_ref() {
            // Each seed has the same size.
            Some(SEED_SIZE * joint_rand_parts.len())
        } else {
            Some(0)
        }
    }
}

impl<const SEED_SIZE: usize> PartialEq for Prio3PublicShare<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Prio3PublicShare<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Prio3PublicShare<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_parts.
        option_ct_eq(
            self.joint_rand_parts.as_deref(),
            other.joint_rand_parts.as_deref(),
        )
    }
}

impl<T, P, const SEED_SIZE: usize> ParameterizedDecode<Prio3<T, P, SEED_SIZE>>
    for Prio3PublicShare<SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        decoding_parameter: &Prio3<T, P, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        if decoding_parameter.typ.joint_rand_len() > 0 {
            let joint_rand_parts = iter::repeat_with(|| Seed::<SEED_SIZE>::decode(bytes))
                .take(decoding_parameter.num_aggregators.into())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Self {
                joint_rand_parts: Some(joint_rand_parts),
            })
        } else {
            Ok(Self {
                joint_rand_parts: None,
            })
        }
    }
}

/// Message which holds blinds for query randomness for verifiers
#[derive(Clone, Debug)]
pub struct Prio3Blinds<const SEED_SIZE: usize> {
    /// Blinding seed used by the Aggregator to compute the query randomness. This field is optional
    /// because not every [`Type`] requires joint randomness.
    query_rand_blind: Option<Seed<SEED_SIZE>>,
}

impl<const SEED_SIZE: usize> PartialEq for Prio3Blinds<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Prio3Blinds<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Prio3Blinds<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_blind.
        option_ct_eq(
            self.query_rand_blind.as_ref(),
            other.query_rand_blind.as_ref(),
        )
    }
}

impl<const SEED_SIZE: usize> Encode for Prio3Blinds<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if let Some(ref blind) = self.query_rand_blind {
            blind.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = 0;
        if let Some(ref blind) = self.query_rand_blind {
            len += blind.encoded_len()?;
        }
        Some(len)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3Blinds<SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, _agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let query_rand_blind = if prio3.typ.query_rand_len() > 0 {
            let blind = Seed::decode(bytes)?;
            Some(blind)
        } else {
            None
        };

        Ok(Prio3Blinds { query_rand_blind })
    }
}

/// Message sent by the [`Client`] to each [`Aggregator`] during the Sharding phase.
#[derive(Clone, Debug)]
pub struct Prio3InputShare<F, const SEED_SIZE: usize> {
    /// The measurement share.
    measurement_share: Share<F, SEED_SIZE>,

    /// The proof share.
    proof_share: Share<F, SEED_SIZE>,

    /// Blinding seed used by the Aggregator to compute the joint randomness. This field is optional
    /// because not every [`Type`] requires joint randomness.
    joint_rand_blind: Option<Seed<SEED_SIZE>>,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3InputShare<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3InputShare<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3InputShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_blind.
        option_ct_eq(
            self.joint_rand_blind.as_ref(),
            other.joint_rand_blind.as_ref(),
        ) & self.measurement_share.ct_eq(&other.measurement_share)
            & self.proof_share.ct_eq(&other.proof_share)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> Encode for Prio3InputShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if matches!(
            (&self.measurement_share, &self.proof_share),
            (Share::Leader(_), Share::Helper(_)) | (Share::Helper(_), Share::Leader(_))
        ) {
            panic!("tried to encode input share with ambiguous encoding")
        }

        self.measurement_share.encode(bytes);
        self.proof_share.encode(bytes);
        if let Some(ref blind) = self.joint_rand_blind {
            blind.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = self.measurement_share.encoded_len()? + self.proof_share.encoded_len()?;
        if let Some(ref blind) = self.joint_rand_blind {
            len += blind.encoded_len()?;
        }
        Some(len)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3InputShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;
        let (input_decoder, proof_decoder) = if agg_id == 0 {
            (
                ShareDecodingParameter::Leader(prio3.typ.input_len()),
                ShareDecodingParameter::Leader(prio3.typ.proof_len()),
            )
        } else {
            (
                ShareDecodingParameter::Helper,
                ShareDecodingParameter::Helper,
            )
        };

        let measurement_share = Share::decode_with_param(&input_decoder, bytes)?;
        let proof_share = Share::decode_with_param(&proof_decoder, bytes)?;
        let joint_rand_blind = if prio3.typ.joint_rand_len() > 0 {
            let blind = Seed::decode(bytes)?;
            Some(blind)
        } else {
            None
        };

        Ok(Prio3InputShare {
            measurement_share,
            proof_share,
            joint_rand_blind,
        })
    }
}

/// Wrapper around a [`Share`] of a Prio3 proof
#[derive(Clone, Debug)]
pub struct Prio3ProofShare<F, const SEED_SIZE: usize> {
    /// The proof share.
    proof_share: Share<F, SEED_SIZE>,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3ProofShare<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3ProofShare<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3ProofShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.proof_share.ct_eq(&other.proof_share)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> Encode for Prio3ProofShare<F, SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        //if matches!(
        //    (&self.proof_share),
        //    (Share::Leader(_), Share::Helper(_)) | (Share::Helper(_), Share::Leader(_))
        //) {
        //    panic!("tried to encode input share with ambiguous encoding")
        //}

        //self.measurement_share.encode(bytes);
        self.proof_share.encode(bytes);
    }

    fn encoded_len(&self) -> Option<usize> {
        let len = self.proof_share.encoded_len()?;
        Some(len)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3ProofShare<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;
        let proof_decoder = if agg_id == 0 {
            ShareDecodingParameter::Leader(prio3.typ.proof_len())
        } else {
            ShareDecodingParameter::Helper
        };

        let proof_share = Share::decode_with_param(&proof_decoder, bytes)?;

        Ok(Prio3ProofShare { proof_share })
    }
}

/// Message sent by each [`Aggregator`] at the end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prio3BatchedOutputShare<F> {
    /// The proof share. Should be shares of 0 for a verifying proof
    pub output_share: F,
}

impl<F: ConstantTimeEq> PartialEq for Prio3BatchedOutputShare<F> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq> Eq for Prio3BatchedOutputShare<F> {}

impl<F: ConstantTimeEq> ConstantTimeEq for Prio3BatchedOutputShare<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_blind.
        self.output_share.ct_eq(&other.output_share)
    }
}

impl<F: FftFriendlyFieldElement> Encode for Prio3BatchedOutputShare<F> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.output_share.encode(bytes);
    }

    fn encoded_len(&self) -> Option<usize> {
        let len = self.output_share.encoded_len()?;
        Some(len)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize>
    ParameterizedDecode<Prio3PrepareState<F, SEED_SIZE>> for Prio3BatchedOutputShare<F>
{
    fn decode_with_param(
        _decoding_parameter: &Prio3PrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let output_share = F::decode(bytes)?;
        Ok(Prio3BatchedOutputShare { output_share })
    }
}

#[derive(Clone, Debug)]
/// Message broadcast by each [`Aggregator`] in each round of the Preparation phase.
pub struct Prio3PrepareShare<F, const SEED_SIZE: usize> {
    /// A share of the FLP verifier message. (See [`Type`].)
    verifier: Vec<F>,

    /// A part of the joint randomness seed.
    joint_rand_part: Option<Seed<SEED_SIZE>>,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3PrepareShare<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3PrepareShare<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareShare<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_part.
        option_ct_eq(
            self.joint_rand_part.as_ref(),
            other.joint_rand_part.as_ref(),
        ) & self.verifier.ct_eq(&other.verifier)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> Encode
    for Prio3PrepareShare<F, SEED_SIZE>
{
    fn encode(&self, bytes: &mut Vec<u8>) {
        for x in &self.verifier {
            x.encode(bytes);
        }
        if let Some(ref seed) = self.joint_rand_part {
            seed.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        // Each element of the verifier has the same size.
        let mut len = F::ENCODED_SIZE * self.verifier.len();
        if let Some(ref seed) = self.joint_rand_part {
            len += seed.encoded_len()?;
        }
        Some(len)
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize>
    ParameterizedDecode<Prio3PrepareState<F, SEED_SIZE>> for Prio3PrepareShare<F, SEED_SIZE>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let mut verifier = Vec::with_capacity(decoding_parameter.verifier_len);
        for _ in 0..decoding_parameter.verifier_len {
            verifier.push(F::decode(bytes)?);
        }

        let joint_rand_part = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareShare {
            verifier,
            joint_rand_part,
        })
    }
}

#[derive(Clone, Debug)]
/// Result of combining a round of [`Prio3PrepareShare`] messages.
pub struct Prio3PrepareMessage<const SEED_SIZE: usize> {
    /// The joint randomness seed computed by the Aggregators.
    joint_rand_seed: Option<Seed<SEED_SIZE>>,
}

impl<const SEED_SIZE: usize> PartialEq for Prio3PrepareMessage<SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<const SEED_SIZE: usize> Eq for Prio3PrepareMessage<SEED_SIZE> {}

impl<const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareMessage<SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presnce or absence of the joint_rand_seed.
        option_ct_eq(
            self.joint_rand_seed.as_ref(),
            other.joint_rand_seed.as_ref(),
        )
    }
}

impl<const SEED_SIZE: usize> Encode for Prio3PrepareMessage<SEED_SIZE> {
    fn encode(&self, bytes: &mut Vec<u8>) {
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encoded_len()
        } else {
            Some(0)
        }
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize>
    ParameterizedDecode<Prio3PrepareState<F, SEED_SIZE>> for Prio3PrepareMessage<SEED_SIZE>
{
    fn decode_with_param(
        decoding_parameter: &Prio3PrepareState<F, SEED_SIZE>,
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let joint_rand_seed = if decoding_parameter.joint_rand_seed.is_some() {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }
}

impl<T, P, const SEED_SIZE: usize> Client<16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    #[allow(clippy::type_complexity)]
    fn shard(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(Self::PublicShare, Vec<Prio3InputShare<T::Field, SEED_SIZE>>), VdafError> {
        let mut random = vec![0u8; self.random_size()];
        getrandom::getrandom(&mut random)?;
        self.shard_with_random(measurement, nonce, &random)
    }
}

impl<T, P, const SEED_SIZE: usize> BatchClient<16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    #[allow(clippy::type_complexity)]
    fn shard_batched(
        &self,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<
        (
            Prio3PublicShare<SEED_SIZE>,
            Vec<Prio3InputShare<T::Field, SEED_SIZE>>,
            Vec<Prio3ProofShare<T::Field, SEED_SIZE>>,
            Prio3PublicShare<SEED_SIZE>,
            Prio3PublicProof<T::Field, SEED_SIZE>,
            Prio3PublicProof<T::Field, SEED_SIZE>,
            Vec<Prio3Blinds<SEED_SIZE>>,
        ),
        VdafError,
    > {
        let mut random = vec![0u8; self.random_size_double_field()];
        getrandom::getrandom(&mut random)?;
        self.shard_with_random_new(measurement, nonce, &random)
    }
}
/// State of each [`Aggregator`] during the Preparation phase.
#[derive(Clone)]
pub struct Prio3PrepareState<F, const SEED_SIZE: usize> {
    measurement_share: Share<F, SEED_SIZE>,
    joint_rand_seed: Option<Seed<SEED_SIZE>>,
    agg_id: u8,
    verifier_len: usize,
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> PartialEq for Prio3PrepareState<F, SEED_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> Eq for Prio3PrepareState<F, SEED_SIZE> {}

impl<F: ConstantTimeEq, const SEED_SIZE: usize> ConstantTimeEq for Prio3PrepareState<F, SEED_SIZE> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // We allow short-circuiting on the presence or absence of the joint_rand_seed, as well as
        // the aggregator ID & verifier length parameters.
        if self.agg_id != other.agg_id || self.verifier_len != other.verifier_len {
            return Choice::from(0);
        }

        option_ct_eq(
            self.joint_rand_seed.as_ref(),
            other.joint_rand_seed.as_ref(),
        ) & self.measurement_share.ct_eq(&other.measurement_share)
    }
}

impl<F, const SEED_SIZE: usize> Debug for Prio3PrepareState<F, SEED_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prio3PrepareState")
            .field("measurement_share", &"[redacted]")
            .field(
                "joint_rand_seed",
                match self.joint_rand_seed {
                    Some(_) => &"Some([redacted])",
                    None => &"None",
                },
            )
            .field("agg_id", &self.agg_id)
            .field("verifier_len", &self.verifier_len)
            .finish()
    }
}

impl<F: FftFriendlyFieldElement, const SEED_SIZE: usize> Encode
    for Prio3PrepareState<F, SEED_SIZE>
{
    /// Append the encoded form of this object to the end of `bytes`, growing the vector as needed.
    fn encode(&self, bytes: &mut Vec<u8>) {
        self.measurement_share.encode(bytes);
        if let Some(ref seed) = self.joint_rand_seed {
            seed.encode(bytes);
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = self.measurement_share.encoded_len()?;
        if let Some(ref seed) = self.joint_rand_seed {
            len += seed.encoded_len()?;
        }
        Some(len)
    }
}

impl<'a, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, usize)>
    for Prio3PrepareState<T::Field, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (prio3, agg_id): &(&'a Prio3<T, P, SEED_SIZE>, usize),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        let agg_id = prio3
            .role_try_from(*agg_id)
            .map_err(|e| CodecError::Other(Box::new(e)))?;

        let share_decoder = if agg_id == 0 {
            ShareDecodingParameter::Leader(prio3.typ.input_len())
        } else {
            ShareDecodingParameter::Helper
        };
        let measurement_share = Share::decode_with_param(&share_decoder, bytes)?;

        let joint_rand_seed = if prio3.typ.joint_rand_len() > 0 {
            Some(Seed::decode(bytes)?)
        } else {
            None
        };

        Ok(Self {
            measurement_share,
            joint_rand_seed,
            agg_id,
            verifier_len: prio3.typ.verifier_len(),
        })
    }
}

impl<T, P, const SEED_SIZE: usize> Aggregator<SEED_SIZE, 16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    //type PrepareState = Prio3PrepareState<T::Field, SEED_SIZE>;
    type PrepareShare = Prio3PrepareShare<T::Field, SEED_SIZE>;
    type PrepareMessage = Prio3PrepareMessage<SEED_SIZE>;

    /// Begins the Prep process with the other aggregators. The result of this process is
    /// the aggregator's output share.
    #[allow(clippy::type_complexity)]
    fn prepare_init(
        &self,
        verify_key: &[u8; SEED_SIZE],
        agg_id: usize,
        _agg_param: &Self::AggregationParam,
        nonce: &[u8; 16],
        public_share: &Self::PublicShare,
        msg: &Prio3InputShare<T::Field, SEED_SIZE>,
    ) -> Result<
        (
            Prio3PrepareState<T::Field, SEED_SIZE>,
            Prio3PrepareShare<T::Field, SEED_SIZE>,
        ),
        VdafError,
    > {
        let agg_id = self.role_try_from(agg_id)?;
        let mut query_rand_xof = P::init(
            verify_key,
            &Self::domain_separation_tag(DST_QUERY_RANDOMNESS),
        );
        query_rand_xof.update(nonce);
        let query_rand = query_rand_xof
            .into_seed_stream()
            .into_field_vec(self.typ.query_rand_len());

        // Create a reference to the (expanded) measurement share.
        let expanded_measurement_share: Option<Vec<T::Field>> = match msg.measurement_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => Some(
                P::seed_stream(
                    seed,
                    &Self::domain_separation_tag(DST_MEASUREMENT_SHARE),
                    &[agg_id],
                )
                .into_field_vec(self.typ.input_len()),
            ),
        };
        let measurement_share = match msg.measurement_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_measurement_share.as_ref().unwrap(),
        };

        // Create a reference to the (expanded) proof share.
        let expanded_proof_share: Option<Vec<T::Field>> = match msg.proof_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => Some(
                P::seed_stream(
                    seed,
                    &Self::domain_separation_tag(DST_PROOF_SHARE),
                    &[agg_id],
                )
                .into_field_vec(self.typ.proof_len()),
            ),
        };
        let proof_share = match msg.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share.as_ref().unwrap(),
        };

        // Compute the joint randomness.
        let (joint_rand_seed, joint_rand_part, joint_rand) = if self.typ.joint_rand_len() > 0 {
            let mut joint_rand_part_xof = P::init(
                msg.joint_rand_blind.as_ref().unwrap().as_ref(),
                &Self::domain_separation_tag(DST_JOINT_RAND_PART),
            );
            joint_rand_part_xof.update(&[agg_id]);
            joint_rand_part_xof.update(nonce);
            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in measurement_share {
                x.encode(&mut encoding_buffer);
                joint_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }
            let own_joint_rand_part = joint_rand_part_xof.into_seed();

            // Make an iterator over the joint randomness parts, but use this aggregator's
            // contribution, computed from the input share, in lieu of the the corresponding part
            // from the public share.
            //
            // The locally computed part should match the part from the public share for honestly
            // generated reports. If they do not match, the joint randomness seed check during the
            // next round of preparation should fail.
            let corrected_joint_rand_parts = public_share
                .joint_rand_parts
                .iter()
                .flatten()
                .take(agg_id as usize)
                .chain(iter::once(&own_joint_rand_part))
                .chain(
                    public_share
                        .joint_rand_parts
                        .iter()
                        .flatten()
                        .skip(agg_id as usize + 1),
                );
            let joint_rand_seed = Self::derive_joint_rand_seed(corrected_joint_rand_parts);

            let joint_rand = P::seed_stream(
                &joint_rand_seed,
                &Self::domain_separation_tag(DST_JOINT_RANDOMNESS),
                &[],
            )
            .into_field_vec(self.typ.joint_rand_len());
            (Some(joint_rand_seed), Some(own_joint_rand_part), joint_rand)
        } else {
            (None, None, Vec::new())
        };

        // Run the query-generation algorithm.
        let verifier_share = self.typ.query(
            measurement_share,
            proof_share,
            &query_rand,
            &joint_rand,
            self.num_aggregators as usize,
        )?;

        Ok((
            Prio3PrepareState {
                measurement_share: msg.measurement_share.clone(),
                joint_rand_seed,
                agg_id,
                verifier_len: verifier_share.len(),
            },
            Prio3PrepareShare {
                verifier: verifier_share,
                joint_rand_part,
            },
        ))
    }

    fn prepare_shares_to_prepare_message<
        M: IntoIterator<Item = Prio3PrepareShare<T::Field, SEED_SIZE>>,
    >(
        &self,
        _: &Self::AggregationParam,
        inputs: M,
    ) -> Result<Prio3PrepareMessage<SEED_SIZE>, VdafError> {
        let mut verifier = vec![T::Field::zero(); self.typ.verifier_len()];
        let mut joint_rand_parts = Vec::with_capacity(self.num_aggregators());
        let mut count = 0;
        for share in inputs.into_iter() {
            count += 1;

            if share.verifier.len() != verifier.len() {
                return Err(VdafError::Uncategorized(format!(
                    "unexpected verifier share length: got {}; want {}",
                    share.verifier.len(),
                    verifier.len(),
                )));
            }

            if self.typ.joint_rand_len() > 0 {
                let joint_rand_seed_part = share.joint_rand_part.unwrap();
                joint_rand_parts.push(joint_rand_seed_part);
            }

            for (x, y) in verifier.iter_mut().zip(share.verifier) {
                *x += y;
            }
        }

        if count != self.num_aggregators {
            return Err(VdafError::Uncategorized(format!(
                "unexpected message count: got {}; want {}",
                count, self.num_aggregators,
            )));
        }

        // Check the proof verifier.
        match self.typ.decide(&verifier) {
            Ok(true) => (),
            Ok(false) => {
                return Err(VdafError::Uncategorized(
                    "proof verifier check failed".into(),
                ))
            }
            Err(err) => return Err(VdafError::from(err)),
        };

        let joint_rand_seed = if self.typ.joint_rand_len() > 0 {
            Some(Self::derive_joint_rand_seed(joint_rand_parts.iter()))
        } else {
            None
        };

        Ok(Prio3PrepareMessage { joint_rand_seed })
    }

    fn prepare_next(
        &self,
        step: Prio3PrepareState<T::Field, SEED_SIZE>,
        msg: Prio3PrepareMessage<SEED_SIZE>,
    ) -> Result<PrepareTransition<Self, SEED_SIZE, 16>, VdafError> {
        if self.typ.joint_rand_len() > 0 {
            // Check that the joint randomness was correct.
            if step
                .joint_rand_seed
                .as_ref()
                .unwrap()
                .ct_ne(msg.joint_rand_seed.as_ref().unwrap())
                .into()
            {
                return Err(VdafError::Uncategorized(
                    "joint randomness mismatch".to_string(),
                ));
            }
        }

        // Compute the output share.
        let measurement_share = match step.measurement_share {
            Share::Leader(data) => data,
            Share::Helper(seed) => {
                let dst = Self::domain_separation_tag(DST_MEASUREMENT_SHARE);
                P::seed_stream(&seed, &dst, &[step.agg_id]).into_field_vec(self.typ.input_len())
            }
        };

        let output_share = match self.typ.truncate(measurement_share) {
            Ok(data) => OutputShare(data),
            Err(err) => {
                return Err(VdafError::from(err));
            }
        };

        Ok(PrepareTransition::Finish(output_share))
    }

    /// Aggregates a sequence of output shares into an aggregate share.
    fn aggregate<It: IntoIterator<Item = OutputShare<T::Field>>>(
        &self,
        _agg_param: &(),
        output_shares: It,
    ) -> Result<AggregateShare<T::Field>, VdafError> {
        let mut agg_share = AggregateShare(vec![T::Field::zero(); self.typ.output_len()]);
        for output_share in output_shares.into_iter() {
            agg_share.accumulate(&output_share)?;
        }

        Ok(agg_share)
    }
}

impl<T, P, const SEED_SIZE: usize> BatchAggregator<SEED_SIZE, 16> for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// This is with batched verification support - degree-1 verification.
    /// Verifying proofs
    fn prepare_batched(
        &self,
        verify_key: &[u8; SEED_SIZE],
        vdaf_batch_key: &VdafBatchedKey<Self>,
        _agg_param: &Self::AggregationParam,
    ) -> Result<(Prio3BatchedOutputShare<T::Field>, OutputShare<T::Field>), VdafError> {
        //assert_eq!(SEED_SIZE, 32); // This version of the function uses a tuple of two 128-bit field elements to get the same soundness as a single 256-bit field would achieve. This requires assuming that each RO call returns two 128-bit field elements - so seed size must be 256.
        let agg_id = self.role_try_from(vdaf_batch_key.agg_id.into())?;

        // We assume for now that msg_r1 and msg_r2 carry the same measurement. Discard msg_r2's measurement; just a placeholder. The one in msg_r1 is used; they both are the same anyways for an honest prover.
        let msg_r1 = &vdaf_batch_key.input_share_0;
        let msg_r2 = &vdaf_batch_key.input_share_1;

        // Create a reference to the (expanded) measurement share.
        let expanded_measurement_share: Option<Vec<T::Field>> = match msg_r1.measurement_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => Some(
                P::seed_stream(
                    seed,
                    &Self::domain_separation_tag(DST_MEASUREMENT_SHARE),
                    &[agg_id],
                )
                .into_field_vec(self.typ.input_len()),
            ),
        };
        let measurement_share = match msg_r1.measurement_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_measurement_share.as_ref().unwrap(),
        };

        let output_share = match self.typ.truncate(measurement_share.clone()) {
            Ok(data) => OutputShare(data),
            Err(err) => {
                return Err(VdafError::from(err));
            }
        };

        // Create a reference to the (expanded) proof share.
        // Run 1
        let expanded_proof_share_r1: Option<Vec<T::Field>> = match msg_r1.proof_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => Some(
                P::seed_stream(
                    seed,
                    &Self::domain_separation_tag(DST_PROOF_SHARE),
                    &[agg_id],
                )
                .into_field_vec(self.typ.proof_len()),
            ),
        };
        let proof_share_r1 = match msg_r1.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share_r1.as_ref().unwrap(),
        };

        // Run 2
        let expanded_proof_share_r2: Option<Vec<T::Field>> = match msg_r2.proof_share {
            Share::Leader(_) => None,
            Share::Helper(ref seed) => Some(
                P::seed_stream(
                    seed,
                    &Self::domain_separation_tag(DST_PROOF_SHARE),
                    &[agg_id],
                )
                .into_field_vec(self.typ.proof_len()),
            ),
        };
        let proof_share_r2 = match msg_r2.proof_share {
            Share::Leader(ref data) => data,
            Share::Helper(_) => expanded_proof_share_r2.as_ref().unwrap(),
        };

        let public_share = &vdaf_batch_key.public_share;
        // Compute the joint randomness.
        // Discard msg_r2's joint randomness; just a placeholder. The one in msg_r1 is used; they both are the same anyways for an honest prover.
        let (_joint_rand_seed, joint_rand_part, joint_rand_fused) = if self.typ.joint_rand_len() > 0
        {
            let mut joint_rand_part_xof = P::init(
                msg_r1.joint_rand_blind.as_ref().unwrap().as_ref(),
                &Self::domain_separation_tag(DST_JOINT_RAND_PART),
            );
            joint_rand_part_xof.update(&[agg_id]);
            joint_rand_part_xof.update(&vdaf_batch_key.nonce);
            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in measurement_share {
                x.encode(&mut encoding_buffer);
                joint_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }
            let own_joint_rand_part = joint_rand_part_xof.into_seed();

            // Make an iterator over the joint randomness parts, but use this aggregator's
            // contribution, computed from the input share, in lieu of the the corresponding part
            // from the public share.
            //
            // The locally computed part should match the part from the public share for honestly
            // generated reports. If they do not match, the joint randomness seed check during the
            // next round of preparation should fail. [old comments]
            let corrected_joint_rand_parts = public_share
                .joint_rand_parts
                .iter()
                .flatten()
                .take(agg_id as usize)
                .chain(iter::once(&own_joint_rand_part))
                .chain(
                    public_share
                        .joint_rand_parts
                        .iter()
                        .flatten()
                        .skip(agg_id as usize + 1),
                );

            let joint_rand_seed = Self::derive_joint_rand_seed(corrected_joint_rand_parts);

            let joint_rand: Vec<T::Field> = P::seed_stream(
                &joint_rand_seed,
                &Self::domain_separation_tag(DST_JOINT_RANDOMNESS),
                &[],
            )
            .into_field_vec(2 * self.typ.joint_rand_len());
            (Some(joint_rand_seed), Some(own_joint_rand_part), joint_rand)
        } else {
            (None, None, Vec::new())
        };

        let joint_rand_split;
        if self.typ.joint_rand_len() > 0 {
            joint_rand_split = joint_rand_fused
                .chunks_exact(self.typ.joint_rand_len())
                .map(|x| x.to_vec())
                .collect::<Vec<_>>();

            assert_eq!(joint_rand_split.len(), 2);
        } else {
            // Placeholder
            joint_rand_split = vec![joint_rand_fused.clone(); 2];
        }

        let public_share_second = &vdaf_batch_key.public_share_second;
        let query_rand_blinds = &vdaf_batch_key.query_rand_blinds;

        // Compute the query randomness
        let (_query_rand_seed, query_rand_part, query_rand_fused) = if self.typ.query_rand_len() > 0
        {
            let mut query_rand_part_xof = P::init(
                query_rand_blinds
                    .query_rand_blind
                    .as_ref()
                    .unwrap()
                    .as_ref(),
                &Self::domain_separation_tag(DST_QUERY_RAND_PART),
            );
            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in joint_rand_fused.iter() {
                x.encode(&mut encoding_buffer);
                // 2nd thing inside RO_i is the previous round joint randomness (captures transcript till previous round)
                query_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }

            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in proof_share_r1.iter() {
                x.encode(&mut encoding_buffer);
                // 3rd thing inside RO_i is the encoded ith proof share of first run
                query_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }

            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in proof_share_r2.iter() {
                x.encode(&mut encoding_buffer);
                // 4th thing inside RO_i is the encoded ith proof share of second run
                query_rand_part_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }

            let own_query_rand_part = query_rand_part_xof.into_seed();

            // Make an iterator over the query randomness parts, but use this aggregator's
            // contribution, computed from the input share and proof share, in lieu of the the corresponding part
            // from the public share (second).
            //
            // The locally computed part should match the part from the public share for honestly
            // generated reports. If they do not match, the query randomness seed check during the
            // next round of preparation should fail.
            let corrected_query_rand_parts = public_share_second
                .joint_rand_parts
                .iter()
                .flatten()
                .take(agg_id as usize)
                .chain(iter::once(&own_query_rand_part))
                .chain(
                    public_share_second
                        .joint_rand_parts
                        .iter()
                        .flatten()
                        .skip(agg_id as usize + 1),
                );

            let query_rand_seed = Self::derive_joint_rand_seed(corrected_query_rand_parts);

            let query_rand = P::seed_stream(
                &query_rand_seed,
                &Self::domain_separation_tag(DST_QUERY_RANDOMNESS),
                &[],
            )
            .into_field_vec(2 * self.typ.query_rand_len());
            (Some(query_rand_seed), Some(own_query_rand_part), query_rand)
        } else {
            (None, None, Vec::new())
        };

        let query_rand_split = query_rand_fused
            .chunks_exact(self.typ.query_rand_len())
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        assert_eq!(query_rand_split.len(), 2);

        // Run the query-generation algorithm.
        // First run
        let verifier_share_r1 = self.typ.query(
            measurement_share,
            proof_share_r1,
            &query_rand_split[0],
            &joint_rand_split[0],
            self.num_aggregators as usize,
        )?;
        // Second run
        let verifier_share_r2 = self.typ.query(
            measurement_share,
            proof_share_r2,
            &query_rand_split[1],
            &joint_rand_split[1],
            self.num_aggregators as usize,
        )?;

        let mut check_count = 0;

        let public_proof_r1 = &vdaf_batch_key.public_proof_0;
        let public_proof_r2 = &vdaf_batch_key.public_proof_1;
        // Compute the difference "diff" between query result shares and public proof
        // Leader subtracts the public proof from the query result share
        // Helpers directly use their query result share
        // Run 1
        check_count += verifier_share_r1.len();
        assert_eq!(public_proof_r1.query_answers.len(), verifier_share_r1.len());
        let _public_proof_diff_share = match msg_r1.measurement_share {
            Share::Leader(_) => {
                let diff = public_proof_r1
                    .query_answers
                    .iter()
                    .zip(verifier_share_r1.iter())
                    .map(|(x, y)| *y - *x)
                    .collect::<Vec<T::Field>>();
                diff
            }
            Share::Helper(_) => verifier_share_r1.clone(),
        };
        // Run 2
        check_count += verifier_share_r2.len();
        assert_eq!(public_proof_r2.query_answers.len(), verifier_share_r2.len());
        let public_proof_diff_share = match msg_r1.measurement_share {
            Share::Leader(_) => {
                let diff = public_proof_r2
                    .query_answers
                    .iter()
                    .zip(verifier_share_r2.iter())
                    .map(|(x, y)| *y - *x)
                    .collect::<Vec<T::Field>>();
                diff
            }
            Share::Helper(_) => verifier_share_r2.clone(),
        };

        // Call Decide using public proof
        /*
        check_count += self.num_aggregators as usize - 1;
        // Check the proof verifier.
        let decide_zero_shares = match self.typ.decide(&public_proof.query_answers) {
            Ok(true) => match msg.measurement_share {
                Share::Leader(_) => {
                    vec![T::Field::zero() - T::Field::one(); self.num_aggregators as usize - 1]
                }
                Share::Helper(_) => {
                    let mut v = vec![T::Field::zero(); self.num_aggregators as usize - 1];
                    v[agg_id as usize - 1] = T::Field::one();
                    v
                }
            },
            Ok(false) => vec![T::Field::one() + T::Field::one(); self.num_aggregators as usize - 1], // Put 2 whenever error occurs and RLC will catch this with overwhelming probability
            Err(err) => vec![T::Field::one() + T::Field::one(); self.num_aggregators as usize - 1], // same as above case
        };
        */
        check_count += 2;
        // Check the proof verifier.
        let mut decide_zero_shares = [T::Field::zero(); 2];
        // Run 1
        decide_zero_shares[0] = match self.typ.decide(&public_proof_r1.query_answers) {
            Ok(true) => {
                //println!("Decide accepts on verifier");
                T::Field::zero()
            } // whene everyone accepts, we get shares of zero, otherwise, we always get shares of something non-zero here
            Ok(false) => {
                //println!("Decide REJECTS on verifier");
                T::Field::one()
            } // Put 2 whenever error occurs and RLC will catch this with overwhelming probability
            Err(_) => T::Field::one(), // same as above case
        };
        // Run 2
        decide_zero_shares[1] = match self.typ.decide(&public_proof_r2.query_answers) {
            Ok(true) => {
                //println!("Decide accepts on verifier");
                T::Field::zero()
            } // whene everyone accepts, we get shares of zero, otherwise, we always get shares of something non-zero here
            Ok(false) => {
                //println!("Decide REJECTS on verifier");
                T::Field::one()
            } // Put 2 whenever error occurs and RLC will catch this with overwhelming probability
            Err(_) => T::Field::one(), // same as above case
        };

        // Check that own_joint_rand and own_query_rand are correct based on what the prover said
        check_count += 2;
        let mut rand_diffs = [T::Field::zero(); 2];
        if self.typ.joint_rand_len() > 0
            && public_share.joint_rand_parts.as_ref().unwrap()[agg_id as usize]
                .ct_ne(&joint_rand_part.unwrap())
                .into()
        {
            //println!("Joint rand part is not correct");
            rand_diffs[0] = T::Field::one();
        }
        if public_share_second.joint_rand_parts.as_ref().unwrap()[agg_id as usize]
            .ct_ne(&query_rand_part.unwrap())
            .into()
        {
            //println!("Query rand part is not correct");
            rand_diffs[1] = T::Field::one();
        }

        // Make shares of zero from public proof r1, public proof r2, public_share, public_share_second hash/xof
        let mut hash_xof = P::init(&[0; SEED_SIZE], &Self::domain_separation_tag(DST_HASH_PART));
        if self.typ.joint_rand_len() > 0 {
            let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
            for x in public_share.joint_rand_parts.as_ref().unwrap() {
                x.encode(&mut encoding_buffer);
                hash_xof.update(&encoding_buffer);
                encoding_buffer.clear();
            }
        }
        let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
        for x in public_share_second.joint_rand_parts.as_ref().unwrap() {
            x.encode(&mut encoding_buffer);
            hash_xof.update(&encoding_buffer);
            encoding_buffer.clear();
        }
        let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
        for x in public_proof_r1.query_answers.iter() {
            x.encode(&mut encoding_buffer);
            hash_xof.update(&encoding_buffer);
            encoding_buffer.clear();
        }
        let mut encoding_buffer = Vec::with_capacity(T::Field::ENCODED_SIZE);
        for x in public_proof_r2.query_answers.iter() {
            x.encode(&mut encoding_buffer);
            hash_xof.update(&encoding_buffer);
            encoding_buffer.clear();
        }
        let hash: Vec<T::Field> = P::seed_stream(
            &hash_xof.into_seed(),
            &Self::domain_separation_tag(DST_HASH_PART),
            &[],
        )
        .into_field_vec(1);

        // Now create shares of zeros from these hashes
        check_count += self.num_aggregators as usize - 1;
        let hash_zero_shares = match msg_r1.measurement_share {
            Share::Leader(_) => {
                //println!("Hash of public parts: {:?}", hash[0]);
                vec![T::Field::zero() - hash[0]; self.num_aggregators as usize - 1]
            }
            Share::Helper(_) => {
                //println!("Hash of public parts: {:?}", hash[0]);
                let mut v = vec![T::Field::zero(); self.num_aggregators as usize - 1];
                v[agg_id as usize - 1] = hash[0];
                v
            }
        };

        let nonce = &vdaf_batch_key.nonce;
        // Use verifier_key to derive check_count random linear coefficients (RLC) for the shares of zero
        let mut rlc_rand_xof =
            P::init(verify_key, &Self::domain_separation_tag(DST_RLC_RANDOMNESS));
        rlc_rand_xof.update(nonce);
        let rlc_rand: Vec<T::Field> = rlc_rand_xof.into_seed_stream().into_field_vec(check_count);

        let mut proof_tag = T::Field::zero();
        public_proof_diff_share
            .iter()
            .chain(decide_zero_shares.iter())
            .chain(rand_diffs.iter())
            .chain(hash_zero_shares.iter())
            .zip(rlc_rand.iter())
            .for_each(|(x, y)| proof_tag += *x * *y);

        Ok((
            Prio3BatchedOutputShare {
                output_share: proof_tag,
            },
            output_share,
        ))
    }
}

#[cfg(feature = "experimental")]
impl<T, P, S, const SEED_SIZE: usize> AggregatorWithNoise<SEED_SIZE, 16, S>
    for Prio3<T, P, SEED_SIZE>
where
    T: TypeWithNoise<S>,
    P: Xof<SEED_SIZE>,
    S: DifferentialPrivacyStrategy,
{
    fn add_noise_to_agg_share(
        &self,
        dp_strategy: &S,
        _agg_param: &Self::AggregationParam,
        agg_share: &mut Self::AggregateShare,
        num_measurements: usize,
    ) -> Result<(), VdafError> {
        self.typ
            .add_noise_to_result(dp_strategy, &mut agg_share.0, num_measurements)?;
        Ok(())
    }
}

impl<T, P, const SEED_SIZE: usize> Collector for Prio3<T, P, SEED_SIZE>
where
    T: Type,
    P: Xof<SEED_SIZE>,
{
    /// Combines aggregate shares into the aggregate result.
    fn unshard<It: IntoIterator<Item = AggregateShare<T::Field>>>(
        &self,
        _agg_param: &Self::AggregationParam,
        agg_shares: It,
        num_measurements: usize,
    ) -> Result<T::AggregateResult, VdafError> {
        let mut agg = AggregateShare(vec![T::Field::zero(); self.typ.output_len()]);
        for agg_share in agg_shares.into_iter() {
            agg.merge(&agg_share)?;
        }

        Ok(self.typ.decode_result(&agg.0, num_measurements)?)
    }
}

#[derive(Clone)]
struct HelperShare<const SEED_SIZE: usize> {
    measurement_share: Seed<SEED_SIZE>,
    proof_share: Seed<SEED_SIZE>,
    joint_rand_blind: Option<Seed<SEED_SIZE>>,
}

impl<const SEED_SIZE: usize> HelperShare<SEED_SIZE> {
    fn from_seeds(
        measurement_share: [u8; SEED_SIZE],
        proof_share: [u8; SEED_SIZE],
        joint_rand_blind: Option<[u8; SEED_SIZE]>,
    ) -> Self {
        HelperShare {
            measurement_share: Seed::from_bytes(measurement_share),
            proof_share: Seed::from_bytes(proof_share),
            joint_rand_blind: joint_rand_blind.map(Seed::from_bytes),
        }
    }
}

fn check_num_aggregators(num_aggregators: u8) -> Result<(), VdafError> {
    if num_aggregators == 0 {
        return Err(VdafError::Uncategorized(format!(
            "at least one aggregator is required; got {num_aggregators}"
        )));
    } else if num_aggregators > 254 {
        return Err(VdafError::Uncategorized(format!(
            "number of aggregators must not exceed 254; got {num_aggregators}"
        )));
    }

    Ok(())
}

impl<'a, F, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, &'a ())>
    for OutputShare<F>
where
    F: FieldElement,
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, SEED_SIZE>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

impl<'a, F, T, P, const SEED_SIZE: usize> ParameterizedDecode<(&'a Prio3<T, P, SEED_SIZE>, &'a ())>
    for AggregateShare<F>
where
    F: FieldElement,
    T: Type,
    P: Xof<SEED_SIZE>,
{
    fn decode_with_param(
        (vdaf, _): &(&'a Prio3<T, P, SEED_SIZE>, &'a ()),
        bytes: &mut Cursor<&[u8]>,
    ) -> Result<Self, CodecError> {
        decode_fieldvec(vdaf.output_len(), bytes).map(Self)
    }
}

// This function determines equality between two optional, constant-time comparable values. It
// short-circuits on the existence (but not contents) of the values -- a timing side-channel may
// reveal whether the values match on Some or None.
#[inline]
fn option_ct_eq<T>(left: Option<&T>, right: Option<&T>) -> Choice
where
    T: ConstantTimeEq + ?Sized,
{
    match (left, right) {
        (Some(left), Some(right)) => left.ct_eq(right),
        (None, None) => Choice::from(1),
        _ => Choice::from(0),
    }
}

/// This is a polyfill for `usize::ilog2()`, which is only available in Rust 1.67 and later. It is
/// based on the implementation in the standard library. It can be removed when the MSRV has been
/// advanced past 1.67.
///
/// # Panics
///
/// This function will panic if `input` is zero.
fn ilog2(input: usize) -> u32 {
    if input == 0 {
        panic!("Tried to take the logarithm of zero");
    }
    (usize::BITS - 1) - input.leading_zeros()
}

/// Finds the optimal choice of chunk length for [`Prio3Histogram`] or [`Prio3SumVec`], given its
/// encoded measurement length. For [`Prio3Histogram`], the measurement length is equal to the
/// length parameter. For [`Prio3SumVec`], the measurement length is equal to the product of the
/// length and bits parameters.
pub fn optimal_chunk_length(measurement_length: usize) -> usize {
    if measurement_length <= 1 {
        return 1;
    }

    /// Candidate set of parameter choices for the parallel sum optimization.
    struct Candidate {
        gadget_calls: usize,
        chunk_length: usize,
    }

    let max_log2 = ilog2(measurement_length + 1);
    let best_opt = (1..=max_log2)
        .rev()
        .map(|log2| {
            let gadget_calls = (1 << log2) - 1;
            let chunk_length = (measurement_length + gadget_calls - 1) / gadget_calls;
            Candidate {
                gadget_calls,
                chunk_length,
            }
        })
        .min_by_key(|candidate| {
            // Compute the proof length, in field elements, for either Prio3Histogram or Prio3SumVec
            (candidate.chunk_length * 2)
                + 2 * ((1 + candidate.gadget_calls).next_power_of_two() - 1)
        });
    // Unwrap safety: max_log2 must be at least 1, because smaller measurement_length inputs are
    // dealt with separately. Thus, the range iterator that the search is over will be nonempty,
    // and min_by_key() will always return Some.
    best_opt.unwrap().chunk_length
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdaf::{
        equality_comparison_test, fieldvec_roundtrip_test, run_vdaf, run_vdaf_batched,
        run_vdaf_prepare,
    };
    use assert_matches::assert_matches;
    use rand::prelude::*;

    #[test]
    fn test_prio3_count_batched() {
        let prio3: Prio3<Count<Field128>, XofShake256, 32> = Prio3::new_count_256(2).unwrap();

        //println!("This is 2 move");
        run_vdaf_batched::<_, _, XofShake256, Count<Field128>, 32>(&prio3, &(), [1, 0, 0, 1, 1])
            .unwrap();
        //run_vdaf_batched::<_, _, XofShake128, Count<Field64>, 16>(&prio3, &(), [1]).unwrap();
        //assert_eq!(run_vdaf(&prio3, &(), [1, 0, 0, 1, 1]).unwrap(), 3);
    }

    #[test]
    fn test_prio3_sum_batched() {
        let prio3: Prio3<Sum<Field128>, XofShake256, 32> = Prio3::new_sum_256(3, 16).unwrap();

        //println!("This is 3 move");
        run_vdaf_batched::<_, _, XofShake256, Sum<Field128>, 32>(
            &prio3,
            &(),
            [0, (1 << 16) - 1, 0, 1, 1],
        )
        .unwrap();
        //assert_eq!(
        //    run_vdaf(&prio3, &(), [0, (1 << 16) - 1, 0, 1, 1]).unwrap(),
        //    (1 << 16) + 1
        //);
    }

    #[test]
    fn test_prio3_count() {
        let prio3 = Prio3::new_count(2).unwrap();

        assert_eq!(run_vdaf(&prio3, &(), [1, 0, 0, 1, 1]).unwrap(), 3);

        let mut nonce = [0; 16];
        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        thread_rng().fill(&mut nonce[..]);

        let (public_share, input_shares) = prio3.shard(&0, &nonce).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares).unwrap();

        let (public_share, input_shares) = prio3.shard(&1, &nonce).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares).unwrap();

        test_serialization(&prio3, &1, &nonce).unwrap();

        let prio3_extra_helper = Prio3::new_count(3).unwrap();
        assert_eq!(
            run_vdaf(&prio3_extra_helper, &(), [1, 0, 0, 1, 1]).unwrap(),
            3,
        );
    }

    #[test]
    fn test_prio3_sum() {
        let prio3 = Prio3::new_sum(3, 16).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, (1 << 16) - 1, 0, 1, 1]).unwrap(),
            (1 << 16) + 1
        );

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = [0; 16];

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        input_shares[0].joint_rand_blind.as_mut().unwrap().0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        assert_matches!(input_shares[0].measurement_share, Share::Leader(ref mut data) => {
            data[0] += Field64::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let (public_share, mut input_shares) = prio3.shard(&1, &nonce).unwrap();
        assert_matches!(input_shares[0].proof_share, Share::Leader(ref mut data) => {
                data[0] += Field64::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), &nonce, public_share, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        test_serialization(&prio3, &1, &nonce).unwrap();
    }

    #[test]
    fn test_prio3_sum_vec() {
        let prio3 = Prio3::new_sum_vec(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    fn test_prio3_sum_vec_128() {
        let prio3 = Prio3::new_sum_vec_128(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    fn test_prio3_sum_vec_256() {
        let prio3 = Prio3::new_sum_vec_256(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_prio3_sum_vec_multithreaded() {
        let prio3 = Prio3::new_sum_vec_multithreaded(2, 2, 20, 4).unwrap();
        assert_eq!(
            run_vdaf(
                &prio3,
                &(),
                [
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    vec![0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0],
                    vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                ]
            )
            .unwrap(),
            vec![1, 3, 1, 0, 3, 1, 0, 1, 2, 2, 3, 3, 1, 5, 1, 2, 1, 3, 0, 2],
        );
    }

    #[test]
    fn test_prio3_histogram_batched() {
        let prio3 = Prio3::new_histogram_256(2, 4, 2).unwrap();

        run_vdaf_batched::<_, _, XofShake128, Count<Field128>, 16>(&prio3, &(), [0, 1, 2, 3])
            .unwrap();
    }

    #[test]
    fn test_prio3_histogram() {
        let prio3 = Prio3::new_histogram(2, 4, 2).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [1, 1, 2, 3]).unwrap(),
            vec![0, 2, 1, 1]
        );
        assert_eq!(run_vdaf(&prio3, &(), [0]).unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [1]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [2]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [3]).unwrap(), vec![0, 0, 0, 1]);
        test_serialization(&prio3, &3, &[0; 16]).unwrap();
    }

    #[test]
    #[cfg(feature = "multithreaded")]
    fn test_prio3_histogram_multithreaded() {
        let prio3 = Prio3::new_histogram_multithreaded(2, 4, 2).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, 1, 2, 3]).unwrap(),
            vec![1, 1, 1, 1]
        );
        assert_eq!(run_vdaf(&prio3, &(), [0]).unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [1]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [2]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [3]).unwrap(), vec![0, 0, 0, 1]);
        test_serialization(&prio3, &3, &[0; 16]).unwrap();
    }

    #[test]
    fn test_prio3_average_batched() {
        let prio3 = Prio3::new_average_256(2, 64).unwrap();

        run_vdaf_batched::<_, _, XofShake128, Count<Field128>, 16>(
            &prio3,
            &(),
            [1, 11, 111, 1111, 3, 8],
        )
        .unwrap();
    }

    #[test]
    fn test_prio3_average() {
        let prio3 = Prio3::new_average(2, 63).unwrap();

        assert_eq!(run_vdaf(&prio3, &(), [17, 8]).unwrap(), 12.5f64);
        assert_eq!(run_vdaf(&prio3, &(), [1, 1, 1, 1]).unwrap(), 1f64);
        assert_eq!(run_vdaf(&prio3, &(), [0, 0, 0, 1]).unwrap(), 0.25f64);
        assert_eq!(
            run_vdaf(&prio3, &(), [1, 11, 111, 1111, 3, 8]).unwrap(),
            207.5f64
        );
    }

    #[test]
    fn test_prio3_input_share() {
        let prio3 = Prio3::new_sum(5, 16).unwrap();
        let (_public_share, input_shares) = prio3.shard(&1, &[0; 16]).unwrap();

        // Check that seed shares are distinct.
        for (i, x) in input_shares.iter().enumerate() {
            for (j, y) in input_shares.iter().enumerate() {
                if i != j {
                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.measurement_share, &y.measurement_share)
                    {
                        assert_ne!(left, right);
                    }

                    if let (Share::Helper(left), Share::Helper(right)) =
                        (&x.proof_share, &y.proof_share)
                    {
                        assert_ne!(left, right);
                    }

                    assert_ne!(x.joint_rand_blind, y.joint_rand_blind);
                }
            }
        }
    }

    fn test_serialization<T, P, const SEED_SIZE: usize>(
        prio3: &Prio3<T, P, SEED_SIZE>,
        measurement: &T::Measurement,
        nonce: &[u8; 16],
    ) -> Result<(), VdafError>
    where
        T: Type,
        P: Xof<SEED_SIZE>,
    {
        let mut verify_key = [0; SEED_SIZE];
        thread_rng().fill(&mut verify_key[..]);
        let (public_share, input_shares) = prio3.shard(measurement, nonce)?;

        let encoded_public_share = public_share.get_encoded();
        let decoded_public_share =
            Prio3PublicShare::get_decoded_with_param(prio3, &encoded_public_share)
                .expect("failed to decode public share");
        assert_eq!(decoded_public_share, public_share);
        assert_eq!(
            public_share.encoded_len().unwrap(),
            encoded_public_share.len()
        );

        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let encoded_input_share = input_share.get_encoded();
            let decoded_input_share =
                Prio3InputShare::get_decoded_with_param(&(prio3, agg_id), &encoded_input_share)
                    .expect("failed to decode input share");
            assert_eq!(&decoded_input_share, input_share);
            assert_eq!(
                input_share.encoded_len().unwrap(),
                encoded_input_share.len()
            );
        }

        let mut prepare_shares = Vec::new();
        let mut last_prepare_state = None;
        for (agg_id, input_share) in input_shares.iter().enumerate() {
            let (prepare_state, prepare_share) =
                prio3.prepare_init(&verify_key, agg_id, &(), nonce, &public_share, input_share)?;

            let encoded_prepare_state = prepare_state.get_encoded();
            let decoded_prepare_state =
                Prio3PrepareState::get_decoded_with_param(&(prio3, agg_id), &encoded_prepare_state)
                    .expect("failed to decode prepare state");
            assert_eq!(decoded_prepare_state, prepare_state);
            assert_eq!(
                prepare_state.encoded_len().unwrap(),
                encoded_prepare_state.len()
            );

            let encoded_prepare_share = prepare_share.get_encoded();
            let decoded_prepare_share =
                Prio3PrepareShare::get_decoded_with_param(&prepare_state, &encoded_prepare_share)
                    .expect("failed to decode prepare share");
            assert_eq!(decoded_prepare_share, prepare_share);
            assert_eq!(
                prepare_share.encoded_len().unwrap(),
                encoded_prepare_share.len()
            );

            prepare_shares.push(prepare_share);
            last_prepare_state = Some(prepare_state);
        }

        let prepare_message = prio3
            .prepare_shares_to_prepare_message(&(), prepare_shares)
            .unwrap();

        let encoded_prepare_message = prepare_message.get_encoded();
        let decoded_prepare_message = Prio3PrepareMessage::get_decoded_with_param(
            &last_prepare_state.unwrap(),
            &encoded_prepare_message,
        )
        .expect("failed to decode prepare message");
        assert_eq!(decoded_prepare_message, prepare_message);
        assert_eq!(
            prepare_message.encoded_len().unwrap(),
            encoded_prepare_message.len()
        );

        Ok(())
    }

    #[test]
    fn roundtrip_output_share() {
        let vdaf = Prio3::new_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Count, OutputShare<Field64>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_sum(2, 17).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Sum, OutputShare<Field64>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_histogram(2, 12, 3).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Histogram, OutputShare<Field64>>(&vdaf, &(), 12);
    }

    #[test]
    fn roundtrip_aggregate_share() {
        let vdaf = Prio3::new_count(2).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Count, AggregateShare<Field64>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_sum(2, 17).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Sum, AggregateShare<Field64>>(&vdaf, &(), 1);

        let vdaf = Prio3::new_histogram(2, 12, 3).unwrap();
        fieldvec_roundtrip_test::<Field64, Prio3Histogram, AggregateShare<Field64>>(&vdaf, &(), 12);
    }

    #[test]
    fn public_share_equality_test() {
        equality_comparison_test(&[
            Prio3PublicShare {
                joint_rand_parts: Some(Vec::from([Seed([0])])),
            },
            Prio3PublicShare {
                joint_rand_parts: Some(Vec::from([Seed([1])])),
            },
            Prio3PublicShare {
                joint_rand_parts: None,
            },
        ])
    }

    #[test]
    fn input_share_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3InputShare {
                measurement_share: Share::Leader(Vec::from([0])),
                proof_share: Share::Leader(Vec::from([1])),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified measurement share.
            Prio3InputShare {
                measurement_share: Share::Leader(Vec::from([100])),
                proof_share: Share::Leader(Vec::from([1])),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified proof share.
            Prio3InputShare {
                measurement_share: Share::Leader(Vec::from([0])),
                proof_share: Share::Leader(Vec::from([101])),
                joint_rand_blind: Some(Seed([2])),
            },
            // Modified joint_rand_blind.
            Prio3InputShare {
                measurement_share: Share::Leader(Vec::from([0])),
                proof_share: Share::Leader(Vec::from([1])),
                joint_rand_blind: Some(Seed([102])),
            },
            // Missing joint_rand_blind.
            Prio3InputShare {
                measurement_share: Share::Leader(Vec::from([0])),
                proof_share: Share::Leader(Vec::from([1])),
                joint_rand_blind: None,
            },
        ])
    }

    #[test]
    fn prepare_share_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareShare {
                verifier: Vec::from([0]),
                joint_rand_part: Some(Seed([1])),
            },
            // Modified verifier.
            Prio3PrepareShare {
                verifier: Vec::from([100]),
                joint_rand_part: Some(Seed([1])),
            },
            // Modified joint_rand_part.
            Prio3PrepareShare {
                verifier: Vec::from([0]),
                joint_rand_part: Some(Seed([101])),
            },
            // Missing joint_rand_part.
            Prio3PrepareShare {
                verifier: Vec::from([0]),
                joint_rand_part: None,
            },
        ])
    }

    #[test]
    fn prepare_message_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareMessage {
                joint_rand_seed: Some(Seed([0])),
            },
            // Modified joint_rand_seed.
            Prio3PrepareMessage {
                joint_rand_seed: Some(Seed([100])),
            },
            // Missing joint_rand_seed.
            Prio3PrepareMessage {
                joint_rand_seed: None,
            },
        ])
    }

    #[test]
    fn prepare_state_equality_test() {
        equality_comparison_test(&[
            // Default.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifier_len: 3,
            },
            // Modified measurement share.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([100])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifier_len: 3,
            },
            // Modified joint_rand_seed.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([101])),
                agg_id: 2,
                verifier_len: 3,
            },
            // Missing joint_rand_seed.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([0])),
                joint_rand_seed: None,
                agg_id: 2,
                verifier_len: 3,
            },
            // Modified agg_id.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 102,
                verifier_len: 3,
            },
            // Modified verifier_len.
            Prio3PrepareState {
                measurement_share: Share::Leader(Vec::from([0])),
                joint_rand_seed: Some(Seed([1])),
                agg_id: 2,
                verifier_len: 103,
            },
        ])
    }

    #[test]
    fn test_optimal_chunk_length() {
        // nonsense argument, but make sure it doesn't panic.
        optimal_chunk_length(0);

        // edge cases on either side of power-of-two jumps
        assert_eq!(optimal_chunk_length(1), 1);
        assert_eq!(optimal_chunk_length(2), 2);
        assert_eq!(optimal_chunk_length(3), 1);
        assert_eq!(optimal_chunk_length(18), 6);
        assert_eq!(optimal_chunk_length(19), 3);

        // additional arbitrary test cases
        assert_eq!(optimal_chunk_length(40), 6);
        assert_eq!(optimal_chunk_length(10_000), 79);
        assert_eq!(optimal_chunk_length(100_000), 393);

        // confirm that the chunk lengths are truly optimal
        for measurement_length in [2, 3, 4, 5, 18, 19, 40] {
            let optimal_chunk_length = optimal_chunk_length(measurement_length);
            let optimal_proof_length = Histogram::<Field64, ParallelSum<_, _>>::new(
                measurement_length,
                optimal_chunk_length,
            )
            .unwrap()
            .proof_len();
            for chunk_length in 1..=measurement_length {
                let proof_length =
                    Histogram::<Field64, ParallelSum<_, _>>::new(measurement_length, chunk_length)
                        .unwrap()
                        .proof_len();
                assert!(proof_length >= optimal_proof_length);
            }
        }
    }
}
