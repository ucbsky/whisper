use common::{prg, BitDecomposable, Group};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionWord {
    pub(crate) seed: prg::PrgSeed,
    pub(crate) bits: (bool, bool),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputCorrection<GOut> {
    pub(crate) word: GOut,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DPFKey<GOut> {
    key_id: bool,
    root_seed: prg::PrgSeed,
    correction_words: Vec<CorrectionWord>,
    output_correction: OutputCorrection<GOut>,
}

#[derive(Clone)]
pub struct EvalState {
    pub(crate) level: usize,
    pub(crate) seed: prg::PrgSeed,
    pub(crate) bit: bool,
}

// Extending "Map" to operate on tuples
trait TupleMapUtil<T, U> {
    type Output;
    fn map<F: FnMut(&T) -> U>(&self, f: F) -> Self::Output;
}

impl<T, U> TupleMapUtil<T, U> for (T, T) {
    type Output = (U, U);

    #[inline(always)]
    fn map<F: FnMut(&T) -> U>(&self, mut f: F) -> Self::Output {
        (f(&self.0), f(&self.1))
    }
}

// Used as return type for TupleUtil::iter_mut
type TupleMutIter<'a, T> =
    std::iter::Chain<std::iter::Once<(bool, &'a mut T)>, std::iter::Once<(bool, &'a mut T)>>;

pub trait TupleUtil<T> {
    fn map_mut<F: Fn(&mut T)>(&mut self, f: F);
    fn get(&self, cond: bool) -> &T;
    fn get_mut(&mut self, cond: bool) -> &mut T;
    fn iter_mut(&mut self) -> TupleMutIter<T>;
}

impl<T> TupleUtil<T> for (T, T) {
    #[inline(always)]
    fn map_mut<F: Fn(&mut T)>(&mut self, f: F) {
        f(&mut self.0);
        f(&mut self.1);
    }

    #[inline(always)]
    fn get(&self, cond: bool) -> &T {
        match cond {
            true => &self.1,
            false => &self.0,
        }
    }

    #[inline(always)]
    fn get_mut(&mut self, cond: bool) -> &mut T {
        match cond {
            true => &mut self.1,
            false => &mut self.0,
        }
    }

    fn iter_mut(&mut self) -> TupleMutIter<T> {
        std::iter::once((false, &mut self.0)).chain(std::iter::once((true, &mut self.1)))
    }
}

pub fn gen_correction(
    cur_bit: bool,
    bits: &mut (bool, bool),
    seeds: &mut (prg::PrgSeed, prg::PrgSeed),
) -> CorrectionWord {
    let expanded = seeds.map(|s| s.expand());

    // If alpha[i] = 0:
    //  - OnSpecialPath = L, ToCorrect = R
    // Else
    //  - OnSpecialPath = R, ToCorrect = L
    let on_special_path = cur_bit;
    let to_correct = !on_special_path;

    // Generate the correction word
    let cw = CorrectionWord {
        seed: expanded.0.seeds.get(to_correct) ^ expanded.1.seeds.get(to_correct),
        bits: (
            expanded.0.bits.0 ^ expanded.1.bits.0 ^ cur_bit ^ true,
            expanded.0.bits.1 ^ expanded.1.bits.1 ^ cur_bit,
        ),
    };

    // Set up the seeds and bits for next level; in-place update seeds and bits
    for (b, seed) in seeds.iter_mut() {
        // Pick up the new expanded seed and update seed
        *seed = expanded.get(b).seeds.get(on_special_path).clone();
        let mut new_bit = *expanded.get(b).bits.get(on_special_path);

        if *bits.get(b) {
            *seed = &*seed ^ &cw.seed;
            new_bit ^= cw.bits.get(on_special_path);
        }

        *bits.get_mut(b) = new_bit;
    }
    cw
}

pub fn gen_output_correction<GOut>(
    beta: &GOut,
    bits: &(bool, bool),
    seeds: &(prg::PrgSeed, prg::PrgSeed),
) -> OutputCorrection<GOut>
where
    GOut: prg::FromRng + Clone + Group + std::fmt::Debug,
{
    let converted = seeds.map(|s| s.convert::<GOut>());

    let mut oc = OutputCorrection { word: GOut::zero() };

    oc.word = beta.clone();
    oc.word.sub(&converted.0.word);
    oc.word.add(&converted.1.word);

    if bits.1 {
        oc.word.negate();
    }

    oc
}

impl<GOut> DPFKey<GOut>
where
    GOut: prg::FromRng + Clone + Group + std::fmt::Debug,
{
    pub fn gen<GIn>(alpha: GIn, beta: &GOut) -> (DPFKey<GOut>, DPFKey<GOut>)
    where
        GIn: prg::FromRng + Clone + Group + BitDecomposable + std::fmt::Debug,
    {
        let alpha_bits = alpha.decompose();

        let root_seeds = (prg::PrgSeed::random(), prg::PrgSeed::random());
        let root_bits = (false, true);

        let mut seeds = root_seeds.clone();
        let mut bits = root_bits;

        let mut correction_words: Vec<CorrectionWord> = Vec::new();
        let mut _output_correction = OutputCorrection::<GOut> { word: GOut::zero() };

        for (i, &cur_bit) in alpha_bits.iter().enumerate() {
            let is_last_level = i == alpha_bits.len() - 1;
            if is_last_level {
                correction_words.push(gen_correction(cur_bit, &mut bits, &mut seeds));
                let oc = gen_output_correction(beta, &bits, &seeds);
                _output_correction = oc;
            } else {
                correction_words.push(gen_correction(cur_bit, &mut bits, &mut seeds));
            }
        }

        (
            DPFKey::<GOut> {
                key_id: false,
                root_seed: root_seeds.0,
                correction_words: correction_words.clone(),
                output_correction: _output_correction.clone(),
            },
            DPFKey::<GOut> {
                key_id: true,
                root_seed: root_seeds.1,
                correction_words,
                output_correction: _output_correction,
            },
        )
    }

    pub fn eval_init(&self) -> EvalState {
        EvalState {
            level: 0,
            seed: self.root_seed.clone(),
            bit: self.key_id,
        }
    }

    fn eval_one_level(&self, state: &EvalState, cur_bit: bool) -> EvalState {
        // If cur_bit is 0, then expand only left direction meaning "left" should be 1 and right should be 0. Vice versa when cur_bit is 1.
        let expanded = state.seed.expand_direction(!cur_bit, cur_bit);

        // Fetch the expanded seed and bit
        let mut seed = expanded.seeds.get(cur_bit).clone();
        let mut new_bit = *expanded.bits.get(cur_bit);

        // Tau computation, but only for one side depending on the cur_bit
        if state.bit {
            seed = &seed ^ &self.correction_words[state.level].seed;
            new_bit ^= self.correction_words[state.level].bits.get(cur_bit);
        }

        EvalState {
            level: state.level + 1,
            seed,
            bit: new_bit,
        }
    }

    fn eval_output_level(&self, state: &EvalState, _cur_bit: bool) -> GOut {
        let converted = state.seed.convert::<GOut>();

        let mut word = converted.word;
        if state.bit {
            word.add(&self.output_correction.word);
        }

        if self.key_id {
            word.negate();
        }

        word
    }

    pub fn eval<GIn>(&self, idx: GIn) -> GOut
    where
        GIn: prg::FromRng + Clone + Group + BitDecomposable + std::fmt::Debug,
    {
        let idx_bits = idx.decompose();

        let mut state = self.eval_init();
        let mut res = GOut::zero();

        for (i, &cur_bit) in idx_bits.iter().enumerate() {
            let is_last_level = i == idx_bits.len() - 1;
            if is_last_level {
                state = self.eval_one_level(&state, cur_bit);
                let word = self.eval_output_level(&state, cur_bit);
                res = word;
            } else {
                state = self.eval_one_level(&state, cur_bit);
            }
        }

        res
    }

    pub fn eval_all<GIn>(&self) -> Vec<GOut>
    where
        GIn: prg::FromRng + Clone + Group + BitDecomposable + std::fmt::Debug,
    {
        let depth = GIn::bitsize();
        let mut res = Vec::new();
        // Storing all inner nodes minimizes the PRG calls compared to if they do in-order traversal of the tree using recursion or stack.
        let node_count = 2 * (1 << depth) - 1;
        let mut nodes: Vec<EvalState> = Vec::new();

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

        // Handle output level
        for i in ((1 << depth) - 1)..node_count {
            let state = nodes[i].clone();
            let converted = state.seed.convert::<GOut>();

            let mut word = converted.word;

            if state.bit {
                word.add(&self.output_correction.word);
            }

            if self.key_id {
                word.negate();
            }

            res.push(word);
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use crate::bucket::Bucket;
    use common::Share;

    use super::*;

    #[test]
    fn test_dpf() {
        let alpha = 190u16;
        let beta = 99910u64;
        let (key0, key1) = DPFKey::gen(alpha, &beta);
        let mx = u16::MAX;

        (0..mx).for_each(|idx| {
            if idx == alpha {
                assert!(key0.eval(idx).wrapping_add(key1.eval(idx)) == beta);
            } else {
                assert!(key0.eval(idx).wrapping_add(key1.eval(idx)) == 0);
            }
        });
    }

    #[test]
    fn test_dpf_small_gin() {
        use common::group::IntModN;
        let alpha = IntModN::<typenum::U1024>::from_u16(100u16);
        let beta = 99910u64;
        let (key0, key1) = DPFKey::gen::<IntModN<typenum::U1024>>(alpha, &beta);

        (0..1024).for_each(|idx| {
            let idx = IntModN::<typenum::U1024>::from_u16(idx);
            if idx.val == alpha.val {
                assert!(key0.eval(idx).wrapping_add(key1.eval(idx)) == beta);
            } else {
                assert!(key0.eval(idx).wrapping_add(key1.eval(idx)) == 0);
            }
        });
    }

    #[test]
    fn test_dpf_full_domain() {
        use std::time::Instant;

        let alpha = 190u16;
        let beta = 99910u64;
        let (key0, key1) = DPFKey::gen(alpha, &beta);

        let now = Instant::now();
        let res0 = key0.eval_all::<u16>();
        let res1 = key1.eval_all::<u16>();
        let elapsed = now.elapsed();
        println!("Time elapsed in EvalAll: {:.2?}", elapsed);

        (0..u16::MAX).for_each(|idx| {
            if idx == alpha {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == beta);
            } else {
                assert!(res0[idx as usize].wrapping_add(res1[idx as usize]) == 0);
            }
        });
    }

    #[test]
    fn test_dpf_full_domain_large_gout() {
        let alpha = 190u16;
        let beta = Bucket::<u64>::random();
        let zero = Bucket::<u64>::zero();
        let (key0, key1) = DPFKey::gen(alpha, &beta);

        let mut res0 = key0.eval_all::<u16>();
        let res1 = key1.eval_all::<u16>();

        (0..u16::MAX).for_each(|idx| {
            res0[idx as usize].add(&res1[idx as usize]);
            if idx == alpha {
                assert_eq!(res0[idx as usize], beta);
            } else {
                assert_eq!(res0[idx as usize], zero);
            }
        });
    }
}
