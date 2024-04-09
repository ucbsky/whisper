use prio::{
    field::Field128,
    vdaf::{prio3::Prio3, xof::XofShake128},
};

#[cfg(feature = "hhclient")]
pub mod hhclient;

#[cfg(feature = "hhserver")]
pub mod hhserver;

#[cfg(feature = "prioclient")]
pub mod prioclient;

#[cfg(feature = "prioserver")]
pub mod prioserver;

#[derive(Clone, Debug)]
pub enum AggFunc {
    SumVec,
    Histogram,
    Average,
}

pub const SEED_SIZE: usize = 16;

pub type F = Field128;
pub type SumVecType =
    prio::flp::types::SumVec<F, prio::flp::gadgets::ParallelSum<F, prio::flp::gadgets::Mul<F>>>;
pub type HistogramType =
    prio::flp::types::Histogram<F, prio::flp::gadgets::ParallelSum<F, prio::flp::gadgets::Mul<F>>>;
pub type AverageType = prio::flp::types::Average<F>;

pub const AVG_BITS: usize = 64;

/// Gadgets for prio3 microbenchmarks
#[derive(Clone)]
pub struct Prio3Gadgets {
    pub prio3sv: Option<Prio3<SumVecType, XofShake128, SEED_SIZE>>,
    pub prio3hist: Option<Prio3<HistogramType, XofShake128, SEED_SIZE>>,
    pub prio3avg: Option<Prio3<AverageType, XofShake128, SEED_SIZE>>,
}

impl Prio3Gadgets {
    pub fn new(agg_fn: &AggFunc, prio3_len: usize, prio3_chunk_len: usize) -> Self {
        let mut prio3: Prio3Gadgets = Prio3Gadgets {
            prio3sv: None,
            prio3hist: None,
            prio3avg: None,
        };
        match agg_fn {
            AggFunc::SumVec => {
                prio3.prio3sv =
                    Some(Prio3::new_sum_vec_256(2, 16, prio3_len, prio3_chunk_len).unwrap())
            }
            AggFunc::Histogram => {
                prio3.prio3hist =
                    Some(Prio3::new_histogram_256(2, prio3_len, prio3_chunk_len).unwrap())
            }
            AggFunc::Average => prio3.prio3avg = Some(Prio3::new_average_256(2, AVG_BITS).unwrap()),
        };
        prio3
    }
}

fn get_sketch_params(recovery_threshold: f32, num_bad_clients: usize) -> (usize, u16) {
    if recovery_threshold == 0.001 {
        if num_bad_clients > 0 {
            (17, 2048)
        } else {
            (17, 1024)
        }
    } else if recovery_threshold == 0.01 {
        if num_bad_clients > 0 {
            (14, 512)
        } else {
            (14, 256)
        }
    } else if recovery_threshold == 0.1 {
        if num_bad_clients > 0 {
            (10, 64)
        } else {
            (10, 32)
        }
    } else {
        panic!("unexpected recovery threshold");
    }
}
