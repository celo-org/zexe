use crate::fields::FpParameters;
use crate::{
    cfg_chunks_mut,
    curves::{batch_bucketed_add, BatchGroupArithmeticSlice, BucketPosition, BATCH_SIZE},
    AffineCurve, PrimeField, ProjectiveCurve, Vec,
};
use num_traits::identities::Zero;

use core::fmt;

use rand::Rng;
#[cfg(feature = "parallel")]
use {rand::thread_rng, rayon::prelude::*};

#[derive(Debug, Clone)]
pub struct VerificationError;

const MAX_BUCKETS_FOR_FULL_CHECK: usize = 8;

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Verification Error. Not in subgroup")
    }
}

fn verify_points<C: AffineCurve, R: Rng>(
    points: &[C],
    num_buckets: usize,
    // Only pass new_security_param if possibly recursing
    new_security_param: Option<f64>,
    rng: &mut R,
) -> Result<(), VerificationError> {
    let n_points = points.len();
    let mut bucket_assign = Vec::with_capacity(points.len());
    for i in 0..n_points {
        bucket_assign.push(BucketPosition {
            bucket: rng.gen_range(0, num_buckets) as u32,
            position: i as u32,
        });
    }
    let _now = timer!();
    let mut buckets = batch_bucketed_add(num_buckets, &mut points.to_vec(), &mut bucket_assign[..]);
    timer_println!(_now, format!("bucketed add({}, {})", num_buckets, n_points));

    // We use the batch_scalar_mul to check the subgroup condition if
    // there are sufficient number of buckets. For SW curves, the number
    // elems for the batch mul to become useful is around 2^24.
    // This is almost certainly not going to be used for the recursive test.
    if num_buckets <= MAX_BUCKETS_FOR_FULL_CHECK || new_security_param == None {
        let _now = timer!();
        let verification_failure = if num_buckets >= BATCH_SIZE {
            cfg_chunks_mut!(buckets, BATCH_SIZE).for_each(|e| {
                let length = e.len();
                e[..].batch_scalar_mul_in_place::<<C::ScalarField as PrimeField>::BigInt>(
                    &mut vec![C::ScalarField::modulus().into(); length][..],
                    4,
                );
            });
            !buckets.iter().all(|&p| p.is_zero())
        } else {
            !buckets
                .iter()
                .all(|&b| b.into_projective().mul(C::ScalarField::modulus()).is_zero())
        };
        timer_println!(_now, "mul by modulus");
        if verification_failure {
            return Err(VerificationError);
        }
    } else {
        // Since !new_security_param.is_none():
        let new_security_param = new_security_param.unwrap();

        if buckets.len() > 4096 {
            batch_verify_in_subgroup_recursive_inner(&buckets[..], new_security_param, rng)?;
        } else {
            batch_verify_in_subgroup_proj_recursive(
                &buckets
                    .iter()
                    .map(|&p| p.into())
                    .collect::<Vec<C::Projective>>()[..],
                new_security_param,
                rng,
            )?;
        }
    }
    Ok(())
}

fn run_rounds<C: AffineCurve, R: Rng>(
    points: &[C],
    num_buckets: usize,
    num_rounds: usize,
    new_security_param: Option<f64>,
    rng: &mut R,
) -> Result<(), VerificationError> {
    #[cfg(feature = "parallel")]
    if num_rounds > 2 {
        use std::sync::Arc;
        // This may take up a lot of memory
        let ref_points = Arc::new(points.to_vec());
        let mut threads = vec![];
        for _ in 0..num_rounds {
            let ref_points_thread = ref_points.clone();
            // We only use std when a multicore environment is available
            threads.push(std::thread::spawn(
                move || -> Result<(), VerificationError> {
                    let mut rng = &mut thread_rng();
                    verify_points(
                        &ref_points_thread[..],
                        num_buckets,
                        new_security_param,
                        &mut rng,
                    )?;
                    Ok(())
                },
            ));
        }
        for thread in threads {
            thread.join().unwrap()?;
        }
    } else {
        for _ in 0..num_rounds {
            verify_points(points, num_buckets, new_security_param, rng)?;
        }
    }

    #[cfg(not(feature = "parallel"))]
    for _ in 0..num_rounds {
        verify_points(points, num_buckets, new_security_param, rng)?;
    }

    Ok(())
}

pub fn batch_verify_in_subgroup<C: AffineCurve, R: Rng>(
    points: &[C],
    security_param: usize,
    rng: &mut R,
) -> Result<(), VerificationError> {
    #[cfg(feature = "std")]
    let cost_estimate = (<C::ScalarField as PrimeField>::Params::MODULUS_BITS as f64
        * (0.5 * 7.0 / 6.0 * 0.8 + 1.0 / 5.0))
        .ceil() as usize;
    #[cfg(not(feature = "std"))]
    let cost_estimate = <C::ScalarField as PrimeField>::Params::MODULUS_BITS as usize * 5 / 4;

    let (num_buckets, num_rounds, _) = get_max_bucket(
        security_param as f64,
        points.len(),
        // We estimate the costs of a single scalar multiplication in the batch affine, w-NAF GLV case as
        // 7/6 * 0.5 * n_bits * 0.8 (doubling) + 0.5 * 1/(w + 1) * n_bits (addition)
        // We take into account that doubling in the batch add model is cheaper as it requires less cache use
        cost_estimate,
    );
    run_rounds(points, num_buckets, num_rounds, None, rng)?;
    Ok(())
}

pub fn batch_verify_in_subgroup_recursive<C: AffineCurve, R: Rng>(
    points: &[C],
    security_param: usize,
    rng: &mut R,
) -> Result<(), VerificationError> {
    let (num_buckets, num_rounds, new_security_param) =
        get_max_bucket_recursive(security_param as f64, points.len(), true);
    run_rounds(
        points,
        num_buckets,
        num_rounds,
        Some(new_security_param),
        rng,
    )?;
    Ok(())
}

fn batch_verify_in_subgroup_recursive_inner<C: AffineCurve, R: Rng>(
    points: &[C],
    security_param: f64,
    rng: &mut R,
) -> Result<(), VerificationError> {
    let (num_buckets, num_rounds, new_security_param) =
        get_max_bucket_recursive(security_param, points.len(), false);
    run_rounds(
        points,
        num_buckets,
        num_rounds,
        Some(new_security_param),
        rng,
    )?;
    Ok(())
}

fn batch_verify_in_subgroup_proj_recursive<C: ProjectiveCurve, R: Rng>(
    points: &[C],
    security_param: f64,
    rng: &mut R,
) -> Result<(), VerificationError> {
    let (num_buckets, num_rounds, new_security_param) =
        get_max_bucket_recursive(security_param, points.len(), false);
    for _ in 0..num_rounds {
        let mut bucket_assign = Vec::with_capacity(points.len());
        for _ in 0..points.len() {
            bucket_assign.push(rng.gen_range(0, num_buckets));
        }
        // If our batch size is too small, we do the naive bucket add
        let zero = C::zero();
        let mut buckets = vec![zero; num_buckets];
        for (p, a) in points.iter().zip(bucket_assign) {
            buckets[a].add_assign(p);
        }

        if num_buckets <= MAX_BUCKETS_FOR_FULL_CHECK {
            if !buckets.iter().all(|b| {
                b.mul(<C::ScalarField as PrimeField>::Params::MODULUS)
                    .is_zero()
            }) {
                return Err(VerificationError);
            }
        } else {
            batch_verify_in_subgroup_proj_recursive(&buckets[..], new_security_param, rng)?;
        }
    }
    Ok(())
}

fn get_max_bucket_recursive(
    security_param: f64,
    n_elems: usize,
    is_first: bool,
) -> (usize, usize, f64) {
    #[cfg(feature = "std")]
    {
        let log2_constant_multiplier = if is_first { 1f64 } else { 2f64 };
        let mut log2_num_buckets = 1f64; // m_i
        let num_rounds = |log2_num_buckets: f64| -> usize {
            (security_param / log2_num_buckets).ceil() as usize
        };

        while num_rounds(log2_num_buckets)
            * (1 << log2_constant_multiplier as usize)
            * (2f64
                .powf(log2_num_buckets + log2_constant_multiplier)
                .ceil() as usize)
            < n_elems
            && num_rounds(log2_num_buckets + 0.1) > 1
        {
            log2_num_buckets += 0.1;
        }
        (
            2f64.powf(log2_num_buckets + log2_constant_multiplier)
                .ceil() as usize, // number of buckets
            num_rounds(log2_num_buckets), // number of rounds
            log2_num_buckets + log2_constant_multiplier, // new security param
        )
    }

    #[cfg(not(feature = "std"))]
    {
        let security_param = security_param as usize;
        let log2_constant_multiplier = if is_first { 1 } else { 2 };
        let mut log2_num_buckets: u32 = 1;
        let num_rounds = |log2_num_buckets: u32| -> usize {
            (security_param as usize - 1) / (log2_num_buckets as usize) + 1
        };

        while num_rounds(log2_num_buckets)
            * (1 << log2_constant_multiplier)
            * (2_i32.pow(log2_num_buckets + log2_constant_multiplier) as usize)
            < n_elems
            && num_rounds(log2_num_buckets + 1) > 1
        {
            log2_num_buckets += 1;
        }
        (
            2_i32.pow(log2_num_buckets + log2_constant_multiplier) as usize, // number of buckets
            num_rounds(log2_num_buckets),                                    // number of rounds
            (log2_num_buckets + log2_constant_multiplier) as f64,   // new security param
        )
    }
}

/// We get the greatest power of 2 number of buckets such that we minimise the
/// number of rounds while satisfying the constraint that
/// n_rounds * buckets * next_check_per_elem_cost < n
fn get_max_bucket(
    security_param: f64,
    n_elems: usize,
    next_check_per_elem_cost: usize,
) -> (usize, usize, f64) {
    #[cfg(feature = "std")]
    {
        let mut log2_num_buckets = 1f64;
        let num_rounds = |log2_num_buckets: f64| -> usize {
            (security_param / log2_num_buckets).ceil() as usize
        };

        while num_rounds(log2_num_buckets)
            * next_check_per_elem_cost
            * (2f64.powf(log2_num_buckets).ceil() as usize)
            < n_elems
            && num_rounds(log2_num_buckets + 0.1) > 1
        {
            log2_num_buckets += 0.1;
        }
        (
            2f64.powf(log2_num_buckets).ceil() as usize, // number of buckets
            num_rounds(log2_num_buckets),                // number of rounds
            log2_num_buckets,                            // new security param
        )
    }

    #[cfg(not(feature = "std"))]
    {
        let security_param = security_param as usize;
        let mut log2_num_buckets: u32 = 1;
        let num_rounds = |log2_num_buckets: u32| -> usize {
            (security_param - 1) / (log2_num_buckets as usize) + 1
        };

        while num_rounds(log2_num_buckets)
            * next_check_per_elem_cost
            * (2_i32.pow(log2_num_buckets) as usize)
            < n_elems
            && num_rounds(log2_num_buckets + 1) > 1
        {
            log2_num_buckets += 1;
        }
        (
            2_i32.pow(log2_num_buckets) as usize, // number of buckets
            num_rounds(log2_num_buckets),         // number of rounds
            log2_num_buckets as f64,              // new security param
        )
    }
}
