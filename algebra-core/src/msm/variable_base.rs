use crate::{
    batch_bucketed_add, batch_bucketed_add_multiple,
    prelude::{AffineCurve, BigInteger, FpParameters, One, PrimeField, ProjectiveCurve, Zero},
    BucketPosition, Vec,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct VariableBaseMSM;

impl VariableBaseMSM {
    fn msm_inner<G: AffineCurve>(
        bases: &[G],
        scalars: &[<G::ScalarField as PrimeField>::BigInt],
    ) -> G::Projective
    where
        G::Projective: ProjectiveCurve<Affine = G>,
    {
        let c = if scalars.len() < 32 {
            3
        } else {
            super::ln_without_floats(scalars.len()) + 2
        };

        let num_bits = <G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize;
        let fr_one = G::ScalarField::one().into_repr();

        let zero = G::Projective::zero();
        let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

        #[cfg(feature = "parallel")]
        let window_starts_iter = window_starts.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let window_starts_iter = window_starts.into_iter();

        // Each window is of size `c`.
        // We divide up the bits 0..num_bits into windows of size `c`, and
        // in parallel process each such window.
        let window_sums: Vec<_> = window_starts_iter
            .map(|w_start| {
                let mut res = zero;
                // We don't need the "zero" bucket, so we only have 2^c - 1 buckets
                let log2_n_bucket = if (w_start % c) != 0 { w_start % c } else { c };
                let mut buckets = vec![zero; (1 << log2_n_bucket) - 1];

                scalars
                    .iter()
                    .zip(bases)
                    .filter(|(s, _)| !s.is_zero())
                    .for_each(|(&scalar, base)| {
                        if scalar == fr_one {
                            // We only process unit scalars once in the first window.
                            if w_start == 0 {
                                res.add_assign_mixed(base);
                            }
                        } else {
                            let mut scalar = scalar;

                            // We right-shift by w_start, thus getting rid of the
                            // lower bits.
                            scalar.divn(w_start as u32);

                            // We mod the remaining bits by the window size.
                            let scalar = scalar.as_ref()[0] % (1 << c);

                            // If the scalar is non-zero, we update the corresponding
                            // bucket.
                            // (Recall that `buckets` doesn't have a zero bucket.)
                            if scalar != 0 {
                                buckets[(scalar - 1) as usize].add_assign_mixed(base);
                            }
                        }
                    });
                let buckets = G::Projective::batch_normalization_into_affine(&buckets);

                let mut running_sum = G::Projective::zero();
                for b in buckets.into_iter().rev() {
                    running_sum.add_assign_mixed(&b);
                    res += &running_sum;
                }

                (res, log2_n_bucket)
            })
            .collect();

        // We store the sum for the lowest window.
        let lowest = window_sums.first().unwrap().0;

        // We're traversing windows from high to low.
        lowest
            + &window_sums[1..].iter().rev().fold(
                zero,
                |total: G::Projective, (sum_i, window_size): &(G::Projective, usize)| {
                    let mut total = total + sum_i;
                    for _ in 0..*window_size {
                        total.double_in_place();
                    }
                    total
                },
            )
    }

    pub fn multi_scalar_mul<G: AffineCurve>(
        bases: &[G],
        scalars: &[<G::ScalarField as PrimeField>::BigInt],
    ) -> G::Projective {
        Self::msm_inner(bases, scalars)
    }

    pub fn multi_scalar_mul_batched<G: AffineCurve, BigInt: BigInteger>(
        bases: &[G],
        scalars: &[BigInt],
        num_bits: usize,
    ) -> G::Projective {
        let c = if scalars.len() < 32 {
            1
        } else {
            super::ln_without_floats(scalars.len()) + 2
        };

        let zero = G::Projective::zero();
        let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

        #[cfg(feature = "parallel")]
        let window_starts_iter = window_starts.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let window_starts_iter = window_starts.into_iter();

        // Each window is of size `c`.
        // We divide up the bits 0..num_bits into windows of size `c`, and
        // in parallel process each such window.
        let window_sums: Vec<_> = window_starts_iter
            .map(|w_start| {
                // We don't need the "zero" bucket, so we only have 2^c - 1 buckets
                let log2_n_bucket = if (w_start % c) != 0 { w_start % c } else { c };
                let n_buckets = (1 << log2_n_bucket) - 1;

                let _now = timer!();
                let mut bucket_positions: Vec<_> = scalars
                    .iter()
                    .enumerate()
                    .map(|(pos, &scalar)| {
                        let mut scalar = scalar;

                        // We right-shift by w_start, thus getting rid of the
                        // lower bits.
                        scalar.divn(w_start as u32);

                        // We mod the remaining bits by the window size.
                        let res = (scalar.as_ref()[0] % (1 << c)) as i32;
                        BucketPosition {
                            bucket: (res - 1) as u32,
                            position: pos as u32,
                        }
                    })
                    .collect();
                timer_println!(_now, "scalars->buckets");

                let _now = timer!();

                let buckets =
                    batch_bucketed_add::<G>(n_buckets, &bases[..], &mut bucket_positions[..]);
                timer_println!(_now, "bucket add");

                let _now = timer!();
                let mut res = zero;
                let mut running_sum = G::Projective::zero();
                for b in buckets.into_iter().rev() {
                    running_sum.add_assign_mixed(&b);
                    res += &running_sum;
                }
                timer_println!(_now, "accumulating sums");
                (res, log2_n_bucket)
            })
            .collect();

        // We store the sum for the lowest window.
        let lowest = window_sums.first().unwrap().0;

        // We're traversing windows from high to low.
        lowest
            + &window_sums[1..].iter().rev().fold(
                zero,
                |total: G::Projective, (sum_i, window_size): &(G::Projective, usize)| {
                    let mut total = total + sum_i;
                    for _ in 0..*window_size {
                        total.double_in_place();
                    }
                    total
                },
            )
    }

    pub fn multi_scalar_mul_scaled<G: AffineCurve, BigInt: BigInteger>(
        bases: &[G],
        scalars: &[BigInt],
        num_bits: usize,
    ) -> G::Projective {
        if cfg!(not(any(feature = "parallel", feature = "std"))) {
            Self::multi_scalar_mul_batched::<G, BigInt>(bases, scalars, num_bits)
        } else {
            const PARALLELISM: usize = 1;

            let c = if scalars.len() < 32 {
                1
            } else {
                super::ln_without_floats(scalars.len()) + 2
            };

            let zero = G::Projective::zero();
            let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

            let mut n_buckets = vec![1 << c; window_starts.len() - 1];
            let w_last = window_starts.last().unwrap();
            let log2_n_bucket = if (w_last % c) != 0 { w_last % c } else { c };
            n_buckets.push(1 << log2_n_bucket);

            let _now = timer!();
            let mut bucket_positions_all: Vec<Vec<_>> = window_starts
                .par_iter()
                .map(|w_start| {
                    let bucket_positions: Vec<_> = scalars
                        .par_iter()
                        .enumerate()
                        .map(|(pos, &scalar)| {
                            let mut scalar = scalar;

                            // We right-shift by w_start, thus getting rid of the
                            // lower bits.
                            scalar.divn(*w_start as u32);

                            // We mod the remaining bits by the window size.
                            let res = (scalar.as_ref()[0] % (1 << c)) as i32;
                            BucketPosition {
                                bucket: res as u32,
                                position: pos as u32,
                            }
                        })
                        .collect();
                    bucket_positions
                })
                .collect();
            timer_println!(_now, "scalars->buckets");

            let _now = timer!();
            let buckets_all = batch_bucketed_add_multiple::<G>(
                &n_buckets[..],
                &bases[..],
                &mut bucket_positions_all[..],
                PARALLELISM,
            );
            timer_println!(_now, "bucket add");

            let _now = timer!();
            let window_sums: Vec<_> = buckets_all
                .par_iter()
                .map(|bucket_v| {
                    let len = bucket_v.len();
                    let chunk_size = (len - 1) / PARALLELISM + 1;
                    bucket_v
                        .par_chunks(chunk_size)
                        .enumerate()
                        .map(|(i, buckets_full)| {
                            let first = buckets_full[0];
                            let buckets = &buckets_full[1..];

                            let mut res = zero;
                            let mut running_sum = G::Projective::zero();
                            for b in buckets.iter().rev() {
                                running_sum.add_assign_mixed(&b);
                                res += &running_sum;
                            }
                            running_sum.add_assign_mixed(&first);
                            let factor = <G::ScalarField as PrimeField>::BigInt::from(
                                (i * chunk_size) as u64,
                            );
                            res + &running_sum.mul(factor)
                        })
                        .sum()
                })
                .collect();
            timer_println!(_now, "accumulating sums");
            // We store the sum for the lowest window.
            let lowest = *window_sums.first().unwrap();

            let _now = timer!();
            // We're traversing windows from high to low.
            let res = lowest
                + &window_sums[1..].iter().rev().fold(
                    zero,
                    |total: G::Projective, sum_i: &G::Projective| {
                        let mut total = total + sum_i;
                        for _ in 0..c {
                            total.double_in_place();
                        }
                        total
                    },
                );
            timer_println!(_now, "final doubling");
            res
        }
    }
}
