#![allow(unused)]
#[cfg(feature = "cuda")]
use accel::*;
use algebra::bw6_761::G1Projective;
use algebra_core::{
    curves::ProjectiveCurve, fields::PrimeField, BatchGroupArithmeticSlice, UniformRand,
};
#[cfg(feature = "cuda")]
use gpu_standalone::bw6_761_g1_scalar_mul_kernel::*;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

const LOG2_N: usize = 20;
// Job size needs to be at least 1 << 17
const JOB_SIZE: usize = 1 << 17;
// We support n_threads up to JOB_SIZE / CHUNK_SIZE
const CHUNK_SIZE: usize = 1 << 12;
const CUDA_GROUP_SIZE: usize = 1 << 5;

pub type G1 = G1Projective;
pub type BigInt = <<G1Projective as ProjectiveCurve>::ScalarField as PrimeField>::BigInt;

use crate::helpers::create_pseudo_uniform_random_elems;

#[cfg(feature = "cuda")]
fn main() -> error::Result<()> {
    let mut rng = XorShiftRng::seed_from_u64(1231275789u64);

    let n = 1 << LOG2_N;
    let mut exps_h = Vec::with_capacity(n);

    let now = std::time::Instant::now();
    let mut bases_h: Vec<<G1 as ProjectiveCurve>::Affine> =
        create_pseudo_uniform_random_elems(&mut rng, LOG2_N);
    for _ in 0..n {
        exps_h.push(<G1 as ProjectiveCurve>::ScalarField::rand(&mut rng).into_repr());
    }
    println!("Generated random elems: {}us", now.elapsed().as_micros());

    let bases_d = bases_h.to_vec();

    let mut exps_cpu = exps_h.to_vec();
    let now = std::time::Instant::now();
    bases_h
        .par_chunks_mut(CHUNK_SIZE)
        .zip(exps_cpu.par_chunks_mut(CHUNK_SIZE))
        .for_each(|(b, s)| b[..].batch_scalar_mul_in_place(&mut s[..], 4));
    println!("CPU mul: {}us", now.elapsed().as_micros());

    if Device::init() {
        let n_devices = Device::get_count().unwrap();

        for _ in 0..10 {
            let now = std::time::Instant::now();
            let bases_static = cpu_gpu_static_partition_run_kernel(
                &bases_d[..],
                &exps_h[..],
                CUDA_GROUP_SIZE,
                CHUNK_SIZE,
            )
            .to_vec();
            println!(
                "GPU+CPU static partition mul: {}us",
                now.elapsed().as_micros()
            );
        }
        let now = std::time::Instant::now();
        let bases_static = cpu_gpu_static_partition_run_kernel(
            &bases_d[..],
            &exps_h[..],
            CUDA_GROUP_SIZE,
            CHUNK_SIZE,
        )
        .to_vec();
        println!(
            "GPU+CPU static partition mul: {}us",
            now.elapsed().as_micros()
        );

        let now = std::time::Instant::now();
        let bases = (0..n_devices)
            .into_par_iter()
            .flat_map(|i| {
                let device = Device::nth(i).unwrap();
                let ctx = device.create_context();

                let _pf = Profiler::start(&ctx);
                cpu_gpu_load_balance_run_kernel(
                    &ctx,
                    &bases_d[..],
                    &exps_h[..],
                    CUDA_GROUP_SIZE,
                    JOB_SIZE,
                    CHUNK_SIZE,
                )
                .to_vec()
            })
            .collect::<Vec<_>>();
        println!("GPU+CPU mul: {}us", now.elapsed().as_micros());

        let now = std::time::Instant::now();
        let mut bases_gpu = (0..n_devices)
            .into_par_iter()
            .flat_map(|i| {
                let device = Device::nth(i).unwrap();
                let ctx = device.create_context();
                let _pf = Profiler::start(&ctx);
                par_run_kernel(&ctx, &bases_d[..], &exps_h[..], CUDA_GROUP_SIZE).to_vec()
            })
            .collect::<Vec<_>>();
        println!("GPU mul: {}us", now.elapsed().as_micros());
        G1::batch_normalization(&mut bases_gpu[..]);

        for ((b_h, b_s), (b, b_gpu)) in bases_h
            .into_iter()
            .zip(bases_static.into_iter())
            .zip(bases.into_iter().zip(bases_gpu.into_iter()))
        {
            assert_eq!(b_h, b_s);
            assert_eq!(b_h, b_gpu.into_affine());
            assert_eq!(b_h, b);
        }
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}

mod helpers {
    use algebra_core::{
        cfg_chunks_mut, AffineCurve, BatchGroupArithmeticSlice, BigInteger64, ProjectiveCurve,
        UniformRand,
    };
    use rand::{distributions::Uniform, prelude::Distribution, Rng};

    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    pub fn create_pseudo_uniform_random_elems<C: AffineCurve, R: Rng>(
        rng: &mut R,
        max_logn: usize,
    ) -> Vec<C> {
        const AFFINE_BATCH_SIZE: usize = 4096;
        println!("Starting");
        let now = std::time::Instant::now();
        // Generate pseudorandom group elements
        let step = Uniform::new(0, 1 << (max_logn + 5));
        let elem = C::Projective::rand(rng).into_affine();
        let mut random_elems = vec![elem; 1 << max_logn];
        let mut scalars: Vec<BigInteger64> = (0..1 << max_logn)
            .map(|_| BigInteger64::from(step.sample(rng)))
            .collect();
        cfg_chunks_mut!(random_elems, AFFINE_BATCH_SIZE)
            .zip(cfg_chunks_mut!(scalars, AFFINE_BATCH_SIZE))
            .for_each(|(e, s)| {
                e[..].batch_scalar_mul_in_place::<BigInteger64>(&mut s[..], 1);
            });

        println!("Initial generation: {:?}", now.elapsed().as_micros());
        random_elems
    }
}
