extern crate algebra;
extern crate algebra_core;

use algebra::bls12_381::G1Projective as GroupProjective;
use algebra_core::{BigInteger384 as BaseFieldBigInt, BigInteger512 as FrWideBigInt};
use glv_lattice_basis::*;

fn main() {
    print_glv_params::<GroupProjective, FrWideBigInt, BaseFieldBigInt>();
}
