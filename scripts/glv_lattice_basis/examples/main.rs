extern crate algebra;
extern crate algebra_core;

use algebra::bn254::G2Projective as GroupProjective;
use algebra_core::{BigInteger512 as FrWideBigInt, BigInteger512 as BaseFieldBigInt};
use glv_lattice_basis::*;

fn main() {
    print_glv_params::<GroupProjective, FrWideBigInt, BaseFieldBigInt>();
}
