#[macro_export]
macro_rules! specialise_affine_to_proj {
    ($GroupProjective: ident) => {
        use crate::biginteger::BigInteger;

        #[derive(Derivative)]
        #[derivative(
            Copy(bound = "P: Parameters"),
            Clone(bound = "P: Parameters"),
            PartialEq(bound = "P: Parameters"),
            Eq(bound = "P: Parameters"),
            Debug(bound = "P: Parameters"),
            Hash(bound = "P: Parameters")
        )]

        pub struct GroupAffine<P: Parameters> {
            pub x: P::BaseField,
            pub y: P::BaseField,
            pub infinity: bool,
            #[derivative(Debug = "ignore")]
            _params: PhantomData<P>,
        }

        impl<P: Parameters> AffineCurve for GroupAffine<P> {
            const COFACTOR: &'static [u64] = P::COFACTOR;
            type BaseField = P::BaseField;
            type ScalarField = P::ScalarField;
            type Projective = $GroupProjective<P>;

            fn prime_subgroup_generator() -> Self {
                Self::new(
                    P::AFFINE_GENERATOR_COEFFS.0,
                    P::AFFINE_GENERATOR_COEFFS.1,
                    false,
                )
            }

            fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
                P::BaseField::from_random_bytes_with_flags(bytes).and_then(|(x, flags)| {
                    let infinity_flag_mask = SWFlags::Infinity.u8_bitmask();
                    let positive_flag_mask = SWFlags::PositiveY.u8_bitmask();
                    // if x is valid and is zero and only the infinity flag is set, then parse this
                    // point as infinity. For all other choices, get the original point.
                    if x.is_zero() && flags == infinity_flag_mask {
                        Some(Self::zero())
                    } else {
                        let is_positive = flags & positive_flag_mask != 0;
                        Self::get_point_from_x(x, is_positive)
                    }
                })
            }

            fn mul<S: Into<<Self::ScalarField as PrimeField>::BigInt>>(&self, by: S) -> Self::Projective {
                let bits = BitIterator::new(by.into());
                self.mul_bits(bits)
            }

            #[inline]
            fn mul_by_cofactor_to_projective(&self) -> Self::Projective {
                self.scale_by_cofactor()
            }

            fn mul_by_cofactor_inv(&self) -> Self {
                self.mul(P::COFACTOR_INV).into()
            }

            // TODO: Generalise to A != 0
            // Computes [-p, p, -3p, 3p, ..., -2^wp, 2^wp]
            fn batch_wnaf_tables(bases: &[Self], w: usize) -> Vec<Vec<Self>> {
                let half_size = 1 << w;
                let batch_size = bases.len();

                let mut tables =
                    vec![Vec::<Self>::with_capacity(half_size); batch_size];

                let mut a_2 = bases[..].to_vec();
                let mut tmp = bases[..].to_vec();

                GroupAffine::<P>::batch_double_in_place_with_edge_cases(
                    &mut a_2,
                    (0..batch_size).collect()
                );

                for i in 0..half_size {
                    if i != 0 {
                        GroupAffine::<P>::batch_add_in_place_with_edge_cases(
                            &mut tmp,
                            &mut a_2.to_vec()[..],
                            (0..batch_size).map(|x| (x, x)).collect()
                        );
                    }

                    for (table, p) in tables.iter_mut().zip(&tmp) {
                        // table.push(p.clone().neg());
                        table.push(p.clone());
                    }
                }
                tables
            }

            // This function mutates the scalars in place
            // We can make this more generic in the future to use other than u16.
            // fn batch_wnaf_opcode_recoding_<BigInt: BigInteger + AsRef<[u64]>>(
            //     scalars: &mut [BigInt],
            //     w: usize
            // ) -> Vec<Vec<Option<u16>>> {
            //     assert!(w > 0);
            //     let batch_size = scalars.len();
            //     let window_size: u16 = 1 << (w + 1);
            //     let half_window_size: u16 = 1 << w;
            //
            //     let mut op_code_vectorised =
            //         Vec::<Vec<Option<u16>>>::with_capacity(scalars[0].as_ref().len() * 64);
            //
            //     let mut all_none = false;
            //     while !all_none {
            //         let mut opcode_row = Vec::with_capacity(batch_size);
            //
            //         for s in scalars.iter_mut() {
            //             if s.is_zero() {
            //                 opcode_row.push(None);
            //             } else {
            //                 let op = if s.is_odd() {
            //                     let mut z: u16 = (s.as_ref()[0] as u16) % window_size;
            //
            //                     if z < half_window_size {
            //                         s.sub_noborrow(&BigInt::from(z as u64));
            //                     } else {
            //                         let tmp = window_size - z;
            //                         s.add_nocarry(&BigInt::from(tmp as u64));
            //                         z = tmp - 1; // z = 0, 2, ..., 2^(w+1) - 2
            //                     }
            //                     z
            //                 } else {
            //                     half_window_size // We encode 0s to be 2^(w+1)
            //                 };
            //                 opcode_row.push(Some(op));
            //                 s.div2();
            //             }
            //         }
            //
            //         all_none = opcode_row.iter().all(|x| x.is_none());
            //         if !all_none {
            //             op_code_vectorised.push(opcode_row);
            //         // } else {
            //         //     break;
            //         }
            //     }
            //     op_code_vectorised
            // }

            // This function mutates the scalars in place
            // We can make this more generic in the future to use other than u16.
            fn batch_wnaf_opcode_recoding<BigInt: BigInteger + AsRef<[u64]>>(
                scalars: &mut [BigInt],
                w: usize
            ) -> Vec<Vec<Option<i16>>> {
                assert!(w > 0);
                let batch_size = scalars.len();
                let window_size: i16 = 1 << (w + 1);
                let half_window_size: i16 = 1 << w;

                let mut op_code_vectorised =
                    Vec::<Vec<Option<i16>>>::with_capacity(scalars[0].as_ref().len() * 64);

                let mut all_none = false;
                while !all_none {
                    let mut opcode_row = Vec::with_capacity(batch_size);

                    for s in scalars.iter_mut() {
                        if s.is_zero() {
                            opcode_row.push(None);
                        } else {
                            let op = if s.is_odd() {
                                let mut z: i16 = (s.as_ref()[0] % (1 << (w + 1))) as i16;

                                if z < half_window_size {
                                    s.sub_noborrow(&BigInt::from(z as u64));
                                } else {
                                    z = z - window_size;
                                    s.add_nocarry(&BigInt::from((-z) as u64));
                                }
                                z
                            } else {
                                0 // We encode 0s to be 2^(w+1)
                            };
                            opcode_row.push(Some(op));
                            s.div2();
                        }
                    }

                    all_none = opcode_row.iter().all(|x| x.is_none());
                    if !all_none {
                        op_code_vectorised.push(opcode_row);
                    // } else {
                    //     break;
                    }
                }
                op_code_vectorised
            }

            // This implementation of batch group ops takes particular
            // care to make most use of points fetched from memory
            // And to reuse memory to prevent reallocations
            // It is directly adapted from Aztec's code.

            // https://github.com/AztecProtocol/barretenberg/blob/standardplonkjson/barretenberg/src/
            // aztec/ecc/curves/bn254/scalar_multiplication/scalar_multiplication.cpp
            #[inline]
            fn batch_double_in_place_with_edge_cases(
                bases: &mut [Self],
                index: Vec<usize>,
            ) {
                let mut inversion_tmp = P::BaseField::one();
                let mut scratch_space = Vec::new(); // with_capacity? How to get size?
                // We run two loops over the data separated by an inversion
                #[cfg(feature = "prefetch")]
                let mut prefetch_iter = index.iter();
                #[cfg(feature = "prefetch")]
                {
                    prefetch_iter.next();
                }

                for idx in index.iter() {
                    // Prefetch next group into cache
                    #[cfg(feature = "prefetch")]
                    {
                        if let Some(idp) = prefetch_iter.next() {
                            prefetch::<Self>(&mut bases[*idp]);
                        }
                    }
                    let mut a = &mut bases[*idx];
                    if !a.is_zero() {
                        if a.y.is_zero() {
                            a.infinity = true;
                        } else {
                            let x_sq = a.x.square();
                            let x_sq_3 = x_sq.double() + &x_sq; // numerator = 3x^2
                            scratch_space.push(x_sq_3 * &inversion_tmp); // 3x^2 * tmp
                            inversion_tmp *= &a.y.double(); // update tmp
                        }
                    }
                }

                inversion_tmp = inversion_tmp.inverse().unwrap(); // this is always in Fp*

                #[cfg(feature = "prefetch")]
                let mut prefetch_iter = index.iter().rev();
                #[cfg(feature = "prefetch")]
                let mut scratch_space_counter = (0..scratch_space.len()).rev();
                #[cfg(feature = "prefetch")]
                {
                    prefetch_iter.next();
                    scratch_space_counter.next();
                }

                for idx in index.iter().rev() {
                    #[cfg(feature = "prefetch")]
                    {
                        if let Some(idp) = prefetch_iter.next() {
                            prefetch::<Self>(&mut bases[*idp]);
                        }
                    }
                    let mut a = &mut bases[*idx];
                    if !a.is_zero() {
                        #[cfg(feature = "prefetch")]
                        {
                            if let Some(idp) = scratch_space_counter.next() {
                                prefetch::<P::BaseField>(&mut scratch_space[idp]);
                            }
                        }
                        let z = scratch_space.pop().unwrap();
                        let lambda = z * &inversion_tmp;
                        inversion_tmp *= &a.y.double(); // Remove the top layer of the denominator

                        // x3 = l^2 + 2x
                        let x3 = &(lambda.square() - &a.x.double());
                        // y3 = l*(x - x3) - y
                        a.y = lambda * &(a.x - x3) - &a.y;
                        a.x = *x3;
                    }
                }
            }

            // Consumes other and mutates self in place. Accepts index function
            #[inline]
            fn batch_add_in_place_with_edge_cases(
                bases: &mut [Self],
                other: &mut [Self],
                index: Vec<(usize, usize)>
            ) {
                let mut inversion_tmp = P::BaseField::one();

                #[cfg(feature = "prefetch")]
                let mut prefetch_iter = index.iter();
                #[cfg(feature = "prefetch")]
                {
                    prefetch_iter.next();
                }

                // let half = P::BaseField::from_repr(P::MODULUS_MINUS_ONE_DIV_TWO) + P::BaseField::one(); // (p + 1)/2 * 2 = 1
                // We run two loops over the data separated by an inversion
                for (idx, idy) in index.iter() {
                    #[cfg(feature = "prefetch")]
                    {
                        if let Some((idp_1, idp_2)) = prefetch_iter.next() {
                            prefetch::<Self>(&mut bases[*idp_1]);
                            prefetch::<Self>(&mut other[*idp_2]);
                        }
                    }
                    let (mut a, mut b) = (&mut bases[*idx], &mut other[*idy]);
                    if a.is_zero() || b.is_zero() {
                        continue;
                    } else if a.x == b.x {
                        // double.
                        // In our model, we consider self additions rare.
                        // So we consider it inconsequential to make them more expensive
                        // This costs 1 modular mul more than a standard squaring
                        if a.y == b.y {
                            let x_sq = b.x.square();
                            b.x -= &b.y; // x - y
                            a.x = b.y.double(); // denominator = 2y
                            a.y = x_sq.double() + &x_sq; // numerator = 3x^2
                            // b.y -= half * &a.y; // y - 3x^2/2
                            a.y *= &inversion_tmp; // 3x^2 * tmp
                            inversion_tmp *= &a.x; // update tmp
                        } else {
                            // No inversions take place if either operand is zero
                            a.infinity = true;
                            b.infinity = true;
                        }
                    } else {
                        a.x -= &b.x; // denominator = x1 - x2. We can recover x1 + x2 from this. Note this is never 0.
                        a.y -= &b.y; // numerator = y1 - y2
                        a.y *= &inversion_tmp; // (y1 - y2)*tmp
                        inversion_tmp *= &a.x // update tmp
                    }
                }

                inversion_tmp = inversion_tmp.inverse().unwrap(); // this is always in Fp*

                #[cfg(feature = "prefetch")]
                let mut prefetch_iter = index.iter().rev();
                #[cfg(feature = "prefetch")]
                {
                    prefetch_iter.next();
                }

                for (idx, idy) in index.iter().rev() {
                    #[cfg(feature = "prefetch")]
                    {
                        if let Some((idp_1, idp_2)) = prefetch_iter.next() {
                            prefetch::<Self>(&mut bases[*idp_1]);
                            prefetch::<Self>(&mut other[*idp_2]);
                        }
                    }
                    let (mut a, b) = (&mut bases[*idx], other[*idy]);
                    if a.is_zero() {
                        *a = b;
                    } else if !b.is_zero() {
                        let lambda = a.y * &inversion_tmp;
                        inversion_tmp *= &a.x; // Remove the top layer of the denominator

                        // x3 = l^2 + x1 + x2 or for squaring: 2y + l^2 + 2x - 2y = l^2 + 2x
                        a.x += &b.x.double();
                        a.x = lambda.square() - &a.x;
                        // y3 = l*(x2 - x3) - y2 or for squaring: 3x^2/2y(x - y - x3) - (y - 3x^2/2) = l*(x - x3) - y
                        a.y = lambda * &(b.x - &a.x) - &b.y;
                    }
                }
            }

            fn batch_scalar_mul_in_place<BigInt: BigInteger>(
                mut bases: &mut [Self],
                scalars: &mut [BigInt],
                w: usize,
            ) {
                // let no_op: u16 = 1 << w; // noop is encoded as half_window_size
                let now = std::time::Instant::now();
                let opcode_vectorised = Self::batch_wnaf_opcode_recoding::<BigInt>(scalars, w);
                println!("recoding: {:?}", now.elapsed().as_micros());
                let now = std::time::Instant::now();
                let tables = Self::batch_wnaf_tables(bases, w);
                println!("table generation: {:?}", now.elapsed().as_micros());

                // Set all points to 0;
                for p in bases.iter_mut() {
                    p.infinity = true;
                }

                let mut total: u128 = 0;
                let now = std::time::Instant::now();
                for opcode_row in opcode_vectorised.iter().rev() {
                    let index_double = opcode_row.iter()
                        .enumerate()
                        .filter(|x| x.1.is_some())
                        .map(|x| x.0)
                        .collect();

                    Self::batch_double_in_place_with_edge_cases(&mut bases, index_double);

                    let then = std::time::Instant::now();
                    // Copying to this vector might be really stupid...
                    let mut add_ops: Vec<Self> = tables.iter()
                        .zip(opcode_row)
                        .filter(|(_, op)| op.is_some() && op.unwrap() != 0)
                        .map(|(t, op)| {
                            let idx = op.unwrap();
                            if idx > 0 {
                                t[(idx as usize)/2].clone()
                            } else {
                                t[((-idx) as usize)/2].clone().neg()
                            }
                        })
                        .collect();

                    let dur = then.elapsed().as_micros();
                    // println!("allocate new points: {:?}", dur);
                    total += dur;


                    let index_add = opcode_row.iter()
                        .enumerate()
                        .filter(|(_, op)| op.is_some() && op.unwrap() != 0)
                        .map(|x| x.0)
                        .enumerate()
                        .map(|(x, y)| (y, x))
                        .collect();

                    Self::batch_add_in_place_with_edge_cases(
                        &mut bases, &mut add_ops[..], index_add);
                }

                println!("total - allocate new points: {:?}", total);
                println!("Scalar mul for {:?} points: {:?}", bases.len(), now.elapsed().as_micros());
            }

            // fn batch_scalar_mul_in_place_glv<BigInt: BigInteger>(
            //     w: usize,
            //     points: &mut Vec<Self>,
            //     scalars: &mut Vec<BigInt>,
            // ) {
            //     assert_eq!(points.len(), scalars.len());
            //     let batch_size = points.len();
            //     let mut k1 = scalars;
            //     // let (mut k1, mut k2) = Self::batch_glv_decomposition(scalars);
            //
            //     // let p2 = points.map(|p| p.glv_endomorphism());
            //     Self::batch_scalar_mul_in_place::<BigInt>(w, points, &mut k1);
            //     // Self::batch_scalar_mul_in_place(w, p2, k2);
            //     // Self::batch_add_in_place_with_edge_cases(points, p2);
            // }
            // }
        }

        impl<P: Parameters> GroupAffine<P> {
            pub fn new(x: P::BaseField, y: P::BaseField, infinity: bool) -> Self {
                Self {
                    x,
                    y,
                    infinity,
                    _params: PhantomData,
                }
            }

            pub fn scale_by_cofactor(&self) -> <Self as AffineCurve>::Projective {
                self.mul_bits(BitIterator::new(P::COFACTOR))
            }

            pub(crate) fn mul_bits<S: AsRef<[u64]>>(
                &self,
                bits: BitIterator<S>,
            ) -> <Self as AffineCurve>::Projective {
                let mut res = <Self as AffineCurve>::Projective::zero();
                for i in bits {
                    res.double_in_place();
                    if i {
                        res.add_assign_mixed(&self)
                    }
                }
                res
            }

            /// Attempts to construct an affine point given an x-coordinate. The
            /// point is not guaranteed to be in the prime order subgroup.
            ///
            /// If and only if `greatest` is set will the lexicographically
            /// largest y-coordinate be selected.
            #[allow(dead_code)]
            pub fn get_point_from_x(x: P::BaseField, greatest: bool) -> Option<Self> {
                // Compute x^3 + ax + b
                let x3b = P::add_b(&((x.square() * &x) + &P::mul_by_a(&x)));

                x3b.sqrt().map(|y| {
                    let negy = -y;

                    let y = if (y < negy) ^ greatest { y } else { negy };
                    Self::new(x, y, false)
                })
            }

            /// Checks that the current point is on the elliptic curve.
            pub fn is_on_curve(&self) -> bool {
                if self.is_zero() {
                    true
                } else {
                    // Check that the point is on the curve
                    let y2 = self.y.square();
                    let x3b = P::add_b(&((self.x.square() * &self.x) + &P::mul_by_a(&self.x)));
                    y2 == x3b
                }
            }

            /// Checks that the current point is in the prime order subgroup given
            /// the point on the curve.
            pub fn is_in_correct_subgroup_assuming_on_curve(&self) -> bool {
                self.mul_bits(BitIterator::new(P::ScalarField::characteristic()))
                    .is_zero()
            }
        }


        impl<P: Parameters> Display for GroupAffine<P> {
            fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
                if self.infinity {
                    write!(f, "GroupAffine(Infinity)")
                } else {
                    write!(f, "GroupAffine(x={}, y={})", self.x, self.y)
                }
            }
        }

        impl<P: Parameters> Zero for GroupAffine<P> {
            fn zero() -> Self {
                Self::new(P::BaseField::zero(), P::BaseField::one(), true)
            }

            fn is_zero(&self) -> bool {
                self.infinity
            }
        }

        impl<P: Parameters> Add<Self> for GroupAffine<P> {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                let mut copy = self;
                copy += &other;
                copy
            }
        }

        impl<'a, P: Parameters> AddAssign<&'a Self> for GroupAffine<P> {
            fn add_assign(&mut self, other: &'a Self) {
                let mut s_proj = <Self as AffineCurve>::Projective::from(*self);
                s_proj.add_assign_mixed(other);
                *self = s_proj.into();
            }
        }

        impl<P: Parameters> Neg for GroupAffine<P> {
            type Output = Self;

            fn neg(self) -> Self {
                if !self.is_zero() {
                    Self::new(self.x, -self.y, false)
                } else {
                    self
                }
            }
        }

        impl<P: Parameters> ToBytes for GroupAffine<P> {
            #[inline]
            fn write<W: Write>(&self, mut writer: W) -> IoResult<()> {
                self.x.write(&mut writer)?;
                self.y.write(&mut writer)?;
                self.infinity.write(writer)
            }
        }

        impl<P: Parameters> FromBytes for GroupAffine<P> {
            #[inline]
            fn read<R: Read>(mut reader: R) -> IoResult<Self> {
                let x = P::BaseField::read(&mut reader)?;
                let y = P::BaseField::read(&mut reader)?;
                let infinity = bool::read(reader)?;
                Ok(Self::new(x, y, infinity))
            }
        }

        impl<P: Parameters> Default for GroupAffine<P> {
            #[inline]
            fn default() -> Self {
                Self::zero()
            }
        }

        #[cfg(feature = "prefetch")]
        #[inline_always]
        pub fn prefetch<T>(p: *const T) {
            unsafe {  core::arch::x86_64::_mm_prefetch(p as *const i8,  core::arch::x86_64::_MM_HINT_T0) }
        }

        impl_sw_curve_serializer!(Parameters);
    }
}
