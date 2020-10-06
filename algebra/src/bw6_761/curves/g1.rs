use crate::{
    biginteger::{BigInteger384, BigInteger768}, //, BigInteger1536},
    bw6_761::{Fq, Fr},
    curves::{
        models::{ModelParameters, SWModelParameters},
        short_weierstrass_jacobian::{GroupAffine, GroupProjective},
        GLVParameters,
    },
    field_new,
    fields::PrimeField,
    impl_scalar_mul_kernel_glv,
};

pub type G1Affine = GroupAffine<Parameters>;
pub type G1Projective = GroupProjective<Parameters>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Parameters;

impl ModelParameters for Parameters {
    type BaseField = Fq;
    type ScalarField = Fr;
}

impl_scalar_mul_kernel_glv!(bw6_761, "bw6_761", g1, G1Projective);

/// The parameters can be obtained from
/// Optimized and secure pairing-friendly elliptic
/// curves suitable for one layer proof composition
/// Youssef El Housni and Aurore Guillevic, 2020.
/// https://eprint.iacr.org/2020/351.pdf
/// and the precomputed parameters Qi, Bi, *_IS_NEG can be obtained from
/// scripts/glv_lattice_basis
impl GLVParameters for Parameters {
    type WideBigInt = BigInteger768;

    /// phi((x, y)) = (\omega x, y)
    /// \omega = 0x531dc16c6ecd27aa846c61024e4cca6c1f31e53bd9603c2d17be416c5e44
    /// 26ee4a737f73b6f952ab5e57926fa701848e0a235a0a398300c65759fc4518315
    /// 1f2f082d4dcb5e37cb6290012d96f8819c547ba8a4000002f962140000000002a
    const OMEGA: Fq = field_new!(
        Fq,
        BigInteger768([
            7467050525960156664,
            11327349735975181567,
            4886471689715601876,
            825788856423438757,
            532349992164519008,
            5190235139112556877,
            10134108925459365126,
            2188880696701890397,
            14832254987849135908,
            2933451070611009188,
            11385631952165834796,
            64130670718986244
        ])
    );

    /// lambda in Z s.t. phi(P) = lambda*P for all P
    /// \lambda = 0x9b3af05dd14f6ec619aaf7d34594aabc5ed1347970dec00452217cc900000008508c00000000001
    const LAMBDA: Self::ScalarField = field_new!(
        Fr,
        (BigInteger384([
            15766275933608376691,
            15635974902606112666,
            1934946774703877852,
            18129354943882397960,
            15437979634065614942,
            101285514078273488
        ]))
    );
    /// |round(B1 * R / n)|
    const Q2: <Self::ScalarField as PrimeField>::BigInt = BigInteger384([
        14430678704534329733,
        14479735877321354361,
        6958676793196883088,
        21,
        0,
        0,
    ]);
    const B1: <Self::ScalarField as PrimeField>::BigInt = BigInteger384([
        9586122913090633729,
        9963140610363752448,
        2588746559005780992,
        0,
        0,
        0,
    ]);
    const B1_IS_NEG: bool = true;
    /// |round(B2 * R / n)|
    const Q1: <Self::ScalarField as PrimeField>::BigInt = BigInteger384([
        11941976086484053770,
        4826578625773784813,
        2319558931065627696,
        7,
        0,
        0,
    ]);
    const B2: <Self::ScalarField as PrimeField>::BigInt = BigInteger384([
        6390748608727089153,
        3321046870121250816,
        862915519668593664,
        0,
        0,
        0,
    ]);
    const R_BITS: u32 = 384;
}

impl SWModelParameters for Parameters {
    /// COEFF_A = 0
    #[rustfmt::skip]

    const COEFF_A: Fq = field_new!(Fq, BigInteger768([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));

    /// COEFF_B = -1
    #[rustfmt::skip]
    const COEFF_B: Fq = field_new!(Fq, BigInteger768([
        0xf29a000000007ab6,
        0x8c391832e000739b,
        0x77738a6b6870f959,
        0xbe36179047832b03,
        0x84f3089e56574722,
        0xc5a3614ac0b1d984,
        0x5c81153f4906e9fe,
        0x4d28be3a9f55c815,
        0xd72c1d6f77d5f5c5,
        0x73a18e069ac04458,
        0xf9dfaa846595555f,
        0xd0f0a60a5be58c,
    ]));

    /// COFACTOR =
    /// 26642435879335816683987677701488073867751118270052650655942102502312977592501693353047140953112195348280268661194876
    #[rustfmt::skip]
    const COFACTOR: &'static [u64] = &[
        0x3de580000000007c,
        0x832ba4061000003b,
        0xc61c554757551c0c,
        0xc856a0853c9db94c,
        0x2c77d5ac34cb12ef,
        0xad1972339049ce76,
    ];

    /// COFACTOR^(-1) mod r =
    /// 91141326767669940707819291241958318717982251277713150053234367522357946997763584490607453720072232540829942217804
    #[rustfmt::skip]
    const COFACTOR_INV: Fr = field_new!(Fr, BigInteger384([
        489703175600125849,
        3883341943836920852,
        1678256062427438196,
        5848789333018172718,
        7127967896440782320,
        71512347676739162,
    ]));

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const AFFINE_GENERATOR_COEFFS: (Self::BaseField, Self::BaseField) =
        (G1_GENERATOR_X, G1_GENERATOR_Y);
    #[inline(always)]
    fn mul_by_a(_elem: &Self::BaseField) -> Self::BaseField {
        use crate::Zero;
        Self::BaseField::zero()
    }

    #[inline(always)]
    fn has_glv() -> bool {
        true
    }

    #[inline(always)]
    fn glv_endomorphism_in_place(elem: &mut Self::BaseField) {
        *elem *= &<Self as GLVParameters>::OMEGA;
    }

    #[inline]
    fn glv_scalar_decomposition(
        k: <Self::ScalarField as PrimeField>::BigInt,
    ) -> (
        (bool, <Self::ScalarField as PrimeField>::BigInt),
        (bool, <Self::ScalarField as PrimeField>::BigInt),
    ) {
        <Self as GLVParameters>::glv_scalar_decomposition_inner(k)
    }

    fn scalar_mul_kernel(
        ctx: &Context,
        grid: impl Into<Grid>,
        block: impl Into<Block>,
        table: *const G1Projective,
        exps: *const u8,
        out: *mut G1Projective,
        n: isize,
    ) -> error::Result<()> {
        scalar_mul(ctx, grid, block, (table, exps, out, n))?;
        Ok(())
    }
}

/// G1_GENERATOR_X =
/// 6238772257594679368032145693622812838779005809760824733138787810501188623461307351759238099287535516224314149266511977132140828635950940021790489507611754366317801811090811367945064510304504157188661901055903167026722666149426237
#[rustfmt::skip]
pub const G1_GENERATOR_X: Fq = field_new!(Fq, BigInteger768([
    0xd6e42d7614c2d770,
    0x4bb886eddbc3fc21,
    0x64648b044098b4d2,
    0x1a585c895a422985,
    0xf1a9ac17cf8685c9,
    0x352785830727aea5,
    0xddf8cb12306266fe,
    0x6913b4bfbc9e949a,
    0x3a4b78d67ba5f6ab,
    0x0f481c06a8d02a04,
    0x91d4e7365c43edac,
    0xf4d17cd48beca5,
]));

/// G1_GENERATOR_Y =
/// 2101735126520897423911504562215834951148127555913367997162789335052900271653517958562461315794228241561913734371411178226936527683203879553093934185950470971848972085321797958124416462268292467002957525517188485984766314758624099
#[rustfmt::skip]
pub const G1_GENERATOR_Y: Fq = field_new!(Fq, BigInteger768([
    0x97e805c4bd16411f,
    0x870d844e1ee6dd08,
    0x1eba7a37cb9eab4d,
    0xd544c4df10b9889a,
    0x8fe37f21a33897be,
    0xe9bf99a43a0885d2,
    0xd7ee0c9e273de139,
    0xaa6a9ec7a38dd791,
    0x8f95d3fcf765da8e,
    0x42326e7db7357c99,
    0xe217e407e218695f,
    0x9d1eb23b7cf684,
]));
