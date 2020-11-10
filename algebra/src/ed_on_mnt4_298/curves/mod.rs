use crate::ed_on_mnt4_298::{fq::Fq, fr::Fr};
use algebra_core::{
    biginteger::BigInteger320,
    curves::{
        models::{ModelParameters, MontgomeryModelParameters, TEModelParameters},
        twisted_edwards_extended::{GroupAffine, GroupProjective},
    },
    field_new, impl_scalar_mul_kernel, impl_scalar_mul_parameters,
};

#[cfg(test)]
mod tests;

pub type EdwardsAffine = GroupAffine<EdwardsParameters>;
pub type EdwardsProjective = GroupProjective<EdwardsParameters>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct EdwardsParameters;

impl ModelParameters for EdwardsParameters {
    type BaseField = Fq;
    type ScalarField = Fr;
}

// Many parameters need to be written down in the Montgomery residue form,
// discussed below. Some useful numbers:
// R for Fq: 223364648326281414938801705359223029554923725549792420683051274872200260503540791531766876
// R for Fr: 104384076783966083500464392945960916666734135485183910065100558776489954102951241798239545

impl_scalar_mul_kernel!(ed_on_mnt4_298, "ed_on_mnt4_298", proj, EdwardsProjective);

impl TEModelParameters for EdwardsParameters {
    /// COEFF_A = -1
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., -1 * R for Fq
    ///     = 252557637842979910814547544293825421990201153003031094870216460866964386803867699028196261
    #[rustfmt::skip]
    const COEFF_A: Fq = field_new!(Fq, BigInteger320([
        17882590928154426277u64,
        6901912683734848330u64,
        364575608937879866u64,
        8740893163049517815u64,
        2181130330288u64,
    ]));

    /// COEFF_D = 4212
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., 4212 * R for Fq
    ///     = 389461279836940033614665658623660232171971995346409183754923941118154161474636585314923000
    #[rustfmt::skip]
    const COEFF_D: Fq = field_new!(Fq, BigInteger320([
        8040159930071495160u64,
        16503302848883893212u64,
        4541498709509651666u64,
        11429056610118256373u64,
        3363453258354u64,
    ]));

    /// COFACTOR = 4
    const COFACTOR: &'static [u64] = &[4];

    /// COFACTOR_INV (mod r) =
    /// 29745142885578832859584328103315528221570304936126890280067991221921526670592508030983158
    /// Needs to be in the Montgomery residue form in Fr
    /// I.e., 29745142885578832859584328103315528221570304936126890280067991221921526670592508030983158 * R for Fr
    ///     = 55841162081570353734700426339805757388253838807422867796343130916044015196330318480543044
    #[rustfmt::skip]
    const COFACTOR_INV: Fr = field_new!(Fr, BigInteger320([
        6539529304383425860u64,
        7567022062893857598u64,
        17399624368177871129u64,
        14575354999847441509u64,
        482253688048u64,
    ]));

    /// Generated randomly
    const AFFINE_GENERATOR_COEFFS: (Self::BaseField, Self::BaseField) = (GENERATOR_X, GENERATOR_Y);

    type MontgomeryModelParameters = EdwardsParameters;

    /// Multiplication by `a` is just negation.
    #[inline(always)]
    fn mul_by_a(elem: &Self::BaseField) -> Self::BaseField {
        -*elem
    }

    impl_scalar_mul_parameters!(EdwardsProjective);
}

impl MontgomeryModelParameters for EdwardsParameters {
    /// COEFF_A = 203563247015667910991582090642011229452721346107806307863040223071914240315202967004285204
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., 203563247015667910991582090642011229452721346107806307863040223071914240315202967004285204 * R for Fq
    ///     = 184958108588233737086787169006685138672129232027042887479256778022373554352621152610883766
    #[rustfmt::skip]
    const COEFF_A: Fq = field_new!(Fq, BigInteger320([
        13866101745789245622u64,
        14126297534943667090u64,
        11307578615387704385u64,
        8263080598809044705u64,
        1597329401399u64,
    ]));
    /// COEFF_B = 272359039153593414761767159011037222092403532445017207690227512667250406992205523555677931
    /// Needs to be in the Montgomery residue form in Fq
    //  I.e., 272359039153593414761767159011037222092403532445017207690227512667250406992205523555677931 * R for Fq
    //      = 320157167097726084542307919580965705308273073979019302261176143711555219255114245445508756
    #[rustfmt::skip]
    const COEFF_B: Fq = field_new!(Fq, BigInteger320([
        3452336036810055316u64,
        18124271906235581187u64,
        7868316676197606962u64,
        9218705727289990924u64,
        2764931259177u64,
    ]));

    type TEModelParameters = EdwardsParameters;
}

/// GENERATOR_X =
/// 282406820114868156776872298252698015906762052916420164316497572033519876761239463633892227
/// Needs to be in the Montgomery residue form in Fq
/// I.e., 282406820114868156776872298252698015906762052916420164316497572033519876761239463633892227 * R for Fq
///     = 6917556742108450905978293995070573074174231920036503115659104908111915200040057661385715
#[rustfmt::skip]
const GENERATOR_X: Fq = field_new!(Fq, BigInteger320([
    797921980254612467u64,
    14323677897559322103u64,
    16879595040064082265u64,
    5138786402348661261u64,
    59741186014u64,
]));

/// GENERATOR_Y =
/// 452667754940241021433619311795265643711152068500301853535337412655162600774122192283142703
/// Needs to be in the Montgomery residue form in Fq
/// I.e., 452667754940241021433619311795265643711152068500301853535337412655162600774122192283142703 * R for Fq
///     = 411219337323952690830344109182130393590959634960952808951091963301565250764467583592890490
#[rustfmt::skip]
const GENERATOR_Y: Fq = field_new!(Fq, BigInteger320([
    16522567711648317562u64,
    4273808507945498262u64,
    17459848913470201097u64,
    16519670308098023011u64,
    3551359510243u64,
]));
