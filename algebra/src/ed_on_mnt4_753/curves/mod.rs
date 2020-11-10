use crate::ed_on_mnt4_753::{fq::Fq, fr::Fr};
use algebra_core::{
    biginteger::BigInteger768,
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
// R for Fq: 11407975440035778516953587871987109648531742722982233186120790377529569367095961954159305159259556262528904776132787438725571821295685691762729353555475679813615501328617736020411951837995932262333059670631633855898874183380802
// R for Fr: 933352698056040166367534174176950366489065242993745918174914647273231163953185260894581718311971532174387033963715296372791285468903747270837716556902938133611910788060028435531754797383796835009316018259656953442114538695438

impl_scalar_mul_kernel!(ed_on_mnt4_753, "ed_on_mnt4_753", proj, EdwardsProjective);
impl TEModelParameters for EdwardsParameters {
    /// COEFF_A = -1
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., -1 * R for Fq
    ///     = 30490515527883174885390626919253527479638967196971715885662712543495783445475144818899588604530782658889166195755671038597601236195908163306966888299320716352105914996732328421058466299850466207278876048428274308321910292779199
    #[rustfmt::skip]
    const COEFF_A: Fq = field_new!(Fq, BigInteger768([
        2265581976117350591u64,
        18442012872391748519u64,
        3807704300793525789u64,
        12280644139289115082u64,
        10655371227771325282u64,
        1346491763263331896u64,
        7477357615964975877u64,
        12570239403004322603u64,
        2180620924574446161u64,
        12129628062772479841u64,
        8853285699251153944u64,
        362282887012814u64,
    ]));

    /// COEFF_D = 317690
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., 317690 * R for Fq
    ///     = 22147310944926701613095824060993292411108298129020138512675871596899298127988454048404371067902679066037332245471578749765607461010546427833106841035248048771826362113332201923280907352099197626899000000763383579702914883060881
    #[rustfmt::skip]
    const COEFF_D: Fq = field_new!(Fq, BigInteger768([
        17599538631181665425u64,
        541385733032329781u64,
        10984951882154109942u64,
        6745898816867096302u64,
        8606788232777167026u64,
        17697068798460151905u64,
        7726746940317276687u64,
        16708084840201435716u64,
        10141323747759975110u64,
        6527904409415579649u64,
        18367733563217133340u64,
        263150412834478u64,
    ]));

    /// COFACTOR = 8
    const COFACTOR: &'static [u64] = &[8];

    /// COFACTOR_INV (mod r) =
    /// 4582647449616135528381398492791944685893671397494963179726320631987147963874964803303316505414568319530101512550297775574042810022553679071007001162683923594233560231270043634777390699589793776691858866199511300853468155295505
    /// Needs to be in the Montgomery residue form in Fr
    /// I.e., COFACTOR_INV * R for Fr
    ///     = 1425996930004472314619198483998388706066467840372779148265098797191196170886995244269913144907444532816113097116978062210611142118628305286285286330379702579339648914584658878663580978127201397716695606910888919424112361707074
    #[rustfmt::skip]
    const COFACTOR_INV: Fr = field_new!(Fr, BigInteger768([
        18349096995079034434u64,
        12232096963923221952u64,
        10313403112747203584u64,
        7266093872567585103u64,
        9102010985112647012u64,
        11539789563873699451u64,
        5062476400815403157u64,
        3112383580531982668u64,
        9803941911066678468u64,
        11670110706913295633u64,
        5956199581925454898u64,
        16943442107464u64,
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
    /// COEFF_A = 40212480635445336270302172549278415015971955924352275480357619589919378421241453024646804979794897776496091377551124233752850182852486874251193367187677349266115879541798515219680194853352256809837126277708211496794264654247419
    /// Needs to be in the Montgomery residue form in Fq
    /// I.e., COEFF_A * R for Fq
    ///     = 30548714567617468394128273134168309733495884043859854416819409495212098575586848195824755026287273763308450716502830186864520759966983420083939453225231731740328282532297868204762840705631404761799649264638732114864775402781225
    #[rustfmt::skip]
    const COEFF_A: Fq = field_new!(Fq, BigInteger768([
        4717325759818398249u64,
        9984799932299155706u64,
        1320735555238925850u64,
        17027346723122076572u64,
        2632519042034336982u64,
        15439824589583270152u64,
        8351651296737343223u64,
        11351622927160584696u64,
        3108522085485690820u64,
        6958456540352275598u64,
        16034686916204205245u64,
        362974397660347u64,
    ]));
    /// COEFF_B = 1686010332473617132042042241962222112198753995601673591425883331105974391329653748412088783995441144921979594337334243570322874639106980818502874667119046899605536783551549221790223284494141659774809441351696667426519821912580
    /// Needs to be in the Montgomery residue form in Fq
    //  I.e., COEFF_B * R for Fq
    //      = 30432316488148881376652980704338745225782050350083577354506015591779468315363441441974422182774291554469881675008511890330681712424832906529994323373409700963883547461166788637354091894069527652758102832217816501779045182777173
    #[rustfmt::skip]
    const COEFF_B: Fq = field_new!(Fq, BigInteger768([
        18260582266125854549u64,
        8452481738774789715u64,
        6294673046348125729u64,
        7533941555456153592u64,
        231479339798761966u64,
        5699903010652945257u64,
        6603063935192608530u64,
        13788855878848060510u64,
        1252719763663201502u64,
        17300799585192684084u64,
        1671884482298102643u64,
        361591376365281u64,
    ]));

    type TEModelParameters = EdwardsParameters;
}

/// GENERATOR_X =
/// 41126137307536311801428235632419266329480236393691483739251051053325519918069469184425962602019877935619960143044210127218431046103600632347238890180171944971817510488009355627861577881883236134824745174469522277738875418206826
/// Needs to be in the Montgomery residue form in Fq
/// I.e., GENERATOR_X * R for Fq
///     = 17458296603084005843875564204476809882690765950143935590811069375604430769391871724158635621148427226413334766092842987247361751645959801401160673759590522483750685475882467271029344718076741595831312033991612062403782328664175
#[rustfmt::skip]
const GENERATOR_X: Fq = field_new!(Fq, BigInteger768([
    13391543849638641775u64,
    1472718285337442467u64,
    1704796371472020786u64,
    1309193942690519845u64,
    11187264906425773918u64,
    11963130799714018220u64,
    10821241385017749516u64,
    4661882526685671286u64,
    8328914571224024668u64,
    17202160931109725769u64,
    4708938015393622850u64,
    207436377712515u64,
]));

/// GENERATOR_Y =
/// 18249602579663240810999977712212098844157230095713722119136881953011435881503578209163288529034825612841855863913294174196656077002578342108932925693640046298989762289691399012056048139253937882385653600831389370198228562812681
/// Needs to be in the Montgomery residue form in Fq
/// I.e., GENERATOR_Y * R for Fq
///     = 9017791529346511307345374145466037779022974291216533108328228023141994468888559894991603799439817566592668010556604996318161436165296215592281656017954181737938978992370627048110847574165717052386876801764386102664064737203581
#[rustfmt::skip]
const GENERATOR_Y: Fq = field_new!(Fq, BigInteger768([
    16764059510974436733u64,
    10694630934032454957u64,
    15899992550979352399u64,
    17663221529566141065u64,
    3780246386961240559u64,
    6062186621379836072u64,
    11042203340250178810u64,
    1263100291243127914u64,
    14407501552666806512u64,
    13385165116432280059u64,
    11978187531853934313u64,
    107147796394053u64,
]));
