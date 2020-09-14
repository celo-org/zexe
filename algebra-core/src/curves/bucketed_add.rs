use crate::{
    curves::{BatchGroupArithmeticSlice, BATCH_SIZE},
    AffineCurve, Vec,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "std")]
use {core::cmp::Ordering, voracious_radix_sort::*};

#[cfg(not(feature = "std"))]
use crate::log2;

#[derive(Copy, Clone, Debug)]
pub struct BucketPosition {
    pub bucket: u32,
    pub position: u32,
}

#[cfg(feature = "std")]
impl PartialOrd for BucketPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.bucket.partial_cmp(&other.bucket)
    }
}

#[cfg(feature = "std")]
impl Radixable<u32> for BucketPosition {
    type Key = u32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.bucket
    }
}

impl PartialEq for BucketPosition {
    fn eq(&self, other: &Self) -> bool {
        self.bucket == other.bucket
    }
}

/// This does the same thing as batch_bucketed_add, but parallelises
/// jobs over fixed number of threads.
#[cfg(feature = "std")]
pub fn batch_bucketed_add_multiple<C: AffineCurve>(
    buckets_vec: &[usize],
    elems: &[C],
    bucket_positions_vec: &mut [Vec<BucketPosition>],
    parallelism: usize,
) -> Vec<Vec<C>> {
    assert_eq!(buckets_vec.len(), bucket_positions_vec.len());

    let zero = C::zero();
    let mut res: Vec<Vec<C>> = buckets_vec
        .iter()
        .map(|&buckets| vec![zero; buckets])
        .collect();

    if cfg!(feature = "parallel") && elems.len() >= BATCH_SIZE * (1 << 4) {
        let mut split_pointers_vec = buckets_vec
            .par_iter()
            .zip(bucket_positions_vec.par_iter_mut())
            .zip(res.par_iter_mut())
            .map(|((&buckets, bucket_positions_v), res_single)| {
                assert_eq!(elems.len(), bucket_positions_v.len());

                let mut res_ref = &mut res_single[..];
                let mut bucket_positions = &mut bucket_positions_v[..];

                let _now = timer!();
                dlsd_radixsort(bucket_positions, 8);
                timer_println!(_now, "radixsort");

                let n_groups = parallelism;
                let group_size = ((buckets - 1) / n_groups + 1) as u32;

                let mut pointer = 0;
                let mut group_count = 0u32;
                let mut buckets_left = buckets;
                let mut prev_bucket = 0;

                let mut split_pointers = Vec::with_capacity(n_groups);

                while pointer < bucket_positions.len() {
                    let current_bucket = bucket_positions[pointer].bucket;
                    if current_bucket > (group_count + 1) * group_size
                        && current_bucket < buckets as u32
                    {
                        // Since group_size is > 0, current_bucket > 0
                        let buckets_consumed = (current_bucket - prev_bucket) as usize;

                        let (bottom, top) = bucket_positions.split_at_mut(pointer);
                        let (bottom_res, top_res) = res_ref.split_at_mut(buckets_consumed);

                        split_pointers.push((buckets_consumed, prev_bucket, bottom, bottom_res));

                        bucket_positions = top;
                        res_ref = top_res;

                        buckets_left -= buckets_consumed;
                        prev_bucket = current_bucket;
                        group_count += 1;
                        pointer = 0;
                    }
                    // We have to check every assignment as we need to ascertain the boundaries.
                    // A faster method is binary search on sorted array. But this method is 0.1% of total runtime.
                    pointer += 1;
                }
                // push remaining assignments
                split_pointers.push((buckets_left, prev_bucket, bucket_positions, res_ref));
                split_pointers
            })
            .collect::<Vec<_>>();
        let _now = timer!();
        rayon::scope(|s| {
            for split_pointers in split_pointers_vec.iter_mut() {
                for (buckets, offset, bp, r) in split_pointers.iter_mut() {
                    s.spawn(move |_| batch_bucketed_add_inner(*buckets, *offset, elems, bp, r));
                }
            }
        });
        timer_println!(_now, "bucket add multiple");
    } else {
        for ((&buckets, bucket_positions), res_single) in buckets_vec
            .iter()
            .zip(bucket_positions_vec.iter_mut())
            .zip(res.iter_mut())
        {
            assert_eq!(elems.len(), bucket_positions.len());
            batch_bucketed_add_inner(
                buckets,
                0,
                elems,
                &mut bucket_positions[..],
                &mut res_single[..],
            );
        }
    }
    res
}

#[cfg(feature = "std")]
pub fn batch_bucketed_add<C: AffineCurve>(
    buckets: usize,
    elems: &[C],
    bucket_positions: &mut [BucketPosition],
) -> Vec<C> {
    assert_eq!(elems.len(), bucket_positions.len());

    let _now = timer!();
    dlsd_radixsort(bucket_positions, 8);
    timer_println!(_now, "radixsort");

    let zero = C::zero();
    let mut res = vec![zero; buckets];
    batch_bucketed_add_inner(buckets, 0, elems, bucket_positions, &mut res[..]);
    res
}

#[inline]
#[cfg(feature = "std")]
#[inline]
pub fn batch_bucketed_add_inner<C: AffineCurve>(
    buckets: usize,
    offset: u32,
    elems: &[C],
    bucket_positions: &mut [BucketPosition],
    res: &mut [C],
) {
    assert!(elems.len() > 0);

    let mut len = bucket_positions.len();
    let mut all_ones = true;
    let mut new_len = 0; // len counter
    let mut glob = 0; // global counters
    let mut loc = 1; // local counter
    let mut batch = 0; // batch counter
    let mut instr = Vec::<(u32, u32)>::with_capacity(BATCH_SIZE);
    let mut new_elems = Vec::<C>::with_capacity(elems.len() * 3 / 8);

    let mut scratch_space = Vec::<Option<C>>::with_capacity(BATCH_SIZE / 2);

    let _now = timer!();
    // In the first loop, we copy the results of the first in place addition tree
    // to a local vector, new_elems
    // Subsequently, we perform all the operations in place
    while glob < len {
        let current_bucket = bucket_positions[glob].bucket;
        while glob + 1 < len && bucket_positions[glob + 1].bucket == current_bucket {
            glob += 1;
            loc += 1;
        }
        if current_bucket >= buckets as u32 + offset {
            loc = 1;
        } else if loc > 1 {
            // all ones is false if next len is not 1
            if loc > 2 {
                all_ones = false;
            }
            let is_odd = loc % 2 == 1;
            let half = loc / 2;
            for i in 0..half {
                instr.push((
                    bucket_positions[glob - (loc - 1) + 2 * i].position,
                    bucket_positions[glob - (loc - 1) + 2 * i + 1].position,
                ));
                bucket_positions[new_len + i] = BucketPosition {
                    bucket: current_bucket - offset,
                    position: (new_len + i) as u32,
                };
            }
            if is_odd {
                instr.push((bucket_positions[glob].position, !0u32));
                bucket_positions[new_len + half] = BucketPosition {
                    bucket: current_bucket - offset,
                    position: (new_len + half) as u32,
                };
            }
            // Reset the local_counter and update state
            new_len += half + (loc % 2);
            batch += half;
            loc = 1;

            if batch >= BATCH_SIZE / 2 {
                // We need instructions for copying data in the case
                // of noops. We encode noops/copies as !0u32
                elems[..].batch_add_write(&instr[..], &mut new_elems, &mut scratch_space);

                instr.clear();
                batch = 0;
            }
        } else {
            instr.push((bucket_positions[glob].position, !0u32));
            bucket_positions[new_len] = BucketPosition {
                bucket: current_bucket - offset,
                position: new_len as u32,
            };
            new_len += 1;
        }
        glob += 1;
    }
    if instr.len() > 0 {
        elems[..].batch_add_write(&instr[..], &mut new_elems, &mut scratch_space);
        instr.clear();
    }
    glob = 0;
    batch = 0;
    loc = 1;
    len = new_len;
    new_len = 0;

    while !all_ones {
        all_ones = true;
        while glob < len {
            let current_bucket = bucket_positions[glob].bucket;
            while glob + 1 < len && bucket_positions[glob + 1].bucket == current_bucket {
                glob += 1;
                loc += 1;
            }
            if current_bucket >= buckets as u32 {
                loc = 1;
            } else if loc > 1 {
                // all ones is false if next len is not 1
                if loc != 2 {
                    all_ones = false;
                }
                let is_odd = loc % 2 == 1;
                let half = loc / 2;
                for i in 0..half {
                    instr.push((
                        bucket_positions[glob - (loc - 1) + 2 * i].position,
                        bucket_positions[glob - (loc - 1) + 2 * i + 1].position,
                    ));
                    bucket_positions[new_len + i] = bucket_positions[glob - (loc - 1) + 2 * i];
                }
                if is_odd {
                    bucket_positions[new_len + half] = bucket_positions[glob];
                }
                // Reset the local_counter and update state
                new_len += half + (loc % 2);
                batch += half;
                loc = 1;

                if batch >= BATCH_SIZE / 2 {
                    &mut new_elems[..].batch_add_in_place_same_slice(&instr[..]);
                    instr.clear();
                    batch = 0;
                }
            } else {
                bucket_positions[new_len] = bucket_positions[glob];
                new_len += 1;
            }
            glob += 1;
        }
        if instr.len() > 0 {
            &mut new_elems[..].batch_add_in_place_same_slice(&instr[..]);
            instr.clear();
        }
        glob = 0;
        batch = 0;
        loc = 1;
        len = new_len;
        new_len = 0;
    }
    timer_println!(_now, "addition tree");

    let _now = timer!();
    for bp in bucket_positions[..len].iter() {
        res[bp.bucket as usize] = new_elems[bp.position as usize];
    }
    timer_println!(_now, "reassign");
}

#[cfg(not(feature = "std"))]
pub fn batch_bucketed_add<C: AffineCurve>(
    buckets: usize,
    elems: &[C],
    bucket_assign: &[BucketPosition],
) -> Vec<C> {
    assert_eq!(elems.len(), bucket_assign.len());

    let zero = C::zero();
    let mut res = vec![zero; buckets];

    batch_bucketed_add_inner(buckets, elems, bucket_assign, &mut res);
    res
}

#[cfg(not(feature = "std"))]
pub fn batch_bucketed_add_multiple<C: AffineCurve>(
    buckets_vec: &[usize],
    elems: &[C],
    bucket_positions_vec: &mut [Vec<BucketPosition>],
    _parallelism: usize,
) -> Vec<Vec<C>> {
    assert_eq!(buckets_vec.len(), bucket_positions_vec.len());

    let zero = C::zero();
    let mut res: Vec<Vec<C>> = buckets_vec
        .iter()
        .map(|&buckets| vec![zero; buckets])
        .collect();

    for ((&buckets, bucket_positions), res_single) in buckets_vec
        .iter()
        .zip(bucket_positions_vec.iter_mut())
        .zip(res.iter_mut())
    {
        batch_bucketed_add_inner(
            buckets,
            elems,
            &mut bucket_positions[..],
            &mut res_single[..],
        );
    }
    res
}
#[cfg(not(feature = "std"))]
pub fn batch_bucketed_add_inner<C: AffineCurve>(
    buckets: usize,
    elems: &[C],
    bucket_assign: &[BucketPosition],
    res: &mut [C],
) {
    let mut elems = elems.to_vec();
    let num_split = 2i32.pow(log2(buckets) / 2 + 2) as usize;
    let split_size = (buckets - 1) / num_split + 1;
    let ratio = elems.len() / buckets * 2;
    // Get the inverted index for the positions assigning to each bucket
    let mut bucket_split = vec![vec![]; num_split];
    let mut index = vec![Vec::with_capacity(ratio); buckets];

    for bucket_pos in bucket_assign.iter() {
        let (bucket, position) = (bucket_pos.bucket as usize, bucket_pos.position as usize);
        // Check the bucket assignment is valid
        if bucket < buckets {
            // index[bucket].push(position);
            bucket_split[bucket / split_size].push((bucket, position));
        }
    }

    for split in bucket_split {
        for (bucket, position) in split {
            index[bucket].push(position as u32);
        }
    }

    // Instructions for indexes for the in place addition tree
    let mut instr: Vec<Vec<(u32, u32)>> = vec![];
    // Find the maximum depth of the addition tree
    let max_depth = index.iter()
        // log_2
        .map(|x| log2(x.len()))
        .max().unwrap();

    // Generate in-place addition instructions that implement the addition tree
    // for each bucket from the leaves to the root
    for i in 0..max_depth {
        let mut instr_row = Vec::<(u32, u32)>::with_capacity(buckets);
        for to_add in index.iter_mut() {
            if to_add.len() > 1 << (max_depth - i - 1) {
                let mut new_to_add = vec![];
                for j in 0..(to_add.len() / 2) {
                    new_to_add.push(to_add[2 * j]);
                    instr_row.push((to_add[2 * j], to_add[2 * j + 1]));
                }
                if to_add.len() % 2 == 1 {
                    new_to_add.push(*to_add.last().unwrap());
                }
                *to_add = new_to_add;
            }
        }
        instr.push(instr_row);
    }

    for instr_row in instr.iter() {
        for instr in C::get_chunked_instr::<(u32, u32)>(&instr_row[..], BATCH_SIZE).iter() {
            elems[..].batch_add_in_place_same_slice(&instr[..]);
        }
    }

    for (i, to_add) in index.iter().enumerate() {
        if to_add.len() == 1 {
            res[i] = elems[to_add[0] as usize];
        } else if to_add.len() > 1 {
            debug_assert!(false, "Did not successfully reduce to_add");
        }
    }
}
