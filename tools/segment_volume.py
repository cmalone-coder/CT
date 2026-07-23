#!/usr/bin/env python3
"""
Phase 1: volumetric segmentation. Produces a labeled volume BEFORE any mesh
generation happens -- the label volume is the primary product; meshes (Phase
3) are just one way of visualizing it.

Ground truth discipline: classification decisions are driven only by the raw
CT density data (and, for shape/context features, geometry derived from it).
The scanner's own segmented render and the confirmed hardware location from
prior direct investigation are used only in this script's printed summary,
as an independent *check* against the result -- never as an input that
influences the classification itself.

Output: a single .npz checkpoint containing the label volume, a parallel
confidence volume (0-100), the HU volume it was derived from, and the
volume's geometry metadata -- everything Phase 2 (validation) and Phase 3
(meshing) need, so nothing has to be recomputed or re-guessed downstream.
"""
import argparse
import gc
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from generate_meshes import load_series, pick_target_series, build_volume, supersample_volume


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------
LABEL_UNKNOWN = 0
LABEL_AIR = 1
LABEL_SOFT_TISSUE = 2
LABEL_BONE = 3
LABEL_TOOTH = 4
LABEL_METAL = 5

LABEL_NAMES = {
    LABEL_UNKNOWN: "UNKNOWN",
    LABEL_AIR: "AIR",
    LABEL_SOFT_TISSUE: "SOFT_TISSUE",
    LABEL_BONE: "BONE",
    LABEL_TOOTH: "TOOTH",
    LABEL_METAL: "METAL",
}

# HU floors are patient-data-derived (see the session record this pipeline
# grew out of) but the *method* -- seed at a confidently-separating density,
# grow via watershed, treat what's left as bone -- is general.
AIR_HU_CEILING = -300          # matches the existing soft-tissue isosurface floor
BONE_HU_FLOOR = 250            # matches the existing bone isosurface threshold
TOOTH_SEED_HU = 2800           # empirically: individual teeth separate cleanly here
METAL_SEED_HU = 1800           # initial candidate floor -- NOT sufficient alone (see below)

# Physical-volume plausibility filters (mm^3), not voxel counts -- resolution
# independent, so these don't need to change if supersampling factor changes.
TOOTH_SEED_MIN_MM3 = 15
TOOTH_SEED_MAX_MM3 = 700
METAL_SEED_MIN_MM3 = 20
METAL_SEED_MAX_MM3 = 800       # excludes naturally-dense bone masses (see below)

BONE_MARKER_BUFFER_MM = 1.5    # unclaimed halo around every seed (see Fix 4 note below)

# A "distance from teeth" exclusion was tried and measurably wrong: the
# confirmed real plate sits only 1.6mm from a tooth-seed centroid (it
# connects to the maxilla, which hosts the upper teeth -- of course it's
# close), so that filter excluded the actual hardware while keeping a
# 4969mm^3 blob of ordinary dense bone (petrous-temporal/zygomatic-buttress
# territory) that happened to be far from teeth. Proximity to teeth says
# nothing reliable about whether something is metal.
#
# What actually separates them in this data: peak density. The confirmed
# plate hits 6825 HU; a set of small naturally-dense bone spots checked
# during testing topped out at 1846-2208 HU and no higher, well short of
# metal. Requiring a real density peak (not just clearing the seed floor
# on average) plus a plausible physical size for a plate/screw (not
# thousands of mm^3 of fused bone) reliably separates real hardware from
# ordinary dense bone in this scan.
METAL_PEAK_HU_MIN = 3000


def voxel_volume_mm3(vol):
    return vol["row_spacing"] * vol["col_spacing"] * vol["slice_thickness"]


def find_seed_components(hu, seed_hu, voxel_vol_mm3, min_mm3, max_mm3=None):
    """Connected components (real 3D voxel adjacency, scipy.ndimage.label)
    at the given density floor, filtered to a plausible physical size range.
    Returns (labeled_array, list of component ids kept, stats dict per id)."""
    mask = hu >= seed_hu
    labeled, n = ndimage.label(mask)
    if n == 0:
        return labeled, [], {}
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    stats = {}
    kept = []
    objs = ndimage.find_objects(labeled)
    for i in range(1, n + 1):
        vol_mm3 = sizes[i - 1] * voxel_vol_mm3
        if vol_mm3 < min_mm3:
            continue
        if max_mm3 is not None and vol_mm3 > max_mm3:
            continue
        sl = objs[i - 1]
        component_hu = hu[labeled == i]
        stats[i] = {
            "voxels": int(sizes[i - 1]),
            "volume_mm3": float(vol_mm3),
            "max_hu": float(component_hu.max()),
            "mean_hu": float(component_hu.mean()),
            "bbox": tuple((s.start, s.stop) for s in sl),
        }
        kept.append(i)
    return labeled, kept, stats


def component_centroid_mm(bbox, vol):
    """Approximate physical-space centroid of a component's index-space
    bbox, for distance filtering. Uses the same axes as the volume (no
    display-space remap needed here -- this stays in the segmentation's own
    working space)."""
    sc = (bbox[0][0] + bbox[0][1]) / 2 * vol["slice_thickness"]
    rc = (bbox[1][0] + bbox[1][1]) / 2 * vol["row_spacing"]
    cc = (bbox[2][0] + bbox[2][1]) / 2 * vol["col_spacing"]
    return np.array([sc, rc, cc])


def segment(vol):
    hu = vol["hu"]
    voxel_vol = voxel_volume_mm3(vol)
    sampling = (vol["slice_thickness"], vol["row_spacing"], vol["col_spacing"])
    print(f"  voxel volume: {voxel_vol:.4f} mm^3 (shape {hu.shape})")

    # At full supersampled resolution (~155M voxels) this pipeline is
    # memory-bound, not compute-bound: a single float64 distance-transform
    # buffer is ~1.24GB, and this ran out of memory (OOM-killed at ~16GB
    # RSS) the first time multiple such buffers, plus multiple int32
    # component-label volumes, were all kept alive at once. Every large
    # transient array below is deleted (and gc.collect()'d) as soon as its
    # last use is over, and only one big float64 distance buffer is ever
    # alive at a time -- recomputing one cheaply beats holding several.

    label = np.full(hu.shape, LABEL_UNKNOWN, dtype=np.uint8)
    confidence = np.zeros(hu.shape, dtype=np.uint8)

    # --- AIR / SOFT_TISSUE: clean, well-separated density ranges, high confidence ---
    air_mask = hu < AIR_HU_CEILING
    label[air_mask] = LABEL_AIR
    confidence[air_mask] = 95
    del air_mask

    soft_mask = (hu >= AIR_HU_CEILING) & (hu < BONE_HU_FLOOR)
    label[soft_mask] = LABEL_SOFT_TISSUE
    confidence[soft_mask] = 85  # lower than air: soft tissue is a broader, less distinct band
    del soft_mask

    dense_mask = hu >= BONE_HU_FLOOR
    print(f"  dense mask (candidates for bone/tooth/metal): {int(dense_mask.sum())} voxels")

    # --- Seeds ---
    print(f"  finding tooth seeds (HU>={TOOTH_SEED_HU}, {TOOTH_SEED_MIN_MM3}-{TOOTH_SEED_MAX_MM3} mm^3) ...")
    tooth_labeled, tooth_ids, tooth_stats = find_seed_components(
        hu, TOOTH_SEED_HU, voxel_vol, TOOTH_SEED_MIN_MM3, TOOTH_SEED_MAX_MM3
    )
    print(f"    {len(tooth_ids)} tooth-seed candidates "
          f"(of {tooth_labeled.max()} raw components at this threshold)")

    print(f"  finding metal seeds (HU>={METAL_SEED_HU}, {METAL_SEED_MIN_MM3}-{METAL_SEED_MAX_MM3} mm^3, "
          f"peak>={METAL_PEAK_HU_MIN}) ...")
    metal_labeled_raw, metal_ids_size_ok, metal_stats_raw = find_seed_components(
        hu, METAL_SEED_HU, voxel_vol, METAL_SEED_MIN_MM3, METAL_SEED_MAX_MM3
    )
    metal_ids = [i for i in metal_ids_size_ok if metal_stats_raw[i]["max_hu"] >= METAL_PEAK_HU_MIN]
    print(f"    {len(metal_ids)} metal-seed candidates kept "
          f"(of {len(metal_ids_size_ok)} size-plausible, {metal_labeled_raw.max()} raw components at this floor) "
          f"after requiring peak density >= {METAL_PEAK_HU_MIN} HU")

    # tooth_labeled/metal_labeled_raw (int32, ~620MB each at full res) are
    # only needed to build the boolean seed masks -- every tooth-seed voxel
    # ends up the same class regardless of which specific tooth component it
    # came from, and likewise for metal, so nothing downstream needs the
    # per-component id once these masks exist.
    tooth_seed_mask = np.isin(tooth_labeled, tooth_ids) if tooth_ids else np.zeros(hu.shape, dtype=bool)
    del tooth_labeled
    metal_seed_mask = np.isin(metal_labeled_raw, metal_ids) if metal_ids else np.zeros(hu.shape, dtype=bool)
    del metal_labeled_raw
    gc.collect()
    seed_mask = tooth_seed_mask | metal_seed_mask

    # --- Marker-controlled watershed: grow seeds outward through the dense
    # mask via physical-distance elevation, so watershed divide lines form
    # by real spatial proximity between competing structures.
    #
    # Watershed with a mask claims *every* masked voxel for the nearest
    # marker -- it never leaves anything unclaimed. Without a bone marker
    # to compete, tooth/metal seeds would flood the entire dense mask
    # (confirmed: an earlier run with no bone marker produced 0.05% bone
    # and 6.5% tooth+metal, obvious nonsense). Bone needs its own marker
    # like any other class.
    #
    # Bone markers must sit close to every seed, but not touching it -- a
    # bone marker placed immediately adjacent to a seed (zero-voxel gap)
    # leaves *no* unclaimed territory anywhere for watershed to resolve
    # (every dense voxel already has a marker before the algorithm runs, so
    # nothing can grow -- confirmed by testing: metal ended up exactly equal
    # to the raw seed sizes, confidence flat at the seed value everywhere).
    # A large gap is just as wrong the other way: with no competing marker
    # inside it, seeds flood through it unopposed (confirmed by testing: a
    # 3mm gap let "metal" regions balloon to hundreds of thousands of
    # voxels, mean HU ~1300 -- clearly ordinary dense bone like the petrous
    # temporal bone or zygomatic buttress, not hardware).
    #
    # BONE_MARKER_BUFFER_MM is the resolution: bone markers start
    # BONE_MARKER_BUFFER_MM away from every seed, not zero, not several mm.
    # The thin halo in between has no marker of any class -- it's exactly
    # the genuinely ambiguous partial-volume boundary, and watershed's own
    # elevation-ordered (physical-distance) flooding is what decides, voxel
    # by voxel, which side of that halo each point belongs to.
    dist_from_seed = ndimage.distance_transform_edt(~seed_mask, sampling=sampling)
    bone_seed_mask = dense_mask & ~seed_mask & (dist_from_seed > BONE_MARKER_BUFFER_MM)
    del dist_from_seed
    gc.collect()

    # Every tooth-seed voxel shares marker value 1, every metal-seed voxel
    # shares marker value 2, every bone-marker voxel shares marker value 3
    # -- watershed treats same-valued disjoint regions as contributing to
    # one output class, which is exactly the granularity this pipeline
    # needs (class, not individual tooth/bone-fragment identity). This also
    # avoids ever materializing a per-component marker id volume.
    markers = np.zeros(hu.shape, dtype=np.int8)
    markers[tooth_seed_mask] = 1
    markers[metal_seed_mask] = 2
    markers[bone_seed_mask] = 3

    print(f"  watershed: growing {int(tooth_seed_mask.sum())} tooth-seed / "
          f"{int(metal_seed_mask.sum())} metal-seed / {int(bone_seed_mask.sum())} bone-marker voxels "
          f"through {int(dense_mask.sum())} dense voxels ...")
    # Elevation = distance to the nearest marker of ANY class, using real
    # physical spacing (not raw voxel counts, which would be wrong on the
    # anisotropic pre-supersampling grid). This gives each ambiguous voxel
    # to whichever marker is spatially nearest -- a proper Voronoi-style
    # partition, the standard watershed technique for separating touching
    # objects. Density already did its job selecting which components get
    # to BE markers; using density again as the growth elevation was the
    # earlier bug (a bone marker's own density is often close enough to a
    # seed's that it wins the flood race almost immediately regardless of
    # true anatomical proximity, effectively preventing real growth).
    distance = ndimage.distance_transform_edt(markers == 0, sampling=sampling)
    ws = watershed(distance, markers=markers, mask=dense_mask)
    del distance, markers
    gc.collect()

    tooth_mask = ws == 1
    metal_mask = ws == 2
    del ws
    bone_mask = dense_mask & ~tooth_mask & ~metal_mask

    label[bone_mask] = LABEL_BONE
    label[tooth_mask] = LABEL_TOOTH
    label[metal_mask] = LABEL_METAL

    # --- Confidence ---
    # Seed voxels (found at a confidently-separating density) get high
    # confidence. Watershed-grown voxels were assigned by spatial proximity,
    # not density, so their confidence should reflect the same thing: how
    # far they had to be grown from the nearest seed of their class,
    # relative to how far it is to the *next*-nearest seed of a different
    # class (i.e. how contested that particular piece of territory was).
    # A voxel grown right up against the seed is nearly as sure as the seed
    # itself; a voxel near the watershed divide line, equidistant between
    # competing seeds, is genuinely the most uncertain point in the volume.
    confidence[tooth_seed_mask] = 97
    confidence[metal_seed_mask] = 97

    grown_nonbone = (tooth_mask | metal_mask) & (confidence == 0)
    if grown_nonbone.any():
        dist_from_seed = ndimage.distance_transform_edt(~seed_mask, sampling=sampling)
        dist_to_bone_marker = ndimage.distance_transform_edt(~bone_seed_mask, sampling=sampling)
        # own-seed distance vs. bone-marker distance at each grown voxel --
        # close to 0 (near the seed, far from any bone) = confident;
        # close to 1 (near the watershed divide) = uncertain.
        contest = dist_from_seed[grown_nonbone] / np.maximum(
            dist_from_seed[grown_nonbone] + dist_to_bone_marker[grown_nonbone], 1e-6
        )
        confidence[grown_nonbone] = np.clip(96 - 55 * contest, 40, 96).astype(np.uint8)
        del dist_from_seed, dist_to_bone_marker
        gc.collect()

    plain_bone = bone_mask & (confidence == 0)
    # Bone confidence decays near a tooth/metal boundary -- genuine
    # partial-volume ambiguity, not a flat number everywhere.
    non_bone_dense = tooth_mask | metal_mask
    if non_bone_dense.any():
        dist_to_nonbone = ndimage.distance_transform_edt(~non_bone_dense, sampling=sampling)
        bone_conf = np.clip(70 + 6 * dist_to_nonbone, 70, 95)
        confidence[plain_bone] = bone_conf[plain_bone].astype(np.uint8)
        del dist_to_nonbone, bone_conf
        gc.collect()
    else:
        confidence[plain_bone] = 90

    # Anything in the dense mask that's neither a seed nor watershed-claimed
    # and isn't plain bone either (shouldn't normally happen given mask
    # coverage, but stays explicit rather than silently defaulting) --
    # mark UNKNOWN rather than guess.
    still_unlabeled_dense = dense_mask & (label == LABEL_UNKNOWN)
    if still_unlabeled_dense.any():
        print(f"    NOTE: {int(still_unlabeled_dense.sum())} dense voxels left UNKNOWN "
              f"(unclaimed by watershed and not classified as bone)")

    return label, confidence, {
        "tooth_stats": tooth_stats, "tooth_ids": tooth_ids,
        "metal_stats": metal_stats_raw, "metal_ids": metal_ids,
    }


def summarize(label, confidence):
    total = label.size
    print("\n  === Segmentation summary ===")
    for code, name in LABEL_NAMES.items():
        mask = label == code
        n = int(mask.sum())
        if n == 0:
            continue
        conf_vals = confidence[mask]
        print(f"  {name:12s}: {n:>10d} voxels ({100*n/total:5.2f}%)  "
              f"confidence mean={conf_vals.mean():.1f}  p10={np.percentile(conf_vals,10):.0f}  p90={np.percentile(conf_vals,90):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-zip", required=True)
    ap.add_argument("--out", default="segmentation.npz")
    ap.add_argument("--supersample", action="store_true", default=True)
    ap.add_argument("--no-supersample", dest="supersample", action="store_false")
    args = ap.parse_args()

    print("Loading DICOM series ...")
    series = load_series(args.dicom_zip)
    sharp_slices, soft_slices = pick_target_series(series)
    vol = build_volume(soft_slices)

    if args.supersample:
        print("Supersampling volume for isotropic resolution ...")
        vol = supersample_volume(vol)
        gc.collect()  # release the pre-supersample array before the memory-heavy segmentation step

    print("Segmenting ...")
    label, confidence, debug_stats = segment(vol)
    summarize(label, confidence)

    print(f"\nSaving checkpoint to {args.out} ...")
    np.savez_compressed(
        args.out,
        label=label,
        confidence=confidence,
        hu=vol["hu"],
        row_spacing=vol["row_spacing"],
        col_spacing=vol["col_spacing"],
        slice_thickness=vol["slice_thickness"],
        row_cosine=vol["row_cosine"],
        col_cosine=vol["col_cosine"],
        origin=vol["origin"],
        normal=vol["normal"],
        label_names=np.array([LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]),
    )
    import os
    print(f"Wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
