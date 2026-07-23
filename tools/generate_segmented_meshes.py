#!/usr/bin/env python3
"""
Phase 3: generate independent per-structure meshes from the validated label
volume (Phase 1's segmentation.npz checkpoint), not from density heuristics
applied after the fact. Each mesh's extent is exactly the voxels Phase 1 (and
Phase 2, for validation) already decided belong to that structure -- nothing
here re-derives or second-guesses which voxels are bone/tooth/metal/soft
tissue. Each structure gets its own extraction isolevel, its own decimation
strategy (bone: heavy decimation, it's the largest structure by far; teeth:
light, they're small and already few triangles; metal: none, it's the
highest-stakes structure and already tiny; soft tissue: moderate, matching
the old skin mesh), and its own coloring (bone: real density gradient
recalibrated from bone-only voxels, now that bone/teeth/metal are no longer
mixed together in one percentile calculation; teeth/metal: flat, distinct,
unmistakable colors per prior explicit request; soft tissue: the existing
skin gradient).

Output: independent GLB files, one per structure (for inspection), plus
data.js with each mesh base64-embedded as its own constant -- the format
the viewer (Phase 4) loads directly, replacing the old single bone+skin
data.js the pre-segmentation pipeline produced.
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from generate_meshes import (
    volume_to_display_mesh, decimate, resolve_normal_sign,
    sample_hu_along_normals, transfer_to_decimated, lerp_color,
    apply_vertex_colors, export_b64, color_skin,
)
from segment_volume import (
    LABEL_BONE, LABEL_TOOTH, LABEL_METAL, LABEL_SOFT_TISSUE, LABEL_NAMES,
)

TEETH_COLOR = (0xd9, 0xcf, 0xfa)   # pale lavender -- matches the prior explicit request
METAL_COLOR = (0x29, 0x8b, 0xff)   # bold saturated blue -- unmistakable regardless of lighting

# Far enough below every isolevel used here (250 for bone/tooth/metal) that
# a masked-out voxel never accidentally crosses back above the isolevel --
# real HU in this dataset never goes anywhere near this low.
MASK_SENTINEL = -2000


def masked_field(hu_real, label, keep_code):
    field = hu_real.astype(np.float32).copy()
    field[label != keep_code] = MASK_SENTINEL
    return field


def crop_to_class(hu_real, label, keep_code, vol_geom, pad_vox=4):
    """Crop to the class's own bounding box (+ padding) before marching
    cubes -- identical result within the crop (nothing in the window
    changes value), but avoids paying full-volume marching-cubes cost for
    small structures (teeth, metal) that occupy a tiny fraction of it.
    Shifts 'origin' by the crop offset so real-world coordinates still
    line up exactly, using the same patient-space basis
    volume_to_display_mesh uses to place vertices."""
    mask = label == keep_code
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None, None, None
    lo = np.maximum(idx.min(axis=0) - pad_vox, 0)
    hi = np.minimum(idx.max(axis=0) + pad_vox + 1, np.array(hu_real.shape))
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    cropped_hu = hu_real[sl]
    cropped_label = label[sl]

    origin = vol_geom["origin"] + (
        lo[0] * vol_geom["slice_thickness"] * vol_geom["normal"]
        + lo[1] * vol_geom["row_spacing"] * vol_geom["col_cosine"]
        + lo[2] * vol_geom["col_spacing"] * vol_geom["row_cosine"]
    )
    cropped_geom = dict(vol_geom)
    cropped_geom["origin"] = origin
    return cropped_hu, cropped_label, cropped_geom


def build_structure_mesh(hu_real, label, vol_geom, keep_code, name, isolevel,
                          target_faces, color_fn, offset_mm=1.0):
    print(f"  [{name}] cropping to bounding box ...")
    cropped_hu, cropped_label, cropped_geom = crop_to_class(hu_real, label, keep_code, vol_geom)
    if cropped_hu is None:
        print(f"  [{name}] no voxels -- skipping")
        return None

    field = masked_field(cropped_hu, cropped_label, keep_code)
    extract_vol = dict(cropped_geom)
    extract_vol["hu"] = field
    print(f"  [{name}] marching_cubes level={isolevel} on {field.shape} crop ...")
    raw_mesh, _ = volume_to_display_mesh(extract_vol, level=isolevel)
    print(f"  [{name}] raw mesh: {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")
    if len(raw_mesh.faces) == 0:
        print(f"  [{name}] empty surface -- skipping")
        return None

    sample_vol = dict(cropped_geom)
    sample_vol["hu"] = cropped_hu  # sample real (unmasked) density for coloring, not the sentinel field
    sign = resolve_normal_sign(raw_mesh, sample_vol, offset_mm=offset_mm)
    hu_at_verts_raw = sample_hu_along_normals(raw_mesh, sample_vol, offset_mm, sign)

    if target_faces is not None and len(raw_mesh.faces) > target_faces:
        print(f"  [{name}] decimating to {target_faces} faces ...")
        decimated = decimate(raw_mesh, target_faces)
        hu_at_verts = transfer_to_decimated(raw_mesh, hu_at_verts_raw, decimated)
    else:
        print(f"  [{name}] no decimation ({len(raw_mesh.faces)} faces)")
        decimated = raw_mesh
        hu_at_verts = hu_at_verts_raw

    colors = color_fn(hu_at_verts)
    apply_vertex_colors(decimated, colors)
    print(f"  [{name}] final: {len(decimated.vertices)} verts, {len(decimated.faces)} faces")
    return decimated


def smooth_gradient_stops(hu_stops, n=128):
    """lerp_color does piecewise-LINEAR interpolation between stops -- with
    only a handful of control points, the slope changes abruptly at each
    one, and the eye is very sensitive to exactly that kind of slope
    discontinuity (Mach banding): it perceives a bright/dark contour line
    at the kink even though the underlying values are perfectly smooth.
    Confirmed this isn't a real geometric ripple first (checked the raw HU
    surface-crossing position slice-by-slice across the forehead at 5
    different columns -- perfectly smooth, no oscillation anywhere), which
    points at the color mapping, not the geometry.

    Fix: fit a shape-preserving smooth curve (PCHIP -- won't overshoot
    between control points the way a natural cubic spline can) through the
    same control colors, then resample it densely. Same control points,
    same intended look, but now the slope changes gradually across many
    tiny segments instead of jumping at a few widely-spaced ones."""
    from scipy.interpolate import PchipInterpolator
    ts = np.array([s[0] for s in hu_stops], dtype=np.float64)
    colors = np.array([s[1] for s in hu_stops], dtype=np.float64)
    fine_t = np.linspace(ts[0], ts[-1], n)
    fine_colors = np.stack([PchipInterpolator(ts, colors[:, c])(fine_t) for c in range(3)], axis=1)
    return list(zip(fine_t, fine_colors))


def color_bone_gradient(hu_values, p5, p995):
    # Same 4 control-point warm density gradient as before, but calibrated
    # from bone-only voxels now that bone/teeth/metal are already separated
    # by the label volume (no teeth/metal override needed here, this mesh
    # IS pure bone by construction), and smoothed to avoid Mach-banding
    # artifacts on the many low-frequency, real, continuous density
    # gradients a skull actually has (frontal bone thickness, sinus
    # pneumatization, etc.) -- see smooth_gradient_stops().
    hu_stops = [
        (p5,                       (0x8a, 0x4a, 0x1e)),  # dark burnt-orange, thin/porous bone
        (p5 + (p995 - p5) * 0.30,  (0xd8, 0xa5, 0x4a)),  # gold
        (p5 + (p995 - p5) * 0.65,  (0xf0, 0xe6, 0xd2)),  # warm off-white, typical dense cortical
        (p995,                     (0xff, 0xff, 0xff)),  # bright white, very dense cortical
    ]
    lo, hi = hu_stops[0][0], hu_stops[-1][0]
    stops01 = smooth_gradient_stops([((hv - lo) / (hi - lo), c) for hv, c in hu_stops])
    t = (hu_values - lo) / (hi - lo)
    return lerp_color(t, stops01) / 255.0


def color_flat(hu_values, rgb):
    return np.tile(np.array(rgb, dtype=np.float64) / 255.0, (len(hu_values), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="segmented_meshes")
    ap.add_argument("--data-js-out", default=None, help="if set, also write a data.js with each mesh base64-embedded")
    ap.add_argument("--bone-faces", type=int, default=800_000)
    ap.add_argument("--teeth-faces", type=int, default=200_000)
    ap.add_argument("--soft-tissue-faces", type=int, default=150_000)
    args = ap.parse_args()

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz)
    hu_real = d["hu"]
    label = d["label"]
    vol_geom = {
        "row_spacing": float(d["row_spacing"]),
        "col_spacing": float(d["col_spacing"]),
        "slice_thickness": float(d["slice_thickness"]),
        "row_cosine": d["row_cosine"],
        "col_cosine": d["col_cosine"],
        "origin": d["origin"],
        "normal": d["normal"],
    }

    os.makedirs(args.out_dir, exist_ok=True)

    bone_vals = hu_real[label == LABEL_BONE]
    p5, p995 = np.percentile(bone_vals, [5, 99.5])
    print(f"Bone HU recalibration (bone-only voxels): p5={p5:.0f} p99.5={p995:.0f}")

    print("Building BONE mesh ...")
    bone_mesh = build_structure_mesh(
        hu_real, label, vol_geom, LABEL_BONE, "bone", isolevel=250,
        target_faces=args.bone_faces,
        color_fn=lambda hu: color_bone_gradient(hu, p5, p995),
    )

    print("Building TOOTH mesh ...")
    tooth_mesh = build_structure_mesh(
        hu_real, label, vol_geom, LABEL_TOOTH, "tooth", isolevel=250,
        target_faces=args.teeth_faces,
        color_fn=lambda hu: color_flat(hu, TEETH_COLOR),
    )

    print("Building METAL mesh ...")
    metal_mesh = build_structure_mesh(
        hu_real, label, vol_geom, LABEL_METAL, "metal", isolevel=250,
        target_faces=None,  # no decimation -- smallest, highest-stakes structure
        color_fn=lambda hu: color_flat(hu, METAL_COLOR),
    )

    # Soft tissue's outer boundary is the skin surface -- exactly the old
    # single-shell skin extraction, unaffected by internal segmentation
    # (masking internal bone/tooth/metal out of this field would push
    # values that were always safely above -300 down to the sentinel,
    # crossing -300 and fabricating a phantom inner shell around them that
    # doesn't exist in the real density profile). Use the real, unmasked
    # volume, exactly as before.
    print("Building SOFT_TISSUE mesh ...")
    sample_vol = dict(vol_geom)
    sample_vol["hu"] = hu_real
    raw_mesh, _ = volume_to_display_mesh(sample_vol, level=-300)
    print(f"  [soft_tissue] raw mesh: {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")
    sign = resolve_normal_sign(raw_mesh, sample_vol, offset_mm=1.5)
    hu_at_verts_raw = sample_hu_along_normals(raw_mesh, sample_vol, 1.5, sign)
    if len(raw_mesh.faces) > args.soft_tissue_faces:
        print(f"  [soft_tissue] decimating to {args.soft_tissue_faces} faces ...")
        soft_mesh = decimate(raw_mesh, args.soft_tissue_faces)
        hu_at_verts = transfer_to_decimated(raw_mesh, hu_at_verts_raw, soft_mesh)
    else:
        soft_mesh = raw_mesh
        hu_at_verts = hu_at_verts_raw
    apply_vertex_colors(soft_mesh, color_skin(hu_at_verts))
    print(f"  [soft_tissue] final: {len(soft_mesh.vertices)} verts, {len(soft_mesh.faces)} faces")

    print(f"\nExporting GLBs to {args.out_dir}/ ...")
    meshes = {"bone": bone_mesh, "tooth": tooth_mesh, "metal": metal_mesh, "soft_tissue": soft_mesh}
    for name, mesh in meshes.items():
        if mesh is None:
            continue
        path = os.path.join(args.out_dir, f"{name}.glb")
        _ = mesh.vertex_normals  # force normals into the export, same fix as the original pipeline
        mesh.export(path, file_type="glb")
        print(f"  wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")

    if args.data_js_out:
        print(f"\nEncoding GLBs into {args.data_js_out} ...")
        const_names = {"bone": "BONE_GLB_B64", "tooth": "TOOTH_GLB_B64",
                        "metal": "METAL_GLB_B64", "soft_tissue": "SOFT_TISSUE_GLB_B64"}
        with open(args.data_js_out, "w") as f:
            for name, mesh in meshes.items():
                const = const_names[name]
                if mesh is None:
                    f.write(f'const {const} = null;\n')
                    continue
                f.write(f'const {const} = "{export_b64(mesh)}";\n')
        size_mb = os.path.getsize(args.data_js_out) / 1e6
        print(f"Wrote {args.data_js_out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
