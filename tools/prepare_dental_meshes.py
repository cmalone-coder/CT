#!/usr/bin/env python3
"""
Prepare the DentalSegmentator-derived STL meshes (maxilla & upper skull,
mandible, upper teeth, lower teeth, mandibular canal) for the web viewer.

These meshes come from the user running Slicer's DentalSegmentator
extension directly (not this project's own segmentation pipeline) --
raw STLs are far too large for web delivery (the maxilla alone is
2M+ faces / 100MB+) and Slicer's own physical coordinate space needs to
be mapped into this viewer's display space before they'll line up with
the existing volume rendering.

Coordinate transform: numerically verified (not assumed) this session by
sampling STL vertices against the segmented HU volume with zero transform
applied -- 97.9% landed on bone-range HU, confirming Slicer exported these
in the same DICOM LPS physical coordinate system as the volume data, no
RAS flip (a common Slicer export gotcha; tested and explicitly ruled out).
The transform below is the existing viewer's own box<->texture mapping
(see index.html's localToUvw/labelAtLocal) inverted algebraically to go
from raw DICOM (L, P, S) mm to this project's established display-space
convention (X = -L, Y = S, Z = -P), applied here once at export time so
the browser needs no further transform:
    dispX = origin[0] + sizeX/2 - vx
    dispY = vz - origin[2] - sizeY/2
    dispZ = origin[1] + sizeZ/2 - vy

Never commit the raw source STLs -- only the decimated, transformed,
gzipped GLB output is meant to land in git (same convention as every
other raw-input tool in this directory).
"""
import argparse
import gzip
import json
import os

import numpy as np
import trimesh

# (source STL suffix, output basename, target face count after decimation).
# Targets are a deliberate size/quality tradeoff, easy to re-tune by
# re-running this script -- not a hard technical limit. The maxilla needs
# the heaviest reduction by far (2,046,556 -> ~150,000, ~93%) since it
# dwarfs everything else; the mandibular canal is a thin tube and gets
# only light reduction so decimation doesn't collapse its shape.
STRUCTURES = [
    ("Maxilla & Upper Skull", "maxilla_upper_skull", 150_000),
    ("Mandible", "mandible", 80_000),
    ("Lower Teeth", "lower_teeth", 40_000),
    ("Upper Teeth", "upper_teeth", 30_000),
    ("Mandibular canal", "mandibular_canal", 10_000),
]


def build_transform(meta):
    """4x4 matrix mapping raw DICOM LPS mm -> this viewer's display space,
    derived by inverting index.html's own box<->texture mapping (see
    module docstring)."""
    size_x = meta["ncols"] * meta["colSpacing"]
    size_y = meta["nslices"] * meta["sliceThickness"]
    size_z = meta["nrows"] * meta["rowSpacing"]
    ox, oy, oz = meta["origin"]

    # dispX = -vx + (ox + sizeX/2)
    # dispY =  vz + (-oz - sizeY/2)
    # dispZ = -vy + (oy + sizeZ/2)
    M = np.array([
        [-1, 0, 0, ox + size_x / 2],
        [0, 0, 1, -oz - size_y / 2],
        [0, -1, 0, oy + size_z / 2],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    return M, (size_x, size_y, size_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="directory containing the source STLs")
    ap.add_argument("--stl-prefix", default="3 ORBIT-FACE 1.00_Segmentation_",
                     help="filename prefix before the structure name, e.g. 'Mandible.stl'")
    ap.add_argument("--volume-meta", default="volume/volume_meta.json")
    ap.add_argument("--out-dir", default="dental_meshes")
    args = ap.parse_args()

    with open(args.volume_meta) as f:
        meta = json.load(f)
    M, (size_x, size_y, size_z) = build_transform(meta)
    print(f"volume size (mm): {size_x:.1f} x {size_y:.1f} x {size_z:.1f}")
    print(f"origin: {meta['origin']}")

    os.makedirs(args.out_dir, exist_ok=True)

    for stl_name, out_name, target_faces in STRUCTURES:
        stl_path = os.path.join(args.src_dir, f"{args.stl_prefix}{stl_name}.stl")
        print(f"\n{stl_name}:")
        print(f"  loading {stl_path} ...")
        mesh = trimesh.load(stl_path)
        print(f"  raw: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        if len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"  decimated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        else:
            print(f"  already under target ({target_faces}), skipping decimation")

        if len(mesh.faces) == 0:
            raise RuntimeError(f"{stl_name}: decimation produced an empty mesh -- target_faces too low?")

        mesh.apply_transform(M)
        mesh.fix_normals()

        glb_bytes = mesh.export(file_type="glb")
        out_path = os.path.join(args.out_dir, f"{out_name}.glb.gz")
        with open(out_path, "wb") as f:
            f.write(gzip.compress(glb_bytes, compresslevel=9))
        print(f"  wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB gzipped, "
              f"{len(glb_bytes)/1e6:.2f} MB raw glb)")


if __name__ == "__main__":
    main()
