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
import subprocess
import tempfile

import numpy as np
import trimesh

# (source STL suffix, output basename, target face count after decimation
# -- None means keep full resolution, no decimation at all). Full
# resolution is the default per explicit user request: this is the user's
# own real DentalSegmentator output and they want it embedded faithfully,
# not simplified. Raw STL is a wasteful format for this though (every
# triangle repeats its own 3 vertices rather than sharing an index
# buffer), so even at full triangle count, re-exporting to indexed
# binary GLB shrinks the file substantially with zero geometry loss --
# this is what actually keeps the large structures under GitHub's 100MB
# hard per-file limit (and avoids Git LFS, which doesn't reliably serve
# real content through GitHub Pages -- it tends to serve the LFS pointer
# text file instead of the binary unless extra proxying is set up).
STL_PREFIX = "3 ORBIT-FACE 1.00_Segmentation_"

# (source filename, output basename, target face count after decimation --
# None means keep full resolution). Filenames are given in full rather
# than assembled from a shared prefix, since not everything here follows
# the DentalSegmentator STL naming convention -- Segment_2.obj is a
# separately, manually-segmented plate+screws mesh (SPACE=LPS per its own
# header, same coordinate convention already verified for the STLs), not
# an automatic export.
STRUCTURES = [
    (f"{STL_PREFIX}Maxilla & Upper Skull.stl", "maxilla_upper_skull", None),
    (f"{STL_PREFIX}Mandible.stl", "mandible", None),
    (f"{STL_PREFIX}Lower Teeth.stl", "lower_teeth", None),
    (f"{STL_PREFIX}Upper Teeth.stl", "upper_teeth", None),
    (f"{STL_PREFIX}Mandibular canal.stl", "mandibular_canal", None),
    ("Segment_2.obj", "metal_hardware", None),
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
    ap.add_argument("--src-dir", required=True, help="directory containing the source meshes")
    ap.add_argument("--volume-meta", default="volume/volume_meta.json")
    ap.add_argument("--out-dir", default="dental_meshes")
    args = ap.parse_args()

    with open(args.volume_meta) as f:
        meta = json.load(f)
    M, (size_x, size_y, size_z) = build_transform(meta)
    print(f"volume size (mm): {size_x:.1f} x {size_y:.1f} x {size_z:.1f}")
    print(f"origin: {meta['origin']}")

    os.makedirs(args.out_dir, exist_ok=True)

    for src_name, out_name, target_faces in STRUCTURES:
        src_path = os.path.join(args.src_dir, src_name)
        print(f"\n{src_name}:")
        print(f"  loading {src_path} ...")
        mesh = trimesh.load(src_path, force="mesh")
        print(f"  raw: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        if target_faces is not None and len(mesh.faces) > target_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"  decimated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        else:
            print(f"  keeping full resolution (no decimation)")

        if len(mesh.faces) == 0:
            raise RuntimeError(f"{src_name}: decimation produced an empty mesh -- target_faces too low?")

        mesh.apply_transform(M)
        mesh.fix_normals()
        # trimesh's glb exporter only includes a NORMAL accessor if
        # vertex_normals has actually been computed/cached first -- it
        # does NOT compute them automatically at export time. Confirmed
        # by direct inspection: without this, the exported GLB had only
        # a POSITION attribute, no NORMAL at all, which is why the live
        # viewer rendered every part as a single flat, shading-less
        # color (MeshStandardMaterial has nothing to light against) with
        # black patches in places (undefined per-face behavior with no
        # normal data) -- not a data or transform problem, an export gap.
        _ = mesh.vertex_normals

        glb_bytes = mesh.export(file_type="glb")

        # Draco-compress the geometry via gltf-pipeline (the standard tool
        # for this -- hand-rolling the KHR_draco_mesh_compression extension
        # wiring in Python is fiddly and error-prone, this is the same
        # well-tested path the wider glTF ecosystem uses). This is
        # compression, not decimation: vertex/face count comes back
        # essentially unchanged after decoding (verified this session --
        # the Draco-compressed maxilla mesh's own declared accessor bounds
        # matched the pre-compression bounds to within ~0.1mm, a third of
        # this scan's voxel pitch), it just quantizes position/normal
        # precision to a level far finer than meaningful here. Cut the
        # maxilla mesh from 49MB to 2.2MB raw glb in this session's test.
        with tempfile.TemporaryDirectory() as tmp:
            plain_path = os.path.join(tmp, "plain.glb")
            draco_path = os.path.join(tmp, "draco.glb")
            with open(plain_path, "wb") as f:
                f.write(glb_bytes)
            subprocess.run(
                ["npx", "--yes", "gltf-pipeline", "-i", plain_path, "-o", draco_path, "-d"],
                check=True, capture_output=True, text=True, shell=True,
            )
            with open(draco_path, "rb") as f:
                draco_bytes = f.read()

        out_path = os.path.join(args.out_dir, f"{out_name}.glb.gz")
        with open(out_path, "wb") as f:
            f.write(gzip.compress(draco_bytes, compresslevel=9))
        print(f"  wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB gzipped, "
              f"{len(draco_bytes)/1e6:.2f} MB draco glb, {len(glb_bytes)/1e6:.2f} MB pre-draco glb)")


if __name__ == "__main__":
    main()
