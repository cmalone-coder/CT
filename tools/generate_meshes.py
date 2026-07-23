#!/usr/bin/env python3
"""
Regenerate the four bone/skin GLB meshes for the CT viewer from the raw DICOM
export and re-embed them into data.js.

Run from anywhere; point --dicom-zip at the raw export. Never commit that zip
or any extracted DICOM/volume data to the repo -- only this script and the
regenerated data.js are meant to land in git.
"""
import argparse
import base64
import io
import zipfile
from collections import defaultdict

import numpy as np
import pydicom
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

import networkx as nx


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_series(zip_path):
    """Group every DICOM file in the zip by SeriesInstanceUID."""
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if not n.endswith("/")]
    series = defaultdict(list)
    for n in names:
        try:
            ds = pydicom.dcmread(io.BytesIO(z.read(n)), force=True)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        uid = str(ds.get("SeriesInstanceUID", ""))
        series[uid].append(ds)
    return series


def pick_target_series(series):
    """Find the 196-slice, 512x512, 1mm-thickness head volumes, split by kernel."""
    candidates = []
    for uid, slices in series.items():
        ds0 = slices[0]
        if (
            len(slices) == 196
            and int(ds0.get("Rows", 0)) == 512
            and int(ds0.get("Columns", 0)) == 512
            and float(ds0.get("SliceThickness", 0)) == 1.0
        ):
            candidates.append((uid, slices))

    if len(candidates) != 2:
        raise RuntimeError(
            f"Expected exactly 2 matching 196-slice/512x512/1mm series, found {len(candidates)}"
        )

    sharp = soft = None
    for uid, slices in candidates:
        kernel = str(slices[0].get("ConvolutionKernel", ""))
        if "60" in kernel:
            sharp = slices
        elif "40" in kernel:
            soft = slices
    if sharp is None or soft is None:
        # Fall back: whichever isn't identified stays ambiguous -- fail loudly
        # rather than silently guessing.
        raise RuntimeError(
            "Could not identify sharp (bone) vs soft (skin) kernel series from "
            f"ConvolutionKernel tags: {[str(c[1][0].get('ConvolutionKernel')) for c in candidates]}"
        )
    return sharp, soft


def build_volume(slices):
    """Sort by physical z position and stack into an HU int16 volume."""
    slices = sorted(slices, key=lambda d: float(d.ImagePositionPatient[2]))
    slope = float(slices[0].get("RescaleSlope", 1))
    intercept = float(slices[0].get("RescaleIntercept", 0))
    raw = np.stack([s.pixel_array for s in slices]).astype(np.int32)
    hu = (raw * slope + intercept).astype(np.int16)

    row_spacing, col_spacing = [float(x) for x in slices[0].PixelSpacing]
    slice_thickness = float(slices[0].SliceThickness)
    orientation = [float(x) for x in slices[0].ImageOrientationPatient]
    row_cosine = np.array(orientation[0:3])  # direction of increasing column index
    col_cosine = np.array(orientation[3:6])  # direction of increasing row index
    origin = np.array([float(x) for x in slices[0].ImagePositionPatient])

    # The cross-product handedness of (row_cosine, col_cosine) doesn't
    # guarantee the slice-stacking normal points the same way as our
    # ascending-IPP[2] sort order -- verify against the real last-slice IPP
    # and flip if needed, rather than trusting an assumed convention.
    normal = np.cross(row_cosine, col_cosine)
    last_ipp = np.array([float(x) for x in slices[-1].ImagePositionPatient])
    n_slices = len(slices)
    predicted_last = origin + (n_slices - 1) * slice_thickness * normal
    actual_delta = last_ipp - origin
    predicted_delta = predicted_last - origin
    if np.dot(actual_delta, predicted_delta) < 0:
        normal = -normal
    # Sanity: predicted last-slice position (using the corrected normal)
    # should now closely match the real one.
    predicted_last = origin + (n_slices - 1) * slice_thickness * normal
    err = np.linalg.norm(predicted_last - last_ipp)
    if err > max(2.0, slice_thickness):
        print(f"  WARNING: slice-normal sanity check off by {err:.2f}mm -- "
              f"non-uniform spacing or gantry tilt may be present")

    return {
        "hu": hu,
        "row_spacing": row_spacing,
        "col_spacing": col_spacing,
        "slice_thickness": slice_thickness,
        "row_cosine": row_cosine,
        "col_cosine": col_cosine,
        "origin": origin,
        "normal": normal,
    }


def validate_kernel_choice(sharp_vol, soft_vol):
    """Print sharpness/noise diagnostics -- informational, doesn't gate the
    already-decided sharp-for-bone/soft-for-skin split."""
    mid = sharp_vol["hu"].shape[0] // 2
    for name, vol in [("sharp(bone src)", sharp_vol), ("soft(skin src)", soft_vol)]:
        sl = vol["hu"][mid].astype(np.float64)
        bone_mask = sl >= 250
        tissue_mask = (sl > 0) & (sl < 80)
        sharpness = np.var(ndimage.laplace(sl)[bone_mask]) if bone_mask.any() else float("nan")
        noise = np.std(sl[tissue_mask]) if tissue_mask.any() else float("nan")
        print(f"  {name}: bone-edge Laplacian variance={sharpness:.1f}  soft-tissue noise std={noise:.2f}")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def volume_to_display_mesh(vol, level):
    """marching_cubes -> real patient-space coords -> LPS-to-display remap."""
    hu = vol["hu"]
    verts_idx, faces, _normals, _values = measure.marching_cubes(
        hu,
        level=level,
        spacing=(vol["slice_thickness"], vol["row_spacing"], vol["col_spacing"]),
    )
    # verts_idx columns are (z_mm, row_mm, col_mm) in *index* space already
    # scaled by spacing -- convert to real per-vertex row/col fractional
    # indices to project through the DICOM orientation formula.
    z_mm = verts_idx[:, 0]
    row_mm = verts_idx[:, 1]
    col_mm = verts_idx[:, 2]
    row_idx = row_mm / vol["row_spacing"]
    col_idx = col_mm / vol["col_spacing"]
    slice_idx = z_mm / vol["slice_thickness"]

    # Standard DICOM patient-space formula (per-slice origin drifts along the
    # slice-normal direction as slice_idx increases -- approximate the origin
    # drift linearly using the volume's own slice spacing since we resliced
    # to uniform thickness).
    normal = vol["normal"]
    origin = vol["origin"]

    patient = (
        origin
        + np.outer(col_idx, vol["col_spacing"] * vol["row_cosine"])
        + np.outer(row_idx, vol["row_spacing"] * vol["col_cosine"])
        + np.outer(slice_idx, vol["slice_thickness"] * normal)
    )
    L, P, S = patient[:, 0], patient[:, 1], patient[:, 2]

    # Documented display remap: screen-right ~= patient's right, up = Superior,
    # +Z = Anterior (faces camera).
    display = np.stack([-L, S, -P], axis=1)

    mesh = trimesh.Trimesh(vertices=display, faces=faces, process=False)
    return mesh, verts_idx


def decimate(mesh, target_faces):
    import fast_simplification

    verts, faces = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_count=target_faces
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ---------------------------------------------------------------------------
# Density-based vertex coloring
# ---------------------------------------------------------------------------

def sample_hu_along_normals(mesh_raw, vol, offset_mm=1.5):
    """Sample HU ~offset_mm inward along the raw (undecimated) marching-cubes
    normals -- these are reliable pre-decimation, unlike normals recomputed
    after trimesh decimation."""
    normals = mesh_raw.vertex_normals
    sample_points_mm = mesh_raw.vertices - normals * offset_mm

    # Convert display-space mm back to (slice_idx, row_idx, col_idx) to sample
    # the HU volume. Invert the same transform used in volume_to_display_mesh.
    origin = vol["origin"]
    row_cosine = vol["row_cosine"]
    col_cosine = vol["col_cosine"]
    normal = vol["normal"]

    neg_L = sample_points_mm[:, 0]
    S = sample_points_mm[:, 1]
    neg_P = sample_points_mm[:, 2]
    L = -neg_L
    P = -neg_P
    patient = np.stack([L, P, S], axis=1) - origin

    # Solve for (col_idx*col_spacing, row_idx*row_spacing, slice_idx*thickness)
    # via the orthonormal-ish basis (row_cosine, col_cosine, normal).
    basis = np.stack([row_cosine, col_cosine, normal], axis=1)  # 3x3, columns are basis vectors
    coeffs = np.linalg.solve(basis, patient.T).T  # Nx3: [col_mm, row_mm, slice_mm]

    col_idx = coeffs[:, 0] / vol["col_spacing"]
    row_idx = coeffs[:, 1] / vol["row_spacing"]
    slice_idx = coeffs[:, 2] / vol["slice_thickness"]

    coords = np.stack([slice_idx, row_idx, col_idx], axis=0)
    hu_vals = ndimage.map_coordinates(
        vol["hu"].astype(np.float32), coords, order=1, mode="nearest"
    )
    return hu_vals


def transfer_to_decimated(raw_mesh, raw_values, decimated_mesh):
    tree = cKDTree(raw_mesh.vertices)
    _, idx = tree.query(decimated_mesh.vertices)
    return raw_values[idx]


def lerp_color(t, stops):
    """stops: list of (t_value, (r,g,b)) sorted ascending, t in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros((len(t), 3), dtype=np.float64)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        mask = (t >= t0) & (t <= t1)
        if not mask.any():
            continue
        local = (t[mask] - t0) / (t1 - t0) if t1 > t0 else 0
        c0a = np.array(c0, dtype=np.float64)
        c1a = np.array(c1, dtype=np.float64)
        out[mask] = c0a[None, :] + local[:, None] * (c1a - c0a)
    return out


BONE_STOPS = [
    (0.00, (0x8a, 0x4a, 0x1e)),  # dark burnt-orange, thin/porous bone
    (0.35, (0xd8, 0xa5, 0x4a)),  # gold
    (0.75, (0xf5, 0xef, 0xe2)),  # crisp white, dense cortical
    (1.00, (0xf8, 0xfb, 0xff)),  # near-white ceiling
]
METAL_COLOR = (0x5a, 0xd2, 0xff)  # cyan
METAL_HU_THRESHOLD = 5500

SKIN_STOPS = [
    (0.00, (0xb9, 0x7a, 0x57)),  # deeper tan (shadow / jaw / neck)
    (1.00, (0xe0, 0xac, 0x85)),  # lighter warm highlight
]


def color_bone(hu_values, p5, p95):
    t = (hu_values - p5) / (p95 - p5)
    colors = lerp_color(t, BONE_STOPS)
    metal_mask = hu_values >= METAL_HU_THRESHOLD
    colors[metal_mask] = np.array(METAL_COLOR, dtype=np.float64)
    return (colors / 255.0)


def color_skin(hu_values):
    p5, p95 = np.percentile(hu_values, [5, 95])
    span = max(p95 - p5, 1e-6)
    t = (hu_values - p5) / span
    colors = lerp_color(t, SKIN_STOPS)
    return colors / 255.0


def apply_vertex_colors(mesh, colors_0_1):
    rgba = np.concatenate(
        [colors_0_1, np.ones((len(colors_0_1), 1))], axis=1
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=(rgba * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Gap repair (fan-triangulation capping + seam-only smoothing)
# ---------------------------------------------------------------------------

def close_gaps(mesh):
    """Fan-triangulate boundary loops using only real existing boundary
    vertices (centroid + real verts, no synthesized geometry beyond the
    fan centroid point), then lightly smooth only the new cap + a 1-ring
    buffer around it so the repair blends in without softening the rest
    of the mesh."""
    edges = mesh.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges) == 0:
        return mesh, np.zeros(len(mesh.vertices), dtype=bool)

    g = nx.Graph()
    g.add_edges_from(boundary_edges.tolist())
    loops = list(nx.connected_components(g))

    new_verts = list(mesh.vertices)
    new_faces = list(mesh.faces)
    cap_vertex_mask_extra = []

    for loop_nodes in loops:
        loop_nodes = list(loop_nodes)
        if len(loop_nodes) < 3:
            continue
        sub = g.subgraph(loop_nodes)
        try:
            ordered = list(nx.cycle_basis(sub))[0]
        except IndexError:
            ordered = loop_nodes
        pts = np.array([new_verts[i] for i in ordered])
        centroid = pts.mean(axis=0)
        centroid_idx = len(new_verts)
        new_verts.append(centroid)
        cap_vertex_mask_extra.append(centroid_idx)
        n = len(ordered)
        for i in range(n):
            a = ordered[i]
            b = ordered[(i + 1) % n]
            new_faces.append([a, b, centroid_idx])

    repaired = trimesh.Trimesh(
        vertices=np.array(new_verts), faces=np.array(new_faces), process=False
    )

    # Build a smoothing mask: the new cap vertices plus a 1-ring buffer
    # around each original boundary vertex.
    mask = np.zeros(len(repaired.vertices), dtype=bool)
    mask[cap_vertex_mask_extra] = True
    boundary_vert_ids = np.unique(boundary_edges)
    mask[boundary_vert_ids] = True
    # expand by 1 ring
    vtx_neighbors = repaired.vertex_neighbors
    expanded = mask.copy()
    for v in np.where(mask)[0]:
        for nb in vtx_neighbors[v]:
            expanded[nb] = True
    return repaired, expanded


def smooth_masked(mesh, mask, iterations=2):
    """Humphrey smoothing restricted to a vertex mask -- blend the smoothed
    positions back in only where mask is True, leaving the rest of the mesh
    untouched."""
    original_verts = mesh.vertices.copy()
    smoothed = mesh.copy()
    trimesh.smoothing.filter_humphrey(smoothed, iterations=iterations)
    new_verts = original_verts.copy()
    new_verts[mask] = smoothed.vertices[mask]
    mesh.vertices = new_verts
    return mesh


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_tissue(vol, level, target_faces, color_fn, offset_mm):
    print(f"  marching_cubes level={level} ...")
    raw_mesh, _ = volume_to_display_mesh(vol, level)
    print(f"  raw mesh: {len(raw_mesh.vertices)} verts, {len(raw_mesh.faces)} faces")

    raw_hu = sample_hu_along_normals(raw_mesh, vol, offset_mm=offset_mm)

    print(f"  decimating to {target_faces} faces ...")
    decimated = decimate(raw_mesh, target_faces)
    dec_hu = transfer_to_decimated(raw_mesh, raw_hu, decimated)
    colors = color_fn(dec_hu)
    apply_vertex_colors(decimated, colors)

    print("  closing gaps ...")
    repaired, smooth_mask = close_gaps(decimated)
    # colors for new cap vertices: nearest-neighbor from real colored verts
    n_original = len(decimated.vertices)
    if len(repaired.vertices) > n_original:
        tree = cKDTree(decimated.vertices)
        new_pts = repaired.vertices[n_original:]
        _, idx = tree.query(new_pts)
        orig_colors = decimated.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        new_colors = orig_colors[idx]
        all_colors = np.concatenate([orig_colors, new_colors], axis=0)
    else:
        all_colors = decimated.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
    apply_vertex_colors(repaired, all_colors)

    smoothed_complete = smooth_masked(repaired.copy(), smooth_mask, iterations=2)
    apply_vertex_colors(smoothed_complete, all_colors)

    return decimated, smoothed_complete


def export_b64(mesh):
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-zip", required=True)
    ap.add_argument("--data-js-out", default="data.js")
    ap.add_argument("--bone-faces", type=int, default=280_000)
    ap.add_argument("--skin-faces", type=int, default=150_000)
    args = ap.parse_args()

    print("Loading DICOM series ...")
    series = load_series(args.dicom_zip)
    sharp_slices, soft_slices = pick_target_series(series)
    print(f"  sharp (bone) kernel: {soft_slices[0].get('ConvolutionKernel')} vs "
          f"{sharp_slices[0].get('ConvolutionKernel')}")

    sharp_vol = build_volume(sharp_slices)  # bone source
    soft_vol = build_volume(soft_slices)  # skin source

    print("Validating kernel choice (informational) ...")
    validate_kernel_choice(sharp_vol, soft_vol)

    bone_hu_all = sharp_vol["hu"]
    bone_voxels = bone_hu_all[bone_hu_all >= 250]
    p5, p95 = np.percentile(bone_voxels, [5, 95])
    print(f"Bone HU recalibration range: p5={p5:.0f} p95={p95:.0f}")

    print("Processing bone ...")
    bone_mesh, bone_complete = process_tissue(
        sharp_vol, level=250, target_faces=args.bone_faces,
        color_fn=lambda hu: color_bone(hu, p5, p95), offset_mm=1.5,
    )

    print("Processing skin ...")
    skin_mesh, skin_complete = process_tissue(
        soft_vol, level=-300, target_faces=args.skin_faces,
        color_fn=color_skin, offset_mm=1.5,
    )

    print("Exporting GLBs and encoding ...")
    bone_b64 = export_b64(bone_mesh)
    skin_b64 = export_b64(skin_mesh)
    bone_complete_b64 = export_b64(bone_complete)
    skin_complete_b64 = export_b64(skin_complete)

    with open(args.data_js_out, "w") as f:
        f.write(f'const BONE_GLB_B64 = "{bone_b64}";\n')
        f.write(f'const SKIN_GLB_B64 = "{skin_b64}";\n')
        f.write(f'const BONE_COMPLETE_GLB_B64 = "{bone_complete_b64}";\n')
        f.write(f'const SKIN_COMPLETE_GLB_B64 = "{skin_complete_b64}";\n')

    import os
    size_mb = os.path.getsize(args.data_js_out) / 1e6
    print(f"Wrote {args.data_js_out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
