import blenderproc as bproc
import argparse
import glob
import json
import os
import sys

import numpy as np
from mathutils import Vector
from PIL import Image


def convert_rgb_dir_to_grayscale(output_dir: str) -> None:
    """Convert all RGBA PNGs written by write_bop (in rgb/) to grayscale+alpha (LA).

    Overwrites the files in-place so the BOP directory layout is preserved.
    Grayscale uses the standard luminosity weights (ITU-R BT.601).
    """
    png_paths = glob.glob(os.path.join(output_dir, "train_pbr", "000000", "rgb", "*.png"))
    for path in sorted(png_paths):
        img = Image.open(path).convert("RGBA")
        r, g, b, a = img.split()
        gray = Image.merge("RGB", (r, g, b)).convert("L")  # luminosity weights
        img_la = Image.merge("LA", (gray, a))
        img_la.save(path)


def render_blender_proc(
    cad_path,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    dataset_name="",
    recenter_origin=False,
    scale_mm_to_m=True,
    convert_to_grayscale=False,
):
    bproc.init()
    bproc.renderer.set_world_background([0, 0, 0], strength=0)

    # Camera at origin, +Z forward, +X right, +Y down
    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.set_intrinsics_from_K_matrix(intrinsic, img_size[1], img_size[0])

    if dataset_name == "itodd":
        # ITODD needs its own lighting rig: 8 point lights at all corners of the
        # front half-cube (x,y in {-1,1}, z in {0,1}), all z>=0 so every light hits
        # the faces visible to the camera. Source: nv-nguyen/gigapose
        # src/lib3d/blenderproc.py. Kept ITODD-only since it changes overall
        # brightness/shading vs. the ZS6D rig used for every other dataset below.
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [0, 1]:
                    light = bproc.types.Light()
                    light.set_type("POINT")
                    light.set_location([x, y, z])
                    light.set_energy(50)
    else:
        # Lights — ZS6D setup: 2 primary key lights + 2 fill lights (fixed, cross-dataset)
        # Source: haberger/ZS6D_template_rendering
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([2, -2, 2])
        light.set_energy(70)

        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([-2, -2, -2])
        light.set_energy(70)

        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([-1, 0, -1])
        light.set_energy(20)

        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([1, 0, 1])
        light.set_energy(20)

    # Load the object
    obj = bproc.loader.load_obj(cad_path)[0]

    # # Material assignment — GigaPose pattern (src/lib3d/blenderproc.py)
    # # T-LESS: textureless objects need explicit gray Principled BSDF
    # # Others: vertex colors baked into PLY, map them to the material
    # if dataset_name == "tless":
    #     mat = obj.get_materials()[0]
    #     mat.set_principled_shader_value("Base Color", [0.4, 0.4, 0.4, 1])
    # else:
    #     for mat in obj.get_materials():
    #         mat.map_vertex_color()

    if dataset_name == "itodd":
        # ITODD objects are dark industrial metal. Use a very dark diffuse base so
        # rendered brightness (~25-35/255) matches real ITODD images (~44/255).
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Base Color", [0.07, 0.07, 0.07, 1])

    # Convert mm -> m via bpy to avoid BProc scaling limitations
    if scale_mm_to_m:
        try:
            obj.set_scale(Vector((0.001, 0.001, 0.001)))
            obj.persist_transformation_into_mesh(
                location=False, rotation=False, scale=True
            )

        except Exception as e:
            print(
                f"[WARN] Failed to set scale via bpy: {e}. Continuing without scale.",
                file=sys.stderr,
            )

    if recenter_origin:
        import bpy

        bpy.ops.object.select_all(action="DESELECT")
        current_obj = bpy.data.objects[obj.get_name()]
        bpy.context.view_layer.objects.active = current_obj
        current_obj.select_set(True)
        bbox = np.array(obj.get_bound_box())
        center = np.mean(bbox, axis=0)

        obj.set_location(-1 * center)
        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

    obj.set_cp("category_id", 1)

    # Render setup
    # activate normal and distance rendering
    bproc.renderer.enable_distance_output(True)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # set the amount of samples, which should be used for the color rendering
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_render_devices(desired_gpu_ids=[0])
    bproc.renderer.set_output_format(enable_transparency=True)

    # Register all pose keyframes (BlenderProc BOP batch pattern)
    import bpy
    obj.blender_obj.rotation_mode = 'QUATERNION'
    # obj_poses = obj_poses[:15]
    for idx_frame, obj_pose in enumerate(obj_poses):
        bpy.context.scene.frame_set(idx_frame)
        obj.set_local2world_mat(obj_pose)
        obj.blender_obj.keyframe_insert(data_path="location", frame=idx_frame)
        obj.blender_obj.keyframe_insert(data_path="rotation_quaternion", frame=idx_frame)
        obj.blender_obj.keyframe_insert(data_path="scale", frame=idx_frame)
        bproc.camera.add_camera_pose(cam2world, frame=idx_frame)

    # Single batch render — Cycles pays BVH build + shader compilation once
    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))

    bproc.writer.write_bop(
        output_dir,
        target_objects=[obj],
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="PNG",
        ignore_dist_thres=10,
        annotation_unit="m",
    )

    if convert_to_grayscale:
        print("Converting RGB to grayscale+alpha for ITODD...")
        convert_rgb_dir_to_grayscale(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cad_path",
        nargs="?",
        help="Path to the model file",
        default="data/models/obj_000001.ply",
    )
    parser.add_argument(
        "obj_pose",
        nargs="?",
        help="Path to obj poses",
        default="template_generation/predefined_poses/obj_poses_level1.npy",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Where to save output",
        default="data/test_template",
    )

    parser.add_argument(
        "camera_path",
        nargs="?",
        help="Path to the dataset camera.json (BOP format)",
        default=None,
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="BOP dataset name (e.g. tless, tudl, lm)",
        default="",
    )
    parser.add_argument(
        "disable_output", nargs="?", help="Disable output of blender", default="false"
    )

    args = parser.parse_args()
    poses = np.load(args.obj_pose)  # poses are already in meters and correct rotations

    if args.camera_path is None:
        raise ValueError("camera_path is required — pass the dataset's camera.json")
    with open(args.camera_path) as f:
        cam = json.load(f)
    intrinsic = np.array([
        [cam["fx"],      0.0, cam["cx"]],
        [     0.0, cam["fy"], cam["cy"]],
        [     0.0,      0.0,       1.0],
    ])
    img_size = [cam["height"], cam["width"]]

    if args.disable_output == "true":
        logfile = os.path.join(args.output_dir, "render.log")
        import os
        import sys

        os.makedirs(args.output_dir, exist_ok=True)
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    render_blender_proc(
        args.cad_path,
        args.output_dir,
        poses,
        intrinsic=intrinsic,
        img_size=img_size,
        dataset_name=args.dataset_name,
        recenter_origin=True,
        scale_mm_to_m=True,
        convert_to_grayscale=(args.dataset_name == "itodd"),
    )

    if args.disable_output == "true":
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system(f"rm {logfile}")
