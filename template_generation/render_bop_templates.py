import glob
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

# from template_generation.utils import save_predefined_poses_with_inplane, get_obj_poses_from_template_level


def call_render(
    idx_obj,
    list_cad_path,
    list_output_dir,
    obj_pose_path,
    camera_path,
    bop_dataset_name,
    disable_output,
):
    root_dir = os.path.dirname(__file__)
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    if os.path.exists(output_dir):
        os.system("rm -r {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    command = f"blenderproc run {root_dir}/bproc_rendering.py {cad_path} {obj_pose_path} {output_dir} {camera_path} {bop_dataset_name}"

    if disable_output:
        command += " true"
    else:
        command += " false"
    # command += " true"  # scale translation to meter
    os.system(command)

    # make sure the number of rendered images is correct
    num_images = len(glob.glob(f"{output_dir}/train_pbr/000000/rgb/*.png"))
    if num_images == len(np.load(obj_pose_path)):
        return True
    else:
        print(f"Found only {num_images} for  {cad_path} {obj_pose_path}")
        return False


def render() -> None:
    disable_output = False
    root_dir = os.path.dirname(__file__)
    dataset_root = "./cap_obj/"
    bop_dataset_name = os.path.basename(dataset_root.rstrip("/"))
    cad_name = "models_cad" if bop_dataset_name == "tless" else "models"
    root_save_dir = os.path.join(dataset_root, "new_templates")
    dataset_name = "level2_icosphere"

    template_poses_path = os.path.join(
        os.path.dirname(root_dir),
        "predefined_template_poses",
        "obj_poses_level2_icosphere.npy",
    )
    template_poses = np.load(template_poses_path)
    template_poses[:, :3, 3] = np.array([0.0, 0.0, 0.4])

    dataset_save_dir = os.path.join(root_save_dir, f"{dataset_name}")
    print(f"Rendering templates for {dataset_name}")
    os.makedirs(dataset_save_dir, exist_ok=True)
    obj_pose_dir = os.path.join(dataset_save_dir, "object_poses")
    os.makedirs(obj_pose_dir, exist_ok=True)

    output_dirs = []
    obj_pose_path = os.path.join(obj_pose_dir, "obj_poses.npy")
    np.save(obj_pose_path, template_poses)

    cad_dir = os.path.join(dataset_root, cad_name)
    cad_paths = []
    for file in os.listdir(cad_dir):
        if file.endswith(".ply") or file.endswith(".fbx"):
        # if file.endswith(".ply"):
            cad_paths.append(os.path.join(cad_dir, file))

    print(f"Found {len(list(cad_paths))} objects")

    for cad_path in cad_paths:
        object_id = 1
        output_dir = os.path.join(dataset_save_dir, f"{object_id:06d}")
        output_dirs.append(output_dir)

    os.makedirs(dataset_save_dir, exist_ok=True)

    print("Start rendering for {} objects".format(len(cad_paths)))
    start_time = time.time()
    camera_path = os.path.join(dataset_root, "camera.json")

    pool = multiprocessing.Pool(processes=1)
    call_render_ = partial(
        call_render,
        list_cad_path=cad_paths,
        list_output_dir=output_dirs,
        obj_pose_path=obj_pose_path,
        camera_path=camera_path,
        bop_dataset_name=bop_dataset_name,
        disable_output=disable_output,
    )
    values = list(
        tqdm(
            pool.imap_unordered(call_render_, range(len(cad_paths))),
            total=len(cad_paths),
        )
    )
    correct_values = [val for val in values]
    print(f"Finished for {len(correct_values)}/{len(cad_paths)} objects")
    finish_time = time.time()
    print(f"Total time {len(cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    render()
