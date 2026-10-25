"""
Configuration utilities for loading and managing YAML configs
"""
import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import shutil


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config (overrides take precedence)"""
    merged = base_config.copy()
    for key, value in overrides.items():
        # Only override if a value was actually provided.
        if value is not None:
            merged[key] = value
    return merged


def save_config(config: Dict[str, Any], save_path: str, file_format: str = 'yaml'):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        if file_format == 'yaml':
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif file_format == 'json':
            json.dump(config, f, indent=2)
        elif file_format == 'txt':
            for key, value in config.items():
                f.write(f"{key}: {value}\n")


def setup_argparser() -> argparse.ArgumentParser:
    """Setup argument parser with all configuration options"""
    parser = argparse.ArgumentParser(description="Patch Correspondence Finder")
    
    # Config file argument (most important)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (required or use default)",
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, help="Path to the dataset root directory")
    parser.add_argument("--obj_id", type=int, help="Object ID in the dataset")
    parser.add_argument("--templates_dir", type=str, help="Path to the templates directory")
    
    # Model arguments
    parser.add_argument("--encoder_checkpoint_path", type=str, help="Path to the trained encoder checkpoint")
    parser.add_argument("--decoder_checkpoint_path", type=str, help="Path to the trained decoder checkpoint")
    parser.add_argument("--use_saved_high_level_embeddings", type=lambda x: str(x).lower() == 'true', help="Whether to use saved embeddings")
    
    # High-level encoding arguments
    parser.add_argument("--num_high_level_nearest_templates_to_search_for", type=int)
    parser.add_argument("--num_high_level_nearest_templates_to_use", type=int)
    parser.add_argument("--min_rot_dist_between_high_level_neighbors", type=int)
    parser.add_argument("--inplane_rotation_angle", type=int)
    
    # Patch extraction arguments
    parser.add_argument("--use_cnos_segmentations", type=lambda x: str(x).lower() == 'true')
    parser.add_argument("--edge_biased_patch_extraction", type=lambda x: str(x).lower() == 'true')
    parser.add_argument("--num_patch_neighbors_to_search_for", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--template_patch_min_distance", type=int)
    parser.add_argument("--query_patch_min_distance", type=int)
    parser.add_argument("--num_patch_neighbors_to_use", type=int)
    parser.add_argument("--estimate_scale", type=lambda x: str(x).lower() == 'true')
    
    # RANSAC arguments
    parser.add_argument("--num_ransac_hypotheses", type=int)
    parser.add_argument("--num_ransac_iterations", type=int)
    parser.add_argument("--ransac_inlier_threshold", type=float)
    parser.add_argument("--min_rot_dist_between_hypotheses", type=int)
    
    # Logging arguments
    parser.add_argument("--estimated_poses_csv", type=str, 
                        help="Path to the saved estimated poses to run MegaPose refineder on")
    parser.add_argument("--log_estimates_csv", type=lambda x: str(x).lower() == 'true')
    parser.add_argument("--results_dir", type=str, help="Directory to save results")
    parser.add_argument("--calc_geo_distances", type=lambda x: str(x).lower() == 'true')
    
    # Visualization arguments (these are lists in the config, handle separately)
    parser.add_argument("--visualizations", type=str, nargs='+', help="List of visualizations to enable")
    parser.add_argument("--refinements", type=str, nargs='+', help="List of refinements to apply")
    
    return parser


def load_config_with_overrides(config_path: Optional[str] = None, 
                               cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Load configuration with command-line overrides
    
    Args:
        config_path: Path to YAML config file
        cli_args: Parsed command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Load base config from file
    if config_path and Path(config_path).exists():
        config = load_yaml_config(config_path)
    else:
        # If no config file, start with empty dict (all values from CLI)
        config = {}
    
    # Apply command-line overrides
    if cli_args:
        overrides = {k: v for k, v in vars(cli_args).items() 
                    if v is not None and k != 'config'}
        config = merge_configs(config, overrides)
    
    return config


def save_run_metadata(config: Dict[str, Any], script_path: Optional[str] = None,
                      config_save_name: Optional[str] = None, save_details: Optional[bool] = True):
    """
    Save configuration and script copy for reproducibility
    
    Args:
        config: Configuration dictionary
        script_path: Path to the current script file
    """
    results_dir = config.get('results_dir', './results')
    
    if not config.get('log_estimates_csv', False):
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config in multiple formats for convenience
    if config_save_name is not None:
        config_save_path = os.path.join(results_dir, config_save_name)
    else:
        config_save_path = os.path.join(results_dir, 'config.yaml')

    if save_details:
        details_save_path = os.path.join(results_dir, 'details.txt')
    
    # Only save if doesn't exist (don't overwrite)
    save_config(config, config_save_path, file_format='yaml')
    save_config(config, details_save_path, file_format='txt')
    
    # Add timestamp and additional metadata
    with open(details_save_path, 'a') as f:
        f.write(f"\n# Run metadata\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
    
    # Copy script for future reference
    if not script_path is None:
        script_save_path = os.path.join(results_dir, os.path.basename(script_path))
        shutil.copyfile(script_path, script_save_path)


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Pretty print configuration"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group by category
    categories = {
        'Dataset': ['dataset_root', 'obj_id', 'templates_dir'],
        'Model': ['encoder_checkpoint_path', 'decoder_checkpoint_path', 'use_saved_high_level_embeddings'],
        'High-Level Search': ['num_high_level_nearest_templates_to_search_for', 
                             'num_high_level_nearest_templates_to_use',
                             'min_rot_dist_between_high_level_neighbors',
                             'inplane_rotation_angle'],
        'Patch Extraction': ['use_cnos_segmentations', 'edge_biased_patch_extraction',
                            'patch_size', 'num_patch_neighbors_to_search_for',
                            'template_patch_min_distance', 'query_patch_min_distance',
                            'num_patch_neighbors_to_use', 'estimate_scale'],
        'RANSAC': ['num_ransac_hypotheses', 'num_ransac_iterations',
                  'ransac_inlier_threshold', 'min_rot_dist_between_hypotheses'],
        'Visualization': ['visualizations', 'refinements'],
        'Logging': ['log_estimates_csv', 'results_dir', 'calc_geo_distances'],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        for key in keys:
            if key in config:
                value = config[key]
                # Format lists nicely
                if isinstance(value, list):
                    if len(value) == 0:
                        print(f"  {key}: []")
                    else:
                        print(f"  {key}:")
                        for item in value:
                            print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
    
    # Print any extra keys not in categories
    all_category_keys = set(k for keys in categories.values() for k in keys)
    extra_keys = set(config.keys()) - all_category_keys
    if extra_keys:
        print(f"\nOther:")
        for key in sorted(extra_keys):
            print(f"  {key}: {config[key]}")
    
    print(f"{'='*60}\n")
