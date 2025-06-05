"""
Playground Environment API Endpoints
Provides listing and XML retrieval for MuJoCo Playground environments
"""

from typing import Dict, List, Optional, Any
import tempfile
import os
from pathlib import Path

# Since playground is always installed in our Modal container, we can import directly
from mujoco_playground import registry


def list_playground_environments() -> Dict[str, List[str]]:
    """
    List all available MuJoCo Playground environments organized by category.
    
    Returns:
        Dictionary mapping category to list of environment names
    """
    environments = {}
    
    # Get locomotion environments
    if hasattr(registry, 'locomotion') and hasattr(registry.locomotion, '_envs'):
        environments['locomotion'] = sorted(list(registry.locomotion._envs.keys()))
    
    # Get manipulation environments
    if hasattr(registry, 'manipulation') and hasattr(registry.manipulation, '_envs'):
        environments['manipulation'] = sorted(list(registry.manipulation._envs.keys()))
    
    # Get dm_control_suite environments
    if hasattr(registry, 'dm_control_suite') and hasattr(registry.dm_control_suite, '_envs'):
        environments['dm_control_suite'] = sorted(list(registry.dm_control_suite._envs.keys()))
    
    return environments


def get_playground_xml(category: str, env_name: str) -> Optional[str]:
    """
    Get the XML content for a specific MuJoCo Playground environment.
    
    Args:
        category: Environment category ('locomotion', 'manipulation', etc.)
        env_name: Name of the environment
        
    Returns:
        XML content as string, or None if not found
    """
    try:
        category_module = getattr(registry, category, None)
        if not category_module:
            return None
            
        # Check if environment exists in _envs
        if hasattr(category_module, '_envs') and env_name in category_module._envs:
            # Instantiate the environment to get its properties
            env_class = category_module._envs[env_name]
            env = env_class()
            
            # Get XML path and read the file
            if hasattr(env, 'xml_path'):
                xml_path = env.xml_path
                if xml_path and Path(xml_path).exists():
                    with open(xml_path, 'r') as f:
                        return f.read()
                        
            # Alternative: try to get from mj_model
            if hasattr(env, 'mj_model') and env.mj_model:
                # MuJoCo model to XML conversion would go here
                # For now, just return None
                pass
                
        return None
    except Exception as e:
        print(f"Error getting XML for {category}/{env_name}: {str(e)}")
        return None


def get_playground_info(category: str, env_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata information about a specific playground environment.
    
    Args:
        category: Category name
        env_name: Environment name
        
    Returns:
        Dictionary with environment metadata or None if not found
    """
    try:
        category_module = getattr(registry, category, None)
        if not category_module:
            return None
            
        # Check if environment exists in _envs
        if hasattr(category_module, '_envs') and env_name in category_module._envs:
            # Instantiate the environment to get its properties
            env_class = category_module._envs[env_name]
            env = env_class()
            
            info = {
                "name": env_name,
                "category": category,
            }
            
            # Handle partial functions
            if hasattr(env_class, 'func'):
                info['class'] = env_class.func.__name__
                info['module'] = env_class.func.__module__
            elif hasattr(env_class, '__name__'):
                info['class'] = env_class.__name__
                info['module'] = env_class.__module__
            
            # Add available properties
            if hasattr(env, 'observation_size'):
                info['observation_size'] = env.observation_size
            if hasattr(env, 'action_size'):
                info['action_size'] = env.action_size
            if hasattr(env, 'dt'):
                info['dt'] = env.dt
            if hasattr(env, 'sim_dt'):
                info['sim_dt'] = env.sim_dt
            if hasattr(env, 'n_substeps'):
                info['n_substeps'] = env.n_substeps
            
            # Check for specific methods/capabilities
            capabilities = []
            if hasattr(env, 'render'):
                capabilities.append('render')
            if hasattr(env, 'reset'):
                capabilities.append('reset')
            if hasattr(env, 'step'):
                capabilities.append('step')
            if hasattr(env, 'sample_command'):
                capabilities.append('sample_command')
            
            info['capabilities'] = capabilities
            
            return info
            
        return None
    except Exception as e:
        print(f"Error getting info for {category}/{env_name}: {str(e)}")
        return None


def get_playground_assets(category: str, env_name: str) -> Dict[str, List[str]]:
    """
    Extract asset references (meshes, textures) from an environment's XML.
    
    Args:
        category: Category name
        env_name: Environment name
        
    Returns:
        Dictionary with lists of mesh files, texture files, etc.
    """
    xml = get_playground_xml(category, env_name)
    if not xml:
        return {"meshes": [], "textures": [], "includes": []}
        
    assets = {
        "meshes": [],
        "textures": [],
        "includes": []
    }
    
    # Extract mesh references (.obj, .stl files)
    import re
    mesh_pattern = r'file="([^"]+\.(obj|stl|ply|msh))"'
    for match in re.finditer(mesh_pattern, xml, re.IGNORECASE):
        assets["meshes"].append(match.group(1))
    
    # Extract texture references
    texture_pattern = r'file="([^"]+\.(png|jpg|jpeg|bmp))"'
    for match in re.finditer(texture_pattern, xml, re.IGNORECASE):
        assets["textures"].append(match.group(1))
        
    # Extract includes
    include_pattern = r'<include\s+file="([^"]+)"'
    for match in re.finditer(include_pattern, xml):
        assets["includes"].append(match.group(1))
    
    # Remove duplicates
    assets["meshes"] = list(set(assets["meshes"]))
    assets["textures"] = list(set(assets["textures"]))
    assets["includes"] = list(set(assets["includes"]))
    
    return assets


# Backward compatibility aliases
get_available_environments = list_playground_environments
get_environment_xml = get_playground_xml
get_environment_info = get_playground_info

if __name__ == "__main__":
    # Test the API
    print("Available environments:")
    envs = list_playground_environments()
    
    total_count = 0
    for category, env_list in envs.items():
        print(f"\n{category} ({len(env_list)} environments):")
        # Show first 5 and last 2 as sample
        if len(env_list) > 7:
            for env in env_list[:5]:
                print(f"  - {env}")
            print(f"  ... ({len(env_list) - 7} more)")
            for env in env_list[-2:]:
                print(f"  - {env}")
        else:
            for env in env_list:
                print(f"  - {env}")
        total_count += len(env_list)
    
    print(f"\nTotal environments: {total_count}")
    
    # Test XML retrieval
    print("\n\nTesting XML retrieval for Go1JoystickFlatTerrain...")
    xml = get_playground_xml("locomotion", "Go1JoystickFlatTerrain")
    if xml:
        print(f"✅ Got XML ({len(xml)} characters)")
        print("First 300 chars:", xml[:300])
        
        # Check for asset references
        if '.obj' in xml:
            print("✅ Contains .obj mesh references")
        if 'mesh' in xml:
            print("✅ Contains mesh elements")
        if 'texture' in xml:
            print("✅ Contains texture references")
    else:
        print("❌ Failed to get XML")
        
    # Test environment info
    print("\n\nTesting environment info for Go1JoystickFlatTerrain...")
    info = get_playground_info("locomotion", "Go1JoystickFlatTerrain")
    if info:
        print("✅ Got environment info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Failed to get environment info")
        
    # Test asset extraction
    print("\n\nTesting asset extraction for Go1JoystickFlatTerrain...")
    assets = get_playground_assets("locomotion", "Go1JoystickFlatTerrain")
    print("✅ Extracted assets:")
    for asset_type, asset_list in assets.items():
        if asset_list:
            print(f"  {asset_type}: {len(asset_list)} files")
            for asset in asset_list[:3]:  # Show first 3
                print(f"    - {asset}")
            if len(asset_list) > 3:
                print(f"    ... ({len(asset_list) - 3} more)")
        else:
            print(f"  {asset_type}: None")
