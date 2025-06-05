"""
Menagerie Asset Server
Serves complete scene packages with all assets resolved
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def get_menagerie_mapping() -> Dict[str, str]:
    """Map playground environment names to menagerie folders."""
    return {
        "G1JoystickFlatTerrain": "unitree_g1",
        "G1JoystickRoughTerrain": "unitree_g1",
        "Go1JoystickFlatTerrain": "unitree_go1",
        "Go1JoystickRoughTerrain": "unitree_go1",
        "BarkourJoystick": "google_barkour_vb",
        "SpotJoystickGaitTracking": "boston_dynamics_spot",
        "SpotFlatTerrainJoystick": "boston_dynamics_spot",
        "H1JoystickGaitTracking": "unitree_h1",
        "H1InplaceGaitTracking": "unitree_h1",
        "Op3Joystick": "robotis_op3",
        "BerkeleyHumanoidJoystickFlatTerrain": "berkeley_humanoid",
        "BerkeleyHumanoidJoystickRoughTerrain": "berkeley_humanoid",
        "BerkeleyHumanoidInplaceGaitTracking": "berkeley_humanoid",
        "A1JoystickFlatTerrain": "unitree_a1",
        "A1JoystickRoughTerrain": "unitree_a1",
        "CassieJoystick": "agility_cassie",
    }


def find_xml_file(env_name: str, xml_filename: str, temp_menagerie: Path) -> Optional[Path]:
    """Find the actual XML file, handling various naming conventions."""
    mapping = get_menagerie_mapping()
    folder = mapping.get(env_name)
    
    if not folder:
        return None
        
    base_dir = temp_menagerie / folder
    if not base_dir.exists():
        return None
    
    # Check various possible locations and naming conventions
    possible_files = [
        base_dir / xml_filename,
        base_dir / xml_filename.replace('_mjx_feetonly', ''),
        base_dir / xml_filename.replace('_mjx', ''),
        base_dir / xml_filename.replace('_feetonly', ''),
        base_dir / f"{folder}.xml",
        base_dir / "scene.xml",
        base_dir / "scene_mjx.xml",
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            return file_path
    
    return None


def collect_scene_assets(env_name: str, base_xml: str, temp_menagerie: Path) -> Dict[str, bytes]:
    """
    Collect all assets needed for a scene.
    Returns a dict mapping virtual paths to file contents.
    """
    assets = {}
    mapping = get_menagerie_mapping()
    folder = mapping.get(env_name)
    
    if not folder:
        return assets
        
    base_dir = temp_menagerie / folder
    if not base_dir.exists():
        return assets
    
    # Parse XML to find references
    try:
        root = ET.fromstring(base_xml)
    except:
        return assets
    
    # Process includes
    for include in root.findall('.//include'):
        file_attr = include.get('file')
        if file_attr:
            xml_path = find_xml_file(env_name, file_attr, temp_menagerie)
            if xml_path:
                assets[file_attr] = xml_path.read_bytes()
                
                # Recursively process the included file
                included_xml = xml_path.read_text()
                sub_assets = collect_scene_assets(env_name, included_xml, temp_menagerie)
                assets.update(sub_assets)
    
    # Collect mesh files
    for mesh in root.findall('.//mesh'):
        file_attr = mesh.get('file')
        if file_attr:
            # Check in assets subdirectory first, then base directory
            mesh_paths = [
                base_dir / "assets" / file_attr,
                base_dir / file_attr,
            ]
            for mesh_path in mesh_paths:
                if mesh_path.exists():
                    assets[file_attr] = mesh_path.read_bytes()
                    break
    
    # Collect texture files
    for texture in root.findall('.//texture'):
        file_attr = texture.get('file')
        if file_attr:
            texture_paths = [
                base_dir / "assets" / file_attr,
                base_dir / file_attr,
            ]
            for texture_path in texture_paths:
                if texture_path.exists():
                    assets[file_attr] = texture_path.read_bytes()
                    break
    
    # Collect hfield files
    for hfield in root.findall('.//hfield'):
        file_attr = hfield.get('file')
        if file_attr:
            hfield_paths = [
                base_dir / "assets" / file_attr,
                base_dir / file_attr,
            ]
            for hfield_path in hfield_paths:
                if hfield_path.exists():
                    assets[file_attr] = hfield_path.read_bytes()
                    break
    
    return assets


def resolve_includes_in_xml(xml_content: str, env_name: str, temp_menagerie: Path) -> str:
    """
    Resolve all include tags in the XML by replacing them with the actual content.
    """
    try:
        root = ET.fromstring(xml_content)
    except:
        return xml_content
    
    # Process includes
    includes = root.findall('.//include')
    for include in includes:
        file_attr = include.get('file')
        if file_attr:
            xml_path = find_xml_file(env_name, file_attr, temp_menagerie)
            if xml_path:
                # Read the included XML
                included_content = xml_path.read_text()
                
                # Parse the included XML
                try:
                    included_root = ET.fromstring(included_content)
                    
                    # Get the parent of the include element
                    parent = None
                    for elem in root.iter():
                        if include in elem:
                            parent = elem
                            break
                    
                    if parent is not None:
                        # Get the index before removing
                        insert_index = list(parent).index(include)
                        
                        # Remove the include element
                        parent.remove(include)
                        
                        # Insert all children from the included XML
                        for child in included_root:
                            parent.insert(insert_index, child)
                            insert_index += 1
                except:
                    # If parsing fails, just leave the include as is
                    pass
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')


def get_complete_scene_package(category: str, env_name: str, base_xml: str, temp_menagerie: Path) -> Dict:
    """
    Get a complete scene package with all assets.
    Returns a dict with:
    - xml: The main XML content
    - assets: Dict mapping filenames to base64-encoded content
    """
    import base64
    
    # Resolve includes in the XML
    resolved_xml = resolve_includes_in_xml(base_xml, env_name, temp_menagerie)
    
    # Collect all assets
    binary_assets = collect_scene_assets(env_name, resolved_xml, temp_menagerie)
    
    # Convert binary assets to base64 for JSON transport
    assets = {}
    for filename, content in binary_assets.items():
        assets[filename] = base64.b64encode(content).decode('utf-8')
    
    return {
        "xml": resolved_xml,
        "assets": assets,
        "env_name": env_name,
        "category": category,
    }
