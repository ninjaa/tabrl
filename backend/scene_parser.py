"""
Scene Parser - Extract semantic structure from MuJoCo XML files
Provides LLM with meaningful context about available components
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SceneStructure:
    """Semantic structure of a MuJoCo scene"""
    name: str
    joints: List[str]           # Available joints for control
    bodies: List[str]           # Physical objects/links  
    sites: List[str]            # Attachment points (grippers, sensors)
    sensors: List[str]          # Sensor names
    actuators: List[str]        # Motor/actuator names
    geoms: List[str]           # Collision/visual geometry
    
    # Dimensions
    nq: int                     # Position dimension
    nv: int                     # Velocity dimension  
    nu: int                     # Action dimension
    
    # Task hints from XML
    task_objects: List[str]     # Objects that might be manipulated
    target_sites: List[str]     # Goal locations or targets

def parse_scene_xml(xml_path: str) -> SceneStructure:
    """Parse MuJoCo XML and extract semantic structure"""
    
    try:
        xml_path = Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Process includes to get all elements
        all_elements = [root]
        
        # Find and process included files
        for include in root.findall('.//include'):
            include_file = include.get('file')
            if include_file:
                include_path = xml_path.parent / include_file
                if include_path.exists():
                    try:
                        include_tree = ET.parse(include_path)
                        all_elements.append(include_tree.getroot())
                    except Exception as e:
                        print(f"Warning: Could not parse included file {include_file}: {e}")
        
        # Extract component names from all XML elements
        joints = []
        bodies = []
        sites = []
        sensors = []
        actuators = []
        geoms = []
        
        for element in all_elements:
            # Find all components across included files
            joints.extend([joint.get('name') for joint in element.findall('.//joint') 
                          if joint.get('name')])
            
            bodies.extend([body.get('name') for body in element.findall('.//body') 
                          if body.get('name')])
            
            sites.extend([site.get('name') for site in element.findall('.//site') 
                         if site.get('name')])
            
            sensors.extend([sensor.get('name') for sensor in element.findall('.//sensor') 
                           if sensor.get('name')])
            
            actuators.extend([actuator.get('name') for actuator in element.findall('.//actuator') 
                             if actuator.get('name')])
            
            actuators.extend([general.get('name') for general in element.findall('.//actuator/general') 
                             if general.get('name')])
            actuators.extend([motor.get('name') for motor in element.findall('.//actuator/motor') 
                             if motor.get('name')])
            actuators.extend([position.get('name') for position in element.findall('.//actuator/position') 
                             if position.get('name')])
            actuators.extend([velocity.get('name') for velocity in element.findall('.//actuator/velocity') 
                             if velocity.get('name')])
            
            geoms.extend([geom.get('name') for geom in element.findall('.//geom') 
                         if geom.get('name')])
        
        # Remove duplicates while preserving order
        joints = list(dict.fromkeys(joints))
        bodies = list(dict.fromkeys(bodies))
        sites = list(dict.fromkeys(sites))
        sensors = list(dict.fromkeys(sensors))
        actuators = list(dict.fromkeys(actuators))
        geoms = list(dict.fromkeys(geoms))
        
        # Estimate dimensions (approximation)
        nq = len(joints) + len([b for b in bodies if 'free' in b.lower()])  # Free joints add 7 DOF
        nv = nq  # Usually same for most robots
        nu = len(actuators)
        
        # Extract task-relevant objects
        task_objects = [name for name in bodies + geoms 
                       if any(keyword in name.lower() 
                             for keyword in ['block', 'ball', 'cube', 'target', 'object', 'goal'])]
        
        target_sites = [name for name in sites 
                       if any(keyword in name.lower() 
                             for keyword in ['target', 'goal', 'destination', 'pickup', 'place'])]
        
        # Get model name
        model_name = root.get('model', xml_path.stem)
        
        return SceneStructure(
            name=model_name,
            joints=joints,
            bodies=bodies,
            sites=sites,
            sensors=sensors,
            actuators=actuators,
            geoms=geoms,
            nq=nq,
            nv=nv,
            nu=nu,
            task_objects=task_objects,
            target_sites=target_sites
        )
        
    except Exception as e:
        # Fallback structure if parsing fails
        print(f"Warning: XML parsing failed for {xml_path}: {e}")
        return SceneStructure(
            name="unknown_scene",
            joints=[], bodies=[], sites=[], sensors=[], actuators=[], geoms=[],
            nq=0, nv=0, nu=0, task_objects=[], target_sites=[]
        )

def generate_llm_context(scene_structure: SceneStructure, task_prompt: str) -> str:
    """Generate context string for LLM policy generation"""
    
    context = f"""
You are generating a reinforcement learning policy for the task: "{task_prompt}"

SCENE ANALYSIS:
- Robot/Scene: {scene_structure.name}
- Action Dimension: {scene_structure.nu} (what your policy outputs)
- State Dimension: {scene_structure.nq + scene_structure.nv} (joint pos + vel)

AVAILABLE COMPONENTS:
- Joints: {scene_structure.joints}
- Bodies: {scene_structure.bodies}
- Sites: {scene_structure.sites} (attachment points, often gripper tips)
- Sensors: {scene_structure.sensors}
- Actuators: {scene_structure.actuators}

TASK-RELEVANT OBJECTS:
- Manipulatable Objects: {scene_structure.task_objects}
- Target Locations: {scene_structure.target_sites}

You can access these in reward functions using semantic names:
- get_body_pos('object_name') → [x, y, z] position
- get_joint_angle('joint_name') → angle in radians  
- get_site_pos('site_name') → [x, y, z] position
- get_sensor('sensor_name') → sensor value

Generate 3 different reward function approaches for this task.
"""
    
    return context.strip()
