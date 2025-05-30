import bpy
import numpy as np
import sys
from mathutils import Vector
from pathlib import Path
import datetime


def log(message):
    """Write debug output to file"""
    debug_path = Path.home() / "blender_swarmalator.log"
    with open(debug_path, "a") as f:
        f.write(f"{message}\n")


def load_data():
    """Find the data path passed after --"""
    try:
        sep_index = sys.argv.index("--")
        return Path(sys.argv[sep_index + 1])
    except (ValueError, IndexError):
        log("ERROR: No data file specified after --")
        sys.exit(1)


def load_save_path():
    """Find the save path for the rendered animation if specified.
    If not specified, return None.
    """
    """Find the save path for the rendered animation if specified after --"""
    try:
        sep_index = sys.argv.index("--")
        if sep_index + 2 < len(sys.argv):
            return Path(sys.argv[sep_index + 2])
        else:
            return None
    except (ValueError, IndexError):
        log("No save path specified for the rendered animation.")
        return None


def setup_scene():
    """Clear existing data and setup a new scene"""
    # Clean slate
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create a new collection for our particles
    collection = bpy.data.collections.new("Swarmalators")
    bpy.context.scene.collection.children.link(collection)

    return collection


def create_particle_system(positions, orientations, collection):
    """Create Blender objects for particles and orientation arrows"""
    # Configuration
    particle_scale = 0.1
    arrow_scale = (0.05, 0.05, 0.2)  # (tail, head length)

    # Create template objects
    sphere = bpy.data.meshes.new("ParticleSphere")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=particle_scale)
    sphere_obj = bpy.context.object

    arrow = bpy.data.meshes.new("OrientationArrow")
    bpy.ops.mesh.primitive_cone_add(radius=arrow_scale[0], depth=arrow_scale[2])
    arrow_obj = bpy.context.object

    # Create instances for all particles
    particles = []
    for i in range(positions.shape[1]):
        # Duplicate sphere for particle
        part = sphere_obj.copy()
        part.data = sphere.copy()
        part.name = f"Particle_{i:03d}"
        collection.objects.link(part)

        # Duplicate arrow for orientation
        arr = arrow_obj.copy()
        arr.data = arrow.copy()
        arr.name = f"Arrow_{i:03d}"
        arr.parent = part  # Make arrow child of particle
        collection.objects.link(arr)

        particles.append((part, arr))

    # Remove templates
    bpy.data.meshes.remove(sphere)
    bpy.data.meshes.remove(arrow)

    return particles


def animate_particles(particles, positions, orientations, fps=24):
    """Set keyframes for positions and rotations"""
    scene = bpy.context.scene
    scene.frame_end = positions.shape[0] - 1

    for frame_idx in range(positions.shape[0]):
        scene.frame_set(frame_idx)

        for i, (part, arrow) in enumerate(particles):
            # Set position
            part.location = Vector(positions[frame_idx, i])
            part.keyframe_insert(data_path="location")

            # Set orientation (align arrow with orientation vector)
            rot = Vector(orientations[frame_idx, i]).to_track_quat("Z", "Y")
            arrow.rotation_euler = rot.to_euler()
            arrow.keyframe_insert(data_path="rotation_euler")


def main():
    # Get the data path passed from command line
    log("\n--- New Blender Session ---")

    try:
        # 1. Load data
        data_path = load_data()
        log(f"Loading data from: {data_path}")
        data = np.load(data_path)
        pos, ori = data["pos"], data["ori"]
        log(f"Data loaded: {pos.shape} positions, {ori.shape} orientations")

        # 2. Clean scene
        log("Cleaning existing objects")
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # 3. Create particle system
        log("Creating particle system")
        collection = bpy.data.collections.new("Swarmalators")
        bpy.context.scene.collection.children.link(collection)

        # Create particle system
        particles = create_particle_system(pos, ori, collection)

        # Animate
        animate_particles(particles, pos, ori)

        # Setup camera and lighting (optional)
        bpy.ops.object.camera_add(location=(0, -5, 2))
        bpy.context.scene.camera = bpy.context.object
        bpy.ops.object.light_add(type="SUN", location=(5, 5, 5))

        # Set render settings
        bpy.context.scene.render.fps = 24
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.ffmpeg.format = "MPEG4"

        # specify where to save the rendered animation
        save_path = load_save_path()
        if save_path is None:
            # default save path is to ./animation_outputs with a timestamp
            base_dir = Path.home() / "animation_outputs"
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = base_dir / "blender_animation.blend"
        log(f"Rendering animation to: {save_path}")

        bpy.context.scene.render.filepath = save_path

        print("Animation ready! Render from Blender's Render menu.")
    except Exception as e:
        log(f"FATAL ERROR: {str(e)}")
        import traceback

        log(traceback.format_exc())
        sys.exit(1)
