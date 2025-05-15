import bpy
import numpy as np
import sys
import os


def load_template(filepath, object_name=None, scale=0.1):
    """Load a template object from a .blend, .obj, or .fbx file"""

    # Get the file extension
    file_extension = os.path.splitext(filepath)[1].lower()

    if file_extension == ".blend":
        # Load a .blend file
        return load_from_blend(filepath, object_name, scale)

    elif file_extension == ".obj":
        # Load an .obj file
        return load_from_obj(filepath, scale)

    elif file_extension == ".fbx":
        # Load an .fbx file
        return load_from_fbx(filepath, scale)

    else:
        print(f"Unsupported file format: {file_extension}")
        return None


def load_from_fbx(filepath, scale=0.1):
    """Load an object from a .fbx file"""
    bpy.ops.import_scene.fbx(filepath=filepath)

    # Assuming the object is the active one after import
    obj = bpy.context.view_layer.objects.active
    if obj:
        obj.scale = (scale, scale, scale)  # Scale the object
        return obj
    else:
        print(f"Failed to load object from {filepath}")
        return None


def load_from_blend(filepath, object_name, scale):
    """Load a specific object from a .blend file."""
    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        if object_name in data_from.objects:
            data_to.objects.append(object_name)  # Load the object
        data_to.materials = data_from.materials  # Load all materials

    obj = bpy.data.objects.get(object_name)
    if not obj:
        print(f"Object '{object_name}' not found in {filepath}")
        return None

    obj_copy = obj.copy()
    bpy.context.collection.objects.link(obj_copy)  # Add to scene
    obj_copy.scale = (scale, scale, scale)  # Scale the object

    # Ensure materials and textures are correctly applied
    for mat_slot in obj_copy.material_slots:
        mat = mat_slot.material
        if mat and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    # Ensure texture file path is correct
                    tex_filename = os.path.basename(
                        node.image.filepath
                    )  # Extract filename
                    texture_folder = os.path.dirname(
                        filepath
                    )  # Assume texture is next to blend file
                    tex_path = os.path.join(texture_folder, tex_filename)

                    if os.path.exists(tex_path):
                        node.image.filepath = tex_path  # Update file path
                        node.image.reload()
                        print(f"Texture applied: {tex_path}")
                    else:
                        print(f"Warning: Missing texture at {tex_path}")

    return obj_copy


def load_from_obj(filepath, scale=0.001):
    """Load an object from a .obj file"""
    bpy.ops.wm.obj_import(filepath=filepath)

    # Assuming the object is the active one after import
    obj = bpy.context.view_layer.objects.active
    if obj:
        obj.scale = (scale, scale, scale)  # Scale the object
        return obj
    else:
        print(f"Failed to load object from {filepath}")
        return None


def duplicate_complex_object(obj):
    """Properly duplicates an object with all children"""
    # Create new empty collection
    temp_collection = bpy.data.collections.new("TempCopy")
    bpy.context.scene.collection.children.link(temp_collection)

    # Deep copy hierarchy
    root_copy = obj.copy()
    temp_collection.objects.link(root_copy)

    for child in obj.children:
        child_copy = child.copy()
        child_copy.parent = root_copy
        temp_collection.objects.link(child_copy)

    return root_copy


def main():
    # Get temp file paths from command line
    args = sys.argv[sys.argv.index("--") + 1 :]
    pos_path, ori_path = args[0], args[1]

    try:
        pos = np.load(pos_path)
        ori = np.load(ori_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Number of particles and frames
    T, N = pos.shape[:2]

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create a camera if none exists
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add()
        bpy.context.scene.camera = bpy.context.object

    # fish_template = load_template('/Users/charlesxu/Documents/MIT/geometric algebra/dynamics_inference/assets/betafish.blend')
    # particle_objs = [duplicate_complex_object(fish_template) for _ in range(N)]

    # scale = .006 for swarmalators

    # template_obj = load_template("/Users/charlesxu/Documents/MIT/geometric algebra/dynamics_inference/assets/Fish.blend", "Fish.001")
    # template_obj = load_template("/Users/charlesxu/Documents/MIT/geometric algebra/dynamics_inference/assets/betafish.blend", "Cube")
    template_obj = load_template(
        "/Users/charlesxu/Documents/MIT/geometric algebra/dynamics_inference/assets/cow/source/cow/cow.obj",
        scale=0.006,
    )
    # template_obj = load_template('/Users/charlesxu/Documents/MIT/geometric algebra/dynamics_inference/assets/cow-2/source/vaca/vaca.FBX')
    particle_objs = [template_obj.copy() for _ in range(N)]
    for obj in particle_objs:
        bpy.context.collection.objects.link(obj)

    # Animate particles
    for t in range(T):
        bpy.context.scene.frame_set(t)
        for i in range(N):
            obj = particle_objs[i]  # Fixed: Use current object
            obj.location = pos[t, i]
            obj.keyframe_insert(data_path="location", frame=t)

            # Rotation (quaternion)
            obj.rotation_mode = "QUATERNION"
            q = ori[t, i]
            obj.rotation_quaternion = (q[3], q[0], q[1], q[2])  # (w, x, y, z)
            obj.keyframe_insert("rotation_quaternion", frame=t)

    # Set animation range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = T - 1

    # Adjust camera view
    cam = bpy.context.scene.camera
    cam.location = (0, -10, 5)
    cam.rotation_euler = (np.pi / 2, 0, np.pi)

    light_data = bpy.data.lights.new(
        name="MainLight", type="SUN"
    )  # Creates a sun light
    light_object = bpy.data.objects.new(name="MainLight", object_data=light_data)
    bpy.context.collection.objects.link(light_object)  # Add to scene
    light_object.location = (5, -5, 5)  # Move light above the object

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"  # Switch to Material Preview

    # # Render animation (optional)
    # bpy.context.scene.render.filepath = "/tmp/output_"
    # bpy.ops.render.render(animation=True)

    # Set background color
    bpy.context.scene.world.color = (0.1, 0.1, 0.1)  #
    bpy.ops.wm.save_as_mainfile(filepath="output.blend")
    print("Saved Blender file: output.blend")


if __name__ == "__main__":
    main()
