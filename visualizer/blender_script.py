import bpy
import numpy as np
import sys

def main():
    # Get temp file paths from command line
    args = sys.argv[sys.argv.index("--") + 1:]
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
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create a camera if none exists
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add()
        bpy.context.scene.camera = bpy.context.object

    # Create particles (spheres)
    particle_objs = []
    for i in range(N):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))
        obj = bpy.context.object
        particle_objs.append(obj)

    # Animate particles
    for t in range(T):
        bpy.context.scene.frame_set(t)
        for i in range(N):
            obj = particle_objs[i]  # Fixed: Use current object
            obj.location = pos[t, i]
            obj.keyframe_insert(data_path="location", frame=t)
            
            # Rotation (quaternion)
            obj.rotation_mode = 'QUATERNION'
            q = ori[t, i]
            obj.rotation_quaternion = (q[3], q[0], q[1], q[2])  # (w, x, y, z)
            obj.keyframe_insert("rotation_quaternion", frame=t)

    # Set animation range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = T - 1

    # Adjust camera view
    cam = bpy.context.scene.camera
    cam.location = (0, -10, 5)
    cam.rotation_euler = (np.pi/2, 0, np.pi)

    # # Render animation (optional)
    # bpy.context.scene.render.filepath = "/tmp/output_"
    # bpy.ops.render.render(animation=True)

    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath="output.blend")
    print("Saved Blender file: output.blend")

if __name__ == "__main__":
    main()