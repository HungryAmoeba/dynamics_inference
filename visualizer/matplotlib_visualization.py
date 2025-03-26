import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def animate_particle_motion(pos, ori=None, interval=50, title="Swarmalator Motion"):
    """
    Animate particle positions with optional orientation arrows.
    
    Args:
        pos: Array of shape (T, N, D) where D=2 or 3
        ori: Optional array of shape (T, N, D) for orientations
        interval: Delay between frames in milliseconds
        title: Title for the animation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Convert JAX arrays to numpy if needed
    if hasattr(pos, 'device_buffer'):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, 'device_buffer'):
        ori = np.array(ori)
        
    T, N, D = pos.shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D data supported")

    # Set up figure and axis
    fig = plt.figure(figsize=(8, 6))
    if D == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=45)
    else:
        ax = fig.add_subplot(111)
        
    # Set axis limits
    all_pos = pos.reshape(-1, D)
    min_vals = all_pos.min(axis=0)
    max_vals = all_pos.max(axis=0)
    padding = 0.1 * (max_vals - min_vals)
    
    if D == 3:
        ax.set_xlim3d(min_vals[0]-padding[0], max_vals[0]+padding[0])
        ax.set_ylim3d(min_vals[1]-padding[1], max_vals[1]+padding[1])
        ax.set_zlim3d(min_vals[2]-padding[2], max_vals[2]+padding[2])
    else:
        ax.set_xlim(min_vals[0]-padding[0], max_vals[0]+padding[0])
        ax.set_ylim(min_vals[1]-padding[1], max_vals[1]+padding[1])
    
    # Initialize visualization elements
    if D == 2:
        scat = ax.scatter(pos[0,:,0], pos[0,:,1], c='blue', s=50, alpha=0.8)
    else:
        scat = ax.scatter(pos[0,:,0], pos[0,:,1], pos[0,:,2], c='blue', s=50, alpha=0.8)
    
    quivers = None
    arrow_scale = 0.1 * (max_vals - min_vals).mean() if ori is not None else None
    
    if ori is not None:
        if D == 3:
            quivers = ax.quiver(pos[0,:,0], pos[0,:,1], pos[0,:,2],
                              ori[0,:,0]*arrow_scale, ori[0,:,1]*arrow_scale, ori[0,:,2]*arrow_scale,
                              length=arrow_scale, normalize=True, color='red')
        else:
            quivers = ax.quiver(pos[0,:,0], pos[0,:,1],
                               ori[0,:,0]*arrow_scale, ori[0,:,1]*arrow_scale,
                               color='red', scale_units='xy', scale=1/arrow_scale)

    def update(frame):
        """Update function for animation"""
        # Update positions
        if D == 2:
            scat.set_offsets(pos[frame])
        else:
            scat._offsets3d = (pos[frame,:,0], pos[frame,:,1], pos[frame,:,2])
            
        # Update orientations if available
        if ori is not None and quivers is not None:
            if D == 3:
                # Remove previous quivers
                for artist in ax.collections[:]:
                    if artist not in [scat]:
                        artist.remove()
                # Add new quivers
                ax.quiver(pos[frame,:,0], 
                          pos[frame,:,1], 
                          pos[frame,:,2],
                          ori[frame,:,0]*arrow_scale, 
                          ori[frame,:,1]*arrow_scale, 
                          ori[frame,:,2]*arrow_scale,
                          length=arrow_scale, 
                          normalize=True, 
                          color='red')
            else:
                # Update 2D quivers in place
                quivers.set_offsets(pos[frame])
                quivers.set_UVC(ori[frame,:,0]*arrow_scale, ori[frame,:,1]*arrow_scale)
        
        ax.set_title(f"{title}\nFrame {frame}/{T}")
        return scat,

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    plt.close()
    return ani


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

def generate_test_data(dim=2, num_particles=10, num_frames=100):
    """Generate test data with predictable patterns"""
    t = np.linspace(0, 4*np.pi, num_frames)
    
    if dim == 2:
        # Circular motion with tangent orientations
        pos = np.zeros((num_frames, num_particles, 2))
        ori = np.zeros_like(pos)
        
        for i in range(num_particles):
            radius = 1 + 0.2*i
            phase = 2*np.pi*i/num_particles
            
            # Positions: rotating circles
            pos[:,i,0] = radius * np.cos(t + phase)
            pos[:,i,1] = radius * np.sin(t + phase)
            
            # Orientations: tangent to the circle
            ori[:,i,0] = -np.sin(t + phase)  # x-component of tangent
            ori[:,i,1] = np.cos(t + phase)   # y-component of tangent
            
    elif dim == 3:
        # Helical motion with spiral orientations
        pos = np.zeros((num_frames, num_particles, 3))
        ori = np.zeros_like(pos)
        
        for i in range(num_particles):
            radius = 1 + 0.2*i
            phase = 2*np.pi*i/num_particles
            z_speed = 0.5
            
            # Positions: rotating helices
            pos[:,i,0] = radius * np.cos(t + phase)
            pos[:,i,1] = radius * np.sin(t + phase)
            pos[:,i,2] = z_speed * t
            
            # Orientations: tangent to the helix
            ori[:,i,0] = -np.sin(t + phase)  # x-component
            ori[:,i,1] = np.cos(t + phase)   # y-component
            ori[:,i,2] = z_speed            # z-component
            
            # Normalize orientations
            norm = np.linalg.norm(ori[:,i,:], axis=1, keepdims=True)
            ori[:,i,:] /= norm
            
    return pos, ori


if __name__ == "__main__":
    # Generate and visualize test data
    pos_2d, ori_2d = generate_test_data(dim=2)
    pos_3d, ori_3d = generate_test_data(dim=3)

    # Create animations
    anim_2d = animate_particle_motion(pos_2d, ori_2d, title="2D Circular Motion Test")
    anim = animate_particle_motion(pos_3d, ori_3d, title="3D Helical Motion Test")

    anim.save("3d_test.mp4", writer="ffmpeg", fps=30)
    anim_2d.save("2d_test.mp4", writer="ffmpeg", fps=30)
    # Display animations
    #HTML(anim_2d.to_html5_video())
    #HTML(anim_3d.to_html5_video())

# To save to files:
# anim_2d.save("2d_test.mp4", writer="ffmpeg", fps=30)
# anim_3d.save("3d_test.mp4", writer="ffmpeg", fps=30)