import numpy as np
import polyscope as ps
import time

ps.init()

# generate some points
points = np.random.rand(100, 3)

# generate a time series of points (n_timesteps, n_points, 3)
n_timesteps = 100
points_t = np.zeros((n_timesteps, 100, 3))
velocity_t = np.random.rand(100, 3)
stepsize = 10
for i in range(n_timesteps):
    points_t[i] = points + i / stepsize * velocity_t

t = 0
while t < n_timesteps:
    # visualize!
    ps_cloud = ps.register_point_cloud("my points", points_t[t])
    ps.frame_tick()
    t += 1
    # sleep for a second
    time.sleep(0.1)


# # visualize!
# ps_cloud = ps.register_point_cloud("my points", points)
# ps.show()

# # with some options
# ps_cloud_opt = ps.register_point_cloud("my points", points,
#                                        radius=0.02, point_render_mode='quad')
# ps.show()
