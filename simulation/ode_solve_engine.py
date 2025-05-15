from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Tsit5
import jax.numpy as jnp


class ODEEngine:
    def __init__(self, system, config):
        self.system = system
        self.config = config

    def run(self):
        # make a lambda function to wrap the compute_derivatives method
        # so that it can be used with the ODETerm
        # note that we pass the system as an argument to the lambda function
        # so that we can access the compute_derivatives method

        vector_field = lambda t, y, args: self.system.compute_derivatives(t, y, args)
        # vector_field = lambda t, y, args: -y
        term = ODETerm(vector_field)
        solver = Tsit5()
        # saveat = SaveAt(ts=[0., 1., 2., 3.])
        t0 = self.config.time.t0
        t1 = self.config.time.t1
        dt = self.config.time.dt
        save_t0 = self.config.saveat.t0 if hasattr(self.config.saveat, "t0") else t0
        save_t1 = self.config.saveat.t1 if hasattr(self.config.saveat, "t1") else t1
        save_dt = self.config.saveat.dt if hasattr(self.config.saveat, "dt") else dt
        saveat = SaveAt(ts=list(jnp.arange(save_t0, save_t1, save_dt)))

        y0 = self.system.return_state()  # initial state

        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            saveat=saveat,
            max_steps=None,
            stepsize_controller=PIDController(rtol=1e-4, atol=1e-7),
        )

        return sol.ys, sol.ts
        # print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
        # print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
        # term = ODETerm(vec_field)
        # # choose the solver based on the config
        # if self.config.solver == "Dopri5":
        #     solver = Dopri5()
        # else:
        #     raise ValueError(f"Unknown solver: {self.config.solver}")

        # t0 = self.config.time.t0
        # t1 = self.config.time.t1
        # dt = self.config.time.dt
        # #tspan = (t0, t1)
        # y0 = self.system.return_state()
        # print(f"Initial state y0: {y0}")
        # saveat = SaveAt(ts=jnp.arange(t0, t1, dt))
        # sol = diffeqsolve(
        #     term,
        #     solver,
        #     y0=y0,
        #     t0 = t0,
        #     t1 = t1,
        #     dt0=dt,
        #     saveat=saveat
        # )

    def run_manual(self):
        """
        Run the ODE solver manually, step by step.
        This is useful for debugging and understanding the solver's behavior.
        """
        # solve using rk4 for now, ensure normalization of orientations at each step

        t0 = self.config.time.t0
        t1 = self.config.time.t1
        dt = self.config.time.dt
        y0 = self.system.return_state()

        t = t0
        ys = []
        ts = []
        while t < t1:
            ys.append(y0)
            ts.append(t)
            # compute the derivatives
            dy = self.system.compute_derivatives(t, y0, None)
            # update the state using Euler's method
            y0 = y0 + dt * dy
            # normalize the orientations to lie on the unit sphere
            # y0[self.system.N * self.system.dim:] /= jnp.linalg.norm(y0[self.system.N * self.system.dim:])

            # # ensure that the orientations are normalized
            # orientations = y0[self.system.N * self.system.dim:]
            # orientations_reshaped = orientations.reshape(-1, self.system.dim)
            # norms = jnp.linalg.norm(orientations_reshaped, axis=1)
            # orientations_normalized = orientations_reshaped / norms[:, jnp.newaxis]
            # y0 = jnp.concatenate((y0[:self.system.N * self.system.dim], orientations_normalized.flatten()))

            # check if y0 has any NaN values
            if jnp.isnan(y0).any():
                print(f"NaN detected in state at time {t}")
                import pdb

                pdb.set_trace()

            t += dt

        # convert to jnp arrays
        ys = jnp.array(ys)
        ts = jnp.array(ts)
        return ys, ts
