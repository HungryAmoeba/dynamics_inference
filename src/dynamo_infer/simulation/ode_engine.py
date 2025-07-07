"""ODE simulation engine using diffrax."""

import jax.numpy as jnp
from typing import Tuple
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Tsit5
from .base import Simulator
from ..config.schemas import SimulationConfig
from ..dynamics.base import DynamicalSystem


class ODESimulator(Simulator):
    """ODE simulator using diffrax for robust integration."""

    def __init__(self, system: DynamicalSystem, config: SimulationConfig):
        super().__init__(system, config)

    def run(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the ODE simulation.

        Returns:
            (trajectory, times) where:
            - trajectory: (T, state_dim) array of states over time
            - times: (T,) array of time points
        """
        print(f"Running ODE simulation with {self.config.solver}")
        print(
            f"Time: {self.config.time.t0} to {self.config.time.t1}, dt={self.config.time.dt}"
        )

        # Create vector field function
        def vector_field(t, y, args):
            return self.system.compute_derivatives(t, y, args)

        # Create ODE term
        term = ODETerm(vector_field)

        # Choose solver
        if self.config.solver.lower() == "tsit5":
            solver = Tsit5()
        else:
            # Default to Tsit5
            solver = Tsit5()

        # Set up time parameters
        t0 = self.config.time.t0
        t1 = self.config.time.t1
        dt0 = self.config.time.dt

        # Set up save points
        save_t0 = getattr(self.config.saveat, "t0", t0)
        save_t1 = getattr(self.config.saveat, "t1", t1)
        save_dt = getattr(self.config.saveat, "dt", dt0)
        saveat = SaveAt(ts=list(jnp.arange(save_t0, save_t1 + save_dt, save_dt)))

        # Get initial state
        y0 = self.system.return_state()

        # Set tolerances - ensure they are scalars
        rtol = float(getattr(self.config, "rtol", 1e-4))
        atol = float(getattr(self.config, "atol", 1e-7))

        # Run the ODE solver
        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            max_steps=None,
            stepsize_controller=PIDController(rtol=rtol, atol=atol),
        )

        # Store results
        self.results = (sol.ys, sol.ts)

        return sol.ys, sol.ts
