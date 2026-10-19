"""Feedback controllers for the UUV depth-control system."""


class PDController:
    """Discrete-time proportional-derivative (PD) controller.

    Computes u[t] = Kp * e[t] + Kd * (e[t] - e[t-1]),
    where e[t] = r[t] - y[t] is the depth tracking error.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.previous_error = 0.0  # e[-1] = 0, as specified

    def reset(self) -> None:
        """Clear stored state so a new simulation starts from e[-1] = 0."""
        self.previous_error = 0.0

    def compute_action(self, reference: float, observation: float) -> float:
        """Return the control action for one time step.

        Args:
            reference: target depth r[t].
            observation: measured depth y[t].
        """
        error = reference - observation
        action = self.kp * error + self.kd * (error - self.previous_error)
        self.previous_error = error
        return action