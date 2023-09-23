import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 7.13, 41.2, 0.0])
y_indexes = {'C': 0, 'P': 1, 'Q': 2, 'Qp': 3}

w0 = jnp.array([48.330000000000005])
w_indexes = {'Pstar': 0}

c = jnp.array([7.13, 41.2, 0.121, 0.00295, 0.0031, 0.0087, 0.729, 0.24, 100.0, 1.0, 1.0]) 
c_indexes = {'P0': 0, 'Q0': 1, 'lambda_P': 2, 'k_PQ': 3, 'k_Qp_P': 4, 'delta_QP': 5, 'gamma': 6, 'KDE': 7, 'K': 8, 'plama': 9, 'tissue': 10}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateC(y, w, c, t), self.RateP(y, w, c, t), self.RateQ(y, w, c, t), self.RateQp(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateC(self, y, w, c, t):
		return -c[7] * (y[0]/1.0)

	def RateP(self, y, w, c, t):
		return c[2] * (y[1]/1.0) * (1 - w[0] / c[8]) + c[4] * (y[3]/1.0) - c[3] * (y[1]/1.0) - c[6] * (y[0]/1.0) * c[7] * (y[1]/1.0)

	def RateQ(self, y, w, c, t):
		return c[3] - c[6] * (y[0]/1.0) * c[7] * (y[2]/1.0)

	def RateQp(self, y, w, c, t):
		return c[6] * (y[0]/1.0) * c[7] * (y[2]/1.0) - c[4] * (y[3]/1.0) - c[5] * (y[3]/1.0)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[1]/1.0) + (y[2]/1.0) + (y[3]/1.0)))

		return w

class ModelStep(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	atol: float = eqx.static_field()
	rtol: float = eqx.static_field()
	mxstep: int = eqx.static_field()
	assignmentfunc: AssignmentRule

	def __init__(self, y_indexes={'C': 0, 'P': 1, 'Q': 2, 'Qp': 3}, w_indexes={'Pstar': 0}, c_indexes={'P0': 0, 'Q0': 1, 'lambda_P': 2, 'k_PQ': 3, 'k_Qp_P': 4, 'delta_QP': 5, 'gamma': 6, 'KDE': 7, 'K': 8, 'plama': 9, 'tissue': 10}, atol=1e-06, rtol=1e-12, mxstep=1000):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes

		self.ratefunc = RateofSpeciesChange()
		self.rtol = rtol
		self.atol = atol
		self.mxstep = mxstep
		self.assignmentfunc = AssignmentRule()

	@jit
	def __call__(self, y, w, c, t, deltaT):
		y_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]	
		t_new = t + deltaT	
		w_new = self.assignmentfunc(y_new, w, c, t_new)	
		return y_new, w_new, c, t_new	

class ModelRollout(eqx.Module):
	deltaT: float = eqx.static_field()
	modelstepfunc: ModelStep

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=1000):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([1.0, 7.13, 41.2, 0.0]), w0=jnp.array([48.330000000000005]), c=jnp.array([7.13, 41.2, 0.121, 0.00295, 0.0031, 0.0087, 0.729, 0.24, 100.0, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

