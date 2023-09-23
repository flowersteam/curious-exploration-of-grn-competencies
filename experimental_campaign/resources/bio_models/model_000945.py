import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([10.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'L_m': 0, 'H_m': 1, 'L_c': 2, 'H_c': 3, 'L_n': 4}

w0 = jnp.array([6.031363088057901, 0.39324487334137515])
w_indexes = {'v_1': 0, 'v_2': 1}

c = jnp.array([0.0289, 0.000309, 1.014, 0.026553, 0.18637, 4.4489, 0.000106, 0.00085341, 28.9, 2000000000000.0, 829.0, 326.0, 10.0, 1.0]) 
c_indexes = {'k_0_m': 0, 'k_i': 1, 'k_e': 2, 'k_o_c': 3, 'k_c_c': 4, 'k_d': 5, 'k_c_m': 6, 'k_b': 7, 'B_T': 8, 'V_m': 9, 'V_c': 10, 'V_n': 11, 'D': 12, 'human_lymphoma_cells': 13}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateL_m(y, w, c, t), self.RateH_m(y, w, c, t), self.RateL_c(y, w, c, t), self.RateH_c(y, w, c, t), self.RateL_n(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateL_m(self, y, w, c, t):
		return -(c[0] + c[1]) * (y[0]/1.0) + c[6] * (y[1]/1.0) + c[2] * w[0] * (y[2]/1.0)

	def RateH_m(self, y, w, c, t):
		return c[0] * (y[0]/1.0) - c[6] * (y[1]/1.0)

	def RateL_c(self, y, w, c, t):
		return ((c[1] * w[0] * (y[0]/1.0) - (c[2] + c[3]) * (y[2]/1.0)) + c[4] * (y[3]/1.0) - c[7] * (c[8] - (y[4]/1.0)) * (y[2]/1.0)) + w[1] * c[5] * (y[4]/1.0)

	def RateH_c(self, y, w, c, t):
		return c[3] * (y[2]/1.0) - c[4] * (y[3]/1.0)

	def RateL_n(self, y, w, c, t):
		return c[7] * w[1] * (c[8] - (y[4]/1.0)) * (y[2]/1.0) - c[5] * (y[4]/1.0)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((c[9] / (c[10] * 400000)) * 0.001))

		w = w.at[1].set((c[11] / c[10]))

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

	def __init__(self, y_indexes={'L_m': 0, 'H_m': 1, 'L_c': 2, 'H_c': 3, 'L_n': 4}, w_indexes={'v_1': 0, 'v_2': 1}, c_indexes={'k_0_m': 0, 'k_i': 1, 'k_e': 2, 'k_o_c': 3, 'k_c_c': 4, 'k_d': 5, 'k_c_m': 6, 'k_b': 7, 'B_T': 8, 'V_m': 9, 'V_c': 10, 'V_n': 11, 'D': 12, 'human_lymphoma_cells': 13}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([10.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([6.031363088057901, 0.39324487334137515]), c=jnp.array([0.0289, 0.000309, 1.014, 0.026553, 0.18637, 4.4489, 0.000106, 0.00085341, 28.9, 2000000000000.0, 829.0, 326.0, 10.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

