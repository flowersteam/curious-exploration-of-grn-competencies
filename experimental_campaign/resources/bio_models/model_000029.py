import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([800.0, 0.0, 0.0, 0.0])
y_indexes = {'M': 0, 'MpY': 1, 'Mpp': 2, 'MpT': 3}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([180.0, 100.0, 410.0, 1.08, 40.0, 0.007, 20.0, 0.008, 300.0, 0.45, 22.0, 0.084, 18.0, 0.06, 34.0, 0.108, 40.0, 1.0]) 
c_indexes = {'MEK': 0, 'MKP3': 1, 'Km1': 2, 'kcat1': 3, 'Km2': 4, 'kcat2': 5, 'Km3': 6, 'kcat3': 7, 'Km4': 8, 'kcat4': 9, 'Km5': 10, 'kcat5': 11, 'Km6': 12, 'kcat6': 13, 'Km7': 14, 'kcat7': 15, 'Km8': 16, 'cell': 17}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t), self.v6(y, w, c, t), self.v7(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[17] * (c[3] * (c[0]/1.0) * (y[0]/1.0) / c[2] / (1 + (y[0]/1.0) * ((c[2] + c[6]) / (c[2] * c[6])) + (y[1]/1.0) / c[4] + (y[3]/1.0) / c[8]))


	def v2(self, y, w, c, t):
		return c[17] * (c[5] * (c[0]/1.0) * (y[1]/1.0) / c[4] / (1 + (y[0]/1.0) * ((c[2] + c[6]) / (c[2] * c[6])) + (y[1]/1.0) / c[4] + (y[3]/1.0) / c[8]))


	def v3(self, y, w, c, t):
		return c[17] * (c[7] * (c[0]/1.0) * (y[0]/1.0) / c[6] / (1 + (y[0]/1.0) * ((c[2] + c[6]) / (c[2] * c[6])) + (y[1]/1.0) / c[4] + (y[3]/1.0) / c[8]))


	def v4(self, y, w, c, t):
		return c[17] * (c[9] * (c[0]/1.0) * (y[3]/1.0) / c[8] / (1 + (y[0]/1.0) * ((c[2] + c[6]) / (c[2] * c[6])) + (y[1]/1.0) / c[4] + (y[3]/1.0) / c[8]))


	def v5(self, y, w, c, t):
		return c[17] * (c[11] * (c[1]/1.0) * (y[2]/1.0) / c[10] / (1 + (y[2]/1.0) / c[10] + (y[3]/1.0) / c[12] + (y[1]/1.0) / c[14] + (y[0]/1.0) / c[16]))


	def v6(self, y, w, c, t):
		return c[17] * (c[13] * (c[1]/1.0) * (y[3]/1.0) / c[12] / (1 + (y[2]/1.0) / c[10] + (y[3]/1.0) / c[12] + (y[1]/1.0) / c[14] + (y[0]/1.0) / c[16]))


	def v7(self, y, w, c, t):
		return c[17] * (c[15] * (c[1]/1.0) * (y[1]/1.0) / c[14] / (1 + (y[2]/1.0) / c[10] + (y[3]/1.0) / c[12] + (y[1]/1.0) / c[14] + (y[0]/1.0) / c[16]))

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
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

	def __init__(self, y_indexes={'M': 0, 'MpY': 1, 'Mpp': 2, 'MpT': 3}, w_indexes={}, c_indexes={'MEK': 0, 'MKP3': 1, 'Km1': 2, 'kcat1': 3, 'Km2': 4, 'kcat2': 5, 'Km3': 6, 'kcat3': 7, 'Km4': 8, 'kcat4': 9, 'Km5': 10, 'kcat5': 11, 'Km6': 12, 'kcat6': 13, 'Km7': 14, 'kcat7': 15, 'Km8': 16, 'cell': 17}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([800.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([180.0, 100.0, 410.0, 1.08, 40.0, 0.007, 20.0, 0.008, 300.0, 0.45, 22.0, 0.084, 18.0, 0.06, 34.0, 0.108, 40.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

