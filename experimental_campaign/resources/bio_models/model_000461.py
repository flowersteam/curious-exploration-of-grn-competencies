import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([100.0, 0.0, 0.0, 0.0])
y_indexes = {'IPTG': 0, 'sigb': 1, 'lacz': 2, 'x': 3}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.044, 100.0, 9.0, 0.76, 0.041, 0.0004, 1.0]) 
c_indexes = {'kbd': 0, 'kbs': 1, 'kxd': 2, 'kxs': 3, 'kzd': 4, 'kzs': 5, 'compartment': 6}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0], [1.0, -1.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return (y[0]/1.0) * c[1] - c[0] * (y[1]/1.0)


	def v2(self, y, w, c, t):
		return -(c[4] * (y[2]/1.0)) + c[5] * (y[1]/1.0) / (1 + (y[3]/1.0))


	def v3(self, y, w, c, t):
		return -(c[2] * (y[3]/1.0)) + c[3] * (y[1]/1.0) / (1 + (y[3]/1.0))

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

	def __init__(self, y_indexes={'IPTG': 0, 'sigb': 1, 'lacz': 2, 'x': 3}, w_indexes={}, c_indexes={'kbd': 0, 'kbs': 1, 'kxd': 2, 'kxs': 3, 'kzd': 4, 'kzs': 5, 'compartment': 6}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([100.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([0.044, 100.0, 9.0, 0.76, 0.041, 0.0004, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

