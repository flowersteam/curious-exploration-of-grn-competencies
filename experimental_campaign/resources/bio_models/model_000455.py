import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([10.0, 0.76876151899652, 0.05625738310526, 4.23123848100348, 1.0, 1.0, 0.0, 0.0, 0.0])
y_indexes = {'y1': 0, 'x2': 1, 'x1': 2, 'x3': 3, 'y4': 4, 'y5': 5, 'y2': 6, 'y3': 7, 'y6': 8}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1.0, 10.0, 1.0, 10.0, 1.0, 50.0, 1.0, 10.0, 1.0, 0.0]) 
c_indexes = {'cell': 0, 'v1_e1': 1, 'v1_p1': 2, 'v2_e2': 3, 'v2_p2': 4, 'v3_e3': 5, 'v3_p3': 6, 'v4_e4': 7, 'v4_p4': 8, 'v5_e5': 9, 'v5_p5': 10}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, -1.0, 0.0], [1.0, -1.0, 0.0, 0.0, -1.0], [0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[1] * (c[2] * (y[0]/1.0) * (y[1]/1.0) - (y[2]/1.0) * (y[3]/1.0)) / (1 + (y[0]/1.0) + (y[1]/1.0) + (y[2]/1.0) + (y[3]/1.0) + (y[0]/1.0) * (y[1]/1.0) + (y[2]/1.0) * (y[3]/1.0))


	def v2(self, y, w, c, t):
		return c[3] * (c[4] * (y[4]/1.0) * (y[3]/1.0) - (y[5]/1.0) * (y[1]/1.0)) / (1 + (y[3]/1.0) + (y[1]/1.0) + (y[4]/1.0) + (y[5]/1.0) + (y[3]/1.0) * (y[4]/1.0) + (y[1]/1.0) * (y[5]/1.0))


	def v3(self, y, w, c, t):
		return c[5] * (c[6] * (y[2]/1.0) - (y[6]/1.0)) / (1 + (y[2]/1.0) + (y[6]/1.0))


	def v4(self, y, w, c, t):
		return c[7] * (c[8] * (y[2]/1.0) - (y[7]/1.0)) / (1 + (y[2]/1.0) + (y[7]/1.0))


	def v5(self, y, w, c, t):
		return c[9] * c[10] * (y[3]/1.0)

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

	def __init__(self, y_indexes={'y1': 0, 'x2': 1, 'x1': 2, 'x3': 3, 'y4': 4, 'y5': 5, 'y2': 6, 'y3': 7, 'y6': 8}, w_indexes={}, c_indexes={'cell': 0, 'v1_e1': 1, 'v1_p1': 2, 'v2_e2': 3, 'v2_p2': 4, 'v3_e3': 5, 'v3_p3': 6, 'v4_e4': 7, 'v4_p4': 8, 'v5_e5': 9, 'v5_p5': 10}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([10.0, 0.76876151899652, 0.05625738310526, 4.23123848100348, 1.0, 1.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 1.0, 10.0, 1.0, 10.0, 1.0, 50.0, 1.0, 10.0, 1.0, 0.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

