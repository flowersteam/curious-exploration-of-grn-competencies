import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.75444831412765, 43.8146704098797, 27.4558812768926])
y_indexes = {'N0': 0, 'N1': 1, 'N2': 2}

w0 = jnp.array([73.02500000089995])
w_indexes = {'T': 0}

c = jnp.array([0.1, 0.218, 1.0, 2.92408052354609, 0.0999999999999998, 0.263, 0.547, 1.0, 29.2408052354609, 0.239254806051979, 1.83, 1.0]) 
c_indexes = {'d0': 0, 'b0': 1, 'c0': 2, 'm0': 3, 'a0': 4, 'd1': 5, 'b1': 6, 'c1': 7, 'm1': 8, 'a1': 9, 'd2': 10, 'compartment': 11}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.R0X(y, w, c, t), self.R01(y, w, c, t), self.R00(y, w, c, t), self.R1X(y, w, c, t), self.R12(y, w, c, t), self.R11(y, w, c, t), self.R2X(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def R0X(self, y, w, c, t):
		return c[0] * (y[0]/1.0)


	def R01(self, y, w, c, t):
		return (c[1] + c[2] * (y[0]/1.0) / ((y[0]/1.0) + c[3])) * (y[0]/1.0)


	def R00(self, y, w, c, t):
		return c[4] * (y[0]/1.0)


	def R1X(self, y, w, c, t):
		return c[5] * (y[1]/1.0)


	def R12(self, y, w, c, t):
		return (c[6] + c[7] * (y[1]/1.0) / ((y[1]/1.0) + c[8])) * (y[1]/1.0)


	def R11(self, y, w, c, t):
		return c[9] * (y[1]/1.0)


	def R2X(self, y, w, c, t):
		return c[10] * (y[2]/1.0)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[0]/1.0) + (y[1]/1.0) + (y[2]/1.0)))

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

	def __init__(self, y_indexes={'N0': 0, 'N1': 1, 'N2': 2}, w_indexes={'T': 0}, c_indexes={'d0': 0, 'b0': 1, 'c0': 2, 'm0': 3, 'a0': 4, 'd1': 5, 'b1': 6, 'c1': 7, 'm1': 8, 'a1': 9, 'd2': 10, 'compartment': 11}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([1.75444831412765, 43.8146704098797, 27.4558812768926]), w0=jnp.array([73.02500000089995]), c=jnp.array([0.1, 0.218, 1.0, 2.92408052354609, 0.0999999999999998, 0.263, 0.547, 1.0, 29.2408052354609, 0.239254806051979, 1.83, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

