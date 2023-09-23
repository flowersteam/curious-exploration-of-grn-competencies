import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([7.38, 15.4, 0.28, 40.4, 2.78])
y_indexes = {'auxin': 0, 'TIR1': 1, 'auxinTIR1': 2, 'VENUS': 3, 'auxinTIR1VENUS': 4}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.334, 0.000822, 0.79, 4.49, 0.175, 1.15, 0.486, 0.00316, 18.5, 30.5, 1.0]) 
c_indexes = {'kd': 0, 'ka': 1, 'mu': 2, 'ld': 3, 'lm': 4, 'la': 5, 'delta': 6, 'lambda': 7, 'TIR1T': 8, 'alpha_tr': 9, 'cell': 10}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.auxin_TIR1association(y, w, c, t), self.auxin_TIR1dissociation(y, w, c, t), self.auxin_TIR1_VENUSassociation(y, w, c, t), self.auxin_TIR1_VENUSdissociation(y, w, c, t), self.auxin_TIR1_VENUSdissociationleadingtoubiquitination(y, w, c, t), self.auxinproduction(y, w, c, t), self.auxindecay(y, w, c, t), self.VENUSproduction(y, w, c, t), self.VENUSphotobleachingdecay(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def auxin_TIR1association(self, y, w, c, t):
		return c[1] * (y[0]/1.0) * (y[1]/1.0)


	def auxin_TIR1dissociation(self, y, w, c, t):
		return c[0] * (y[2]/1.0)


	def auxin_TIR1_VENUSassociation(self, y, w, c, t):
		return c[5] * (y[2]/1.0) * (y[3]/1.0)


	def auxin_TIR1_VENUSdissociation(self, y, w, c, t):
		return c[3] * (y[4]/1.0)


	def auxin_TIR1_VENUSdissociationleadingtoubiquitination(self, y, w, c, t):
		return c[4] * (y[4]/1.0)


	def auxinproduction(self, y, w, c, t):
		return c[9]


	def auxindecay(self, y, w, c, t):
		return c[2] * (y[0]/1.0)


	def VENUSproduction(self, y, w, c, t):
		return c[6]


	def VENUSphotobleachingdecay(self, y, w, c, t):
		return c[7] * (y[3]/1.0)

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

	def __init__(self, y_indexes={'auxin': 0, 'TIR1': 1, 'auxinTIR1': 2, 'VENUS': 3, 'auxinTIR1VENUS': 4}, w_indexes={}, c_indexes={'kd': 0, 'ka': 1, 'mu': 2, 'ld': 3, 'lm': 4, 'la': 5, 'delta': 6, 'lambda': 7, 'TIR1T': 8, 'alpha_tr': 9, 'cell': 10}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([7.38, 15.4, 0.28, 40.4, 2.78]), w0=jnp.array([]), c=jnp.array([0.334, 0.000822, 0.79, 4.49, 0.175, 1.15, 0.486, 0.00316, 18.5, 30.5, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

