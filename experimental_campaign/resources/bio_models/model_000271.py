import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([516.0, 2030.19, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'EpoR': 0, 'Epo': 1, 'Epo_EpoR': 2, 'Epo_EpoRi': 3, 'dEpoi': 4, 'dEpoe': 5}

w0 = jnp.array([2030.19, 0.0])
w_indexes = {'Epo_medium': 0, 'Epo_cells': 1}

c = jnp.array([0.0329366, 516.0, 0.00010496, 0.0172135, 0.0748267, 0.00993805, 0.00317871, 0.0164042, 1.0, 1.0, 1.0]) 
c_indexes = {'kt': 0, 'Bmax': 1, 'kon': 2, 'koff': 3, 'ke': 4, 'kex': 5, 'kdi': 6, 'kde': 7, 'medium': 8, 'cellsurface': 9, 'cell': 10}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.reaction_1(y, w, c, t), self.reaction_2(y, w, c, t), self.reaction_3(y, w, c, t), self.reaction_4(y, w, c, t), self.reaction_5(y, w, c, t), self.reaction_6(y, w, c, t), self.reaction_7(y, w, c, t), self.reaction_8(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def reaction_1(self, y, w, c, t):
		return c[0] * c[1] * c[10]


	def reaction_2(self, y, w, c, t):
		return c[0] * (y[0]/1.0) * c[10]


	def reaction_3(self, y, w, c, t):
		return c[2] * (y[1]/1.0) * (y[0]/1.0) * c[10]


	def reaction_4(self, y, w, c, t):
		return c[3] * (y[2]/1.0) * c[10]


	def reaction_5(self, y, w, c, t):
		return c[4] * (y[2]/1.0) * c[10]


	def reaction_6(self, y, w, c, t):
		return c[5] * (y[3]/1.0) * c[10]


	def reaction_7(self, y, w, c, t):
		return c[6] * (y[3]/1.0) * c[10]


	def reaction_8(self, y, w, c, t):
		return c[7] * (y[3]/1.0) * c[10]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[1]/1.0) + (y[5]/1.0)))

		w = w.at[1].set(((y[3]/1.0) + (y[4]/1.0)))

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

	def __init__(self, y_indexes={'EpoR': 0, 'Epo': 1, 'Epo_EpoR': 2, 'Epo_EpoRi': 3, 'dEpoi': 4, 'dEpoe': 5}, w_indexes={'Epo_medium': 0, 'Epo_cells': 1}, c_indexes={'kt': 0, 'Bmax': 1, 'kon': 2, 'koff': 3, 'ke': 4, 'kex': 5, 'kdi': 6, 'kde': 7, 'medium': 8, 'cellsurface': 9, 'cell': 10}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([516.0, 2030.19, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([2030.19, 0.0]), c=jnp.array([0.0329366, 516.0, 0.00010496, 0.0172135, 0.0748267, 0.00993805, 0.00317871, 0.0164042, 1.0, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

