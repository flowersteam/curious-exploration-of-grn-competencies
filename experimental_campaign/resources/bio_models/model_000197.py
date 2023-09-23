import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([88.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'x1': 0, 'x3': 1, 'x5': 2, 'x2': 3, 'x4': 4}

w0 = jnp.array([88.0, 0.0])
w_indexes = {'BSP_tot': 0, 'BSP_cell': 1}

c = jnp.array([0.0025, 0.0784, 0.0013, 0.0827, 0.0091, 6.4e-05, 0.0397, 1000.0, 0.0098, 1.6, 1000.0, 0.0003, 1.0, 1.0, 1.5]) 
c_indexes = {'p1': 0, 'p2': 1, 'p3': 2, 'p4': 3, 'p5': 4, 'p6': 5, 'p7': 6, 'p8': 7, 'p9': 8, 'p10': 9, 'p11': 10, 'p12': 11, 'basolat': 12, 'cell': 13, 'apical': 14}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0], [1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.OATP1B3(y, w, c, t), self.ABCC2(y, w, c, t), self.endo_in_bl(y, w, c, t), self.endo_ex_bl(y, w, c, t), self.endo_ex_ap(y, w, c, t), self.bl_BSP_binding(y, w, c, t), self.bl_BSP_dissoc(y, w, c, t), self.cellular_BSP_binding(y, w, c, t), self.cellular_BSP_dissoc(y, w, c, t), self.paracell_transp(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def OATP1B3(self, y, w, c, t):
		return c[0] * y[0]


	def ABCC2(self, y, w, c, t):
		return c[1] * y[1]


	def endo_in_bl(self, y, w, c, t):
		return c[2] * y[0]


	def endo_ex_bl(self, y, w, c, t):
		return c[3] * y[1]


	def endo_ex_ap(self, y, w, c, t):
		return c[4] * y[1]


	def bl_BSP_binding(self, y, w, c, t):
		return c[5] * y[0] * (c[7] - y[3])


	def bl_BSP_dissoc(self, y, w, c, t):
		return c[6] * y[3]


	def cellular_BSP_binding(self, y, w, c, t):
		return c[8] * y[1] * (c[10] - y[4])


	def cellular_BSP_dissoc(self, y, w, c, t):
		return c[9] * y[4]


	def paracell_transp(self, y, w, c, t):
		return c[11] * (y[0] / c[12] - y[2] / c[14])

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set((y[0] + y[3] + y[1] + y[4] + y[2]))

		w = w.at[1].set((y[1] + y[4]))

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

	def __init__(self, y_indexes={'x1': 0, 'x3': 1, 'x5': 2, 'x2': 3, 'x4': 4}, w_indexes={'BSP_tot': 0, 'BSP_cell': 1}, c_indexes={'p1': 0, 'p2': 1, 'p3': 2, 'p4': 3, 'p5': 4, 'p6': 5, 'p7': 6, 'p8': 7, 'p9': 8, 'p10': 9, 'p11': 10, 'p12': 11, 'basolat': 12, 'cell': 13, 'apical': 14}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([88.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([88.0, 0.0]), c=jnp.array([0.0025, 0.0784, 0.0013, 0.0827, 0.0091, 6.4e-05, 0.0397, 1000.0, 0.0098, 1.6, 1000.0, 0.0003, 1.0, 1.0, 1.5]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

