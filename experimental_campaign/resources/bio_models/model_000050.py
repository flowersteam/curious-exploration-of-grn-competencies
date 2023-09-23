import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'DFG': 0, 'E1': 1, 'E2': 2, 'Gly': 3, 'Cn': 4, '_3DG': 5, 'FA': 6, '_1DG': 7, 'AA': 8, 'Man': 9, 'Glu': 10, 'Mel': 11, 'MG': 12, 'Fru': 13}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 0.0057, 0.0156, 0.0155, 0.0794, 0.0907, 0.0274, 0.2125, 0.0, 1.9085, 0.0707, 0.1131, 0.0008, 0.0022, 0.0034, 0.0159, 0.0134]) 
c_indexes = {'compartment': 0, 'v1_k1': 1, 'v2_k2': 2, 'v3_k3': 3, 'v4_k4': 4, 'v5_k5': 5, 'v6_k6': 6, 'v7_k7': 7, 'v8_k8': 8, 'v9_k9': 9, 'v10_k10': 10, 'v11_k11': 11, 'v12_k12': 12, 'v13_k13': 13, 'v14_k14': 14, 'v15_k15': 15, 'v16_k16': 16}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t), self.v6(y, w, c, t), self.v7(y, w, c, t), self.v8(y, w, c, t), self.v9(y, w, c, t), self.v10(y, w, c, t), self.v11(y, w, c, t), self.v12(y, w, c, t), self.v13(y, w, c, t), self.v14(y, w, c, t), self.v15(y, w, c, t), self.v16(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[1] * (y[0]/1.0)


	def v2(self, y, w, c, t):
		return c[2] * (y[0]/1.0)


	def v3(self, y, w, c, t):
		return c[3] * (y[0]/1.0)


	def v4(self, y, w, c, t):
		return c[4] * (y[1]/1.0)


	def v5(self, y, w, c, t):
		return c[5] * (y[5]/1.0)


	def v6(self, y, w, c, t):
		return c[6] * (y[5]/1.0)


	def v7(self, y, w, c, t):
		return c[7] * (y[2]/1.0)


	def v8(self, y, w, c, t):
		return c[8] * (y[7]/1.0)


	def v9(self, y, w, c, t):
		return c[9] * (y[7]/1.0)


	def v10(self, y, w, c, t):
		return c[10] * (y[1]/1.0)


	def v11(self, y, w, c, t):
		return c[11] * (y[1]/1.0)


	def v12(self, y, w, c, t):
		return c[12] * (y[9]/1.0)


	def v13(self, y, w, c, t):
		return c[13] * (y[10]/1.0)


	def v14(self, y, w, c, t):
		return c[14] * (y[4]/1.0) * (y[3]/1.0)


	def v15(self, y, w, c, t):
		return c[15] * (y[4]/1.0)


	def v16(self, y, w, c, t):
		return c[16] * (y[2]/1.0)

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

	def __init__(self, y_indexes={'DFG': 0, 'E1': 1, 'E2': 2, 'Gly': 3, 'Cn': 4, '_3DG': 5, 'FA': 6, '_1DG': 7, 'AA': 8, 'Man': 9, 'Glu': 10, 'Mel': 11, 'MG': 12, 'Fru': 13}, w_indexes={}, c_indexes={'compartment': 0, 'v1_k1': 1, 'v2_k2': 2, 'v3_k3': 3, 'v4_k4': 4, 'v5_k5': 5, 'v6_k6': 6, 'v7_k7': 7, 'v8_k8': 8, 'v9_k9': 9, 'v10_k10': 10, 'v11_k11': 11, 'v12_k12': 12, 'v13_k13': 13, 'v14_k14': 14, 'v15_k15': 15, 'v16_k16': 16}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 0.0057, 0.0156, 0.0155, 0.0794, 0.0907, 0.0274, 0.2125, 0.0, 1.9085, 0.0707, 0.1131, 0.0008, 0.0022, 0.0034, 0.0159, 0.0134]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

