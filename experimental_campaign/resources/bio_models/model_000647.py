import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([2.0, 2.5, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 3.0, 0.0])
y_indexes = {'Raf1': 0, 'RKIP': 1, 'Raf1_RKIP': 2, 'ERKPP': 3, 'Raf1_RKIP_ERKPP': 4, 'ERK': 5, 'RKIPP': 6, 'MEKPP': 7, 'MEKPP_ERK': 8, 'RP': 9, 'RKIPP_RP': 10}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.53, 0.0072, 0.625, 0.00245, 0.0315, 0.8, 0.0075, 0.071, 0.92, 0.00122, 0.87, 1.0]) 
c_indexes = {'k1': 0, 'k2': 1, 'k3': 2, 'k4': 3, 'k5': 4, 'k6': 5, 'k7': 6, 'k8': 7, 'k9': 8, 'k10': 9, 'k11': 10, 'cytoplasm': 11}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Raf1_RKIP_complex_formation(y, w, c, t), self.Raf1_RKIP_complex_disassembly(y, w, c, t), self.Raf1_RKIP_ERKPP_complex_formation(y, w, c, t), self.Raf1_RKIP_ERKPP_complex_disassembly__ERK_phosphorylation(y, w, c, t), self.Raf1_RKIP_ERKPP_complex_disassembly__RKIP_phosphorylation(y, w, c, t), self.MEKPP_ERK_complex_formation(y, w, c, t), self.MEKPP_ERK_complex_disassembly__ERK_unphosphorylated(y, w, c, t), self.MEKPP_ERK_complex_disassembly__ERK_phosphorylated(y, w, c, t), self.RKIPP_RP_comlex_formation(y, w, c, t), self.RKIPP_RP_complex_disassembly__phosphorylated_RKIP(y, w, c, t), self.RKIPP_RP_complex_disassembly__unphosphorylated_RKIP(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Raf1_RKIP_complex_formation(self, y, w, c, t):
		return c[11] * c[0] * (y[0]/1.0) * (y[1]/1.0)


	def Raf1_RKIP_complex_disassembly(self, y, w, c, t):
		return c[11] * c[1] * (y[2]/1.0)


	def Raf1_RKIP_ERKPP_complex_formation(self, y, w, c, t):
		return c[11] * c[2] * (y[2]/1.0) * (y[3]/1.0)


	def Raf1_RKIP_ERKPP_complex_disassembly__ERK_phosphorylation(self, y, w, c, t):
		return c[11] * c[3] * (y[4]/1.0)


	def Raf1_RKIP_ERKPP_complex_disassembly__RKIP_phosphorylation(self, y, w, c, t):
		return c[11] * c[4] * (y[4]/1.0)


	def MEKPP_ERK_complex_formation(self, y, w, c, t):
		return c[11] * c[5] * (y[5]/1.0) * (y[7]/1.0)


	def MEKPP_ERK_complex_disassembly__ERK_unphosphorylated(self, y, w, c, t):
		return c[11] * c[6] * (y[8]/1.0)


	def MEKPP_ERK_complex_disassembly__ERK_phosphorylated(self, y, w, c, t):
		return c[11] * c[7] * (y[8]/1.0)


	def RKIPP_RP_comlex_formation(self, y, w, c, t):
		return c[11] * c[8] * (y[6]/1.0) * (y[9]/1.0)


	def RKIPP_RP_complex_disassembly__phosphorylated_RKIP(self, y, w, c, t):
		return c[11] * c[9] * (y[10]/1.0)


	def RKIPP_RP_complex_disassembly__unphosphorylated_RKIP(self, y, w, c, t):
		return c[11] * c[10] * (y[10]/1.0)

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

	def __init__(self, y_indexes={'Raf1': 0, 'RKIP': 1, 'Raf1_RKIP': 2, 'ERKPP': 3, 'Raf1_RKIP_ERKPP': 4, 'ERK': 5, 'RKIPP': 6, 'MEKPP': 7, 'MEKPP_ERK': 8, 'RP': 9, 'RKIPP_RP': 10}, w_indexes={}, c_indexes={'k1': 0, 'k2': 1, 'k3': 2, 'k4': 3, 'k5': 4, 'k6': 5, 'k7': 6, 'k8': 7, 'k9': 8, 'k10': 9, 'k11': 10, 'cytoplasm': 11}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([2.0, 2.5, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 3.0, 0.0]), w0=jnp.array([]), c=jnp.array([0.53, 0.0072, 0.625, 0.00245, 0.0315, 0.8, 0.0075, 0.071, 0.92, 0.00122, 0.87, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

