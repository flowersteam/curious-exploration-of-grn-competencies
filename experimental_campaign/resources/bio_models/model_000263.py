import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 8.52065090518276, 0.0, 0.0, 1.15594897919397, 0.0, 0.0, 3.552336039555, 0.0, 0.0, 8.52065090518276])
y_indexes = {'NGF': 0, 'TrkA': 1, 'NGF_TrkA': 2, 'pTrkA': 3, 'Akt': 4, 'pTrkA_Akt': 5, 'pAkt': 6, 'S6': 7, 'pAkt_S6': 8, 'pS6': 9, 'pro_TrkA': 10}

w0 = jnp.array([0.0, 0.0, 0.0, 0.0])
w_indexes = {'pS6_total': 0, 'pAkt_total': 1, 'pTrkA_total': 2, 'NGF': 3}

c = jnp.array([0.848783474941268, 2.42381211094508, 0.525842718263069, 0.0, 0.0, 30.0, 0.0011032440769796, 60.0, 3600.0, 1.0, 0.00269408, 0.0133747, 0.0882701, 1.47518e-10, 0.0202517, 0.0684084, 68.3666, 5.23519, 0.0056515, 1.28135, 0.000293167, 0.00833178]) 
c_indexes = {'pTrkA_scaleFactor': 0, 'pAkt_scaleFactor': 1, 'pS6_scaleFactor': 2, 'NGF_conc_step': 3, 'NGF_conc_pulse': 4, 'NGF_conc_ramp': 5, 'TrkA_turnover': 6, 'pulse_time': 7, 'ramp_time': 8, 'Cell': 9, 'reaction_1_k1': 10, 'reaction_1_k2': 11, 'reaction_2_k1': 12, 'reaction_2_k2': 13, 'reaction_3_k1': 14, 'reaction_4_k1': 15, 'reaction_5_k1': 16, 'reaction_5_k2': 17, 'reaction_6_k1': 18, 'reaction_7_k1': 19, 'reaction_8_k1': 20, 'reaction_10_k1': 21}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.reaction_1(y, w, c, t), self.reaction_2(y, w, c, t), self.reaction_3(y, w, c, t), self.reaction_4(y, w, c, t), self.reaction_5(y, w, c, t), self.reaction_6(y, w, c, t), self.reaction_7(y, w, c, t), self.reaction_8(y, w, c, t), self.reaction_9(y, w, c, t), self.reaction_10(y, w, c, t), self.reaction_11(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def reaction_1(self, y, w, c, t):
		return c[9] * (c[10] * (y[0]/1.0) * (y[1]/1.0) - c[11] * (y[2]/1.0))


	def reaction_2(self, y, w, c, t):
		return c[9] * (c[12] * (y[3]/1.0) * (y[4]/1.0) - c[13] * (y[5]/1.0))


	def reaction_3(self, y, w, c, t):
		return c[9] * c[14] * (y[5]/1.0)


	def reaction_4(self, y, w, c, t):
		return c[9] * c[15] * (y[3]/1.0)


	def reaction_5(self, y, w, c, t):
		return c[9] * (c[16] * (y[6]/1.0) * (y[7]/1.0) - c[17] * (y[8]/1.0))


	def reaction_6(self, y, w, c, t):
		return c[9] * c[18] * (y[8]/1.0)


	def reaction_7(self, y, w, c, t):
		return c[9] * c[19] * (y[6]/1.0)


	def reaction_8(self, y, w, c, t):
		return c[9] * c[20] * (y[9]/1.0)


	def reaction_9(self, y, w, c, t):
		return c[9] * c[6] * (y[10]/1.0)


	def reaction_10(self, y, w, c, t):
		return c[9] * c[21] * (y[2]/1.0)


	def reaction_11(self, y, w, c, t):
		return c[9] * c[6] * (y[1]/1.0)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[9]/1.0) * c[2]))

		w = w.at[1].set((((y[6]/1.0) + (y[8]/1.0)) * c[1]))

		w = w.at[2].set((((y[3]/1.0) + (y[5]/1.0)) * c[0]))

		w = w.at[3].set(1.0 * (c[3] + jaxfuncs.piecewise(c[4], t <= c[7], 0) + c[5] * t / c[8]))

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

	def __init__(self, y_indexes={'NGF': 0, 'TrkA': 1, 'NGF_TrkA': 2, 'pTrkA': 3, 'Akt': 4, 'pTrkA_Akt': 5, 'pAkt': 6, 'S6': 7, 'pAkt_S6': 8, 'pS6': 9, 'pro_TrkA': 10}, w_indexes={'pS6_total': 0, 'pAkt_total': 1, 'pTrkA_total': 2, 'NGF': 3}, c_indexes={'pTrkA_scaleFactor': 0, 'pAkt_scaleFactor': 1, 'pS6_scaleFactor': 2, 'NGF_conc_step': 3, 'NGF_conc_pulse': 4, 'NGF_conc_ramp': 5, 'TrkA_turnover': 6, 'pulse_time': 7, 'ramp_time': 8, 'Cell': 9, 'reaction_1_k1': 10, 'reaction_1_k2': 11, 'reaction_2_k1': 12, 'reaction_2_k2': 13, 'reaction_3_k1': 14, 'reaction_4_k1': 15, 'reaction_5_k1': 16, 'reaction_5_k2': 17, 'reaction_6_k1': 18, 'reaction_7_k1': 19, 'reaction_8_k1': 20, 'reaction_10_k1': 21}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 8.52065090518276, 0.0, 0.0, 1.15594897919397, 0.0, 0.0, 3.552336039555, 0.0, 0.0, 8.52065090518276]), w0=jnp.array([0.0, 0.0, 0.0, 0.0]), c=jnp.array([0.848783474941268, 2.42381211094508, 0.525842718263069, 0.0, 0.0, 30.0, 0.0011032440769796, 60.0, 3600.0, 1.0, 0.00269408, 0.0133747, 0.0882701, 1.47518e-10, 0.0202517, 0.0684084, 68.3666, 5.23519, 0.0056515, 1.28135, 0.000293167, 0.00833178]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

