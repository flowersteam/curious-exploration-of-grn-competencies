import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 68190.1837333797, 0.0, 0.0, 0.0433090165709309, 0.0, 0.0, 3.54316740542218, 0.0, 0.0, 68190.1837333797])
y_indexes = {'EGF': 0, 'EGFR': 1, 'EGF_EGFR': 2, 'pEGFR': 3, 'Akt': 4, 'pEGFR_Akt': 5, 'pAkt': 6, 'S6': 7, 'pAkt_S6': 8, 'pS6': 9, 'pro_EGFR': 10}

w0 = jnp.array([0.0, 0.0, 0.0, 0.0])
w_indexes = {'EGF': 0, 'pEGFR_total': 1, 'pAkt_total': 2, 'pS6_total': 3}

c = jnp.array([0.000181734813832032, 60.0587507734138, 49886.2313741851, 0.0, 0.0, 30.0, 0.000106386129269658, 60.0, 3600.0, 1.0, 0.00673816, 0.040749, 1.5543e-05, 0.00517473, 0.0305684, 0.0997194, 2.10189e-06, 5.1794e-15, 0.00121498, 0.0327962, 0.00113102, 0.0192391]) 
c_indexes = {'pEGFR_scaleFactor': 0, 'pAkt_scaleFactor': 1, 'pS6_scaleFactor': 2, 'EGF_conc_step': 3, 'EGF_conc_impulse': 4, 'EGF_conc_ramp': 5, 'EGFR_turnover': 6, 'pulse_time': 7, 'ramp_time': 8, 'Cell': 9, 'reaction_1_k1': 10, 'reaction_1_k2': 11, 'reaction_2_k1': 12, 'reaction_2_k2': 13, 'reaction_3_k1': 14, 'reaction_4_k1': 15, 'reaction_5_k1': 16, 'reaction_5_k2': 17, 'reaction_6_k1': 18, 'reaction_7_k1': 19, 'reaction_8_k1': 20, 'reaction_10_k1': 21}

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
		w = w.at[0].set(1.0 * (c[3] + jaxfuncs.piecewise(c[4], t <= c[7], 0) + c[5] * t / c[8]))

		w = w.at[1].set((((y[3]/1.0) + (y[5]/1.0)) * c[0]))

		w = w.at[2].set((((y[6]/1.0) + (y[8]/1.0)) * c[1]))

		w = w.at[3].set(((y[9]/1.0) * c[2]))

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

	def __init__(self, y_indexes={'EGF': 0, 'EGFR': 1, 'EGF_EGFR': 2, 'pEGFR': 3, 'Akt': 4, 'pEGFR_Akt': 5, 'pAkt': 6, 'S6': 7, 'pAkt_S6': 8, 'pS6': 9, 'pro_EGFR': 10}, w_indexes={'EGF': 0, 'pEGFR_total': 1, 'pAkt_total': 2, 'pS6_total': 3}, c_indexes={'pEGFR_scaleFactor': 0, 'pAkt_scaleFactor': 1, 'pS6_scaleFactor': 2, 'EGF_conc_step': 3, 'EGF_conc_impulse': 4, 'EGF_conc_ramp': 5, 'EGFR_turnover': 6, 'pulse_time': 7, 'ramp_time': 8, 'Cell': 9, 'reaction_1_k1': 10, 'reaction_1_k2': 11, 'reaction_2_k1': 12, 'reaction_2_k2': 13, 'reaction_3_k1': 14, 'reaction_4_k1': 15, 'reaction_5_k1': 16, 'reaction_5_k2': 17, 'reaction_6_k1': 18, 'reaction_7_k1': 19, 'reaction_8_k1': 20, 'reaction_10_k1': 21}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 68190.1837333797, 0.0, 0.0, 0.0433090165709309, 0.0, 0.0, 3.54316740542218, 0.0, 0.0, 68190.1837333797]), w0=jnp.array([0.0, 0.0, 0.0, 0.0]), c=jnp.array([0.000181734813832032, 60.0587507734138, 49886.2313741851, 0.0, 0.0, 30.0, 0.000106386129269658, 60.0, 3600.0, 1.0, 0.00673816, 0.040749, 1.5543e-05, 0.00517473, 0.0305684, 0.0997194, 2.10189e-06, 5.1794e-15, 0.00121498, 0.0327962, 0.00113102, 0.0192391]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

