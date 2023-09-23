import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([90.0, 0.0, 127.0, 0.0, 0.0, 0.0, 0.0, 0.0, 224.0, 0.0, 1909.0, 0.0, 0.0, 3316.0, 0.0, 0.0])
y_indexes = {'FADD': 0, 'DISC': 1, 'p55free': 2, 'DISCp55': 3, 'p30': 4, 'p43': 5, 'p18': 6, 'p18inactive': 7, 'Bid': 8, 'tBid': 9, 'PrNES_mCherry': 10, 'PrNES': 11, 'mCherry': 12, 'PrER_mGFP': 13, 'PrER': 14, 'mGFP': 15}

w0 = jnp.array([2.182477492149729])
w_indexes = {'CD95act': 0}

c = jnp.array([12.0, 16.6, 0.00108871858684363, 0.00130854998177646, 0.000364965874405544, 0.00639775937416746, 0.000223246421372882, 5.29906975294056e-05, 0.000644612643975149, 0.000543518631342483, 0.00413530054938906, 0.064713651554491, 0.00052134055139547, 0.00153710001025539, 57.2050013008496, 30.0060394758199, 1.0]) 
c_indexes = {'CD95': 0, 'CD95L': 1, 'kon_FADD': 2, 'koff_FADD': 3, 'kDISC': 4, 'kD216': 5, 'kD216trans_p55': 6, 'kD216trans_p43': 7, 'kD374': 8, 'kD374trans_p55': 9, 'kD374trans_p43': 10, 'kdiss_p18': 11, 'kBid': 12, 'kD374probe': 13, 'KDR': 14, 'KDL': 15, 'cell': 16}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.reaction_1(y, w, c, t), self.reaction_2(y, w, c, t), self.reaction_3(y, w, c, t), self.reaction_4(y, w, c, t), self.reaction_5(y, w, c, t), self.reaction_6(y, w, c, t), self.reaction_7(y, w, c, t), self.reaction_8(y, w, c, t), self.reaction_9(y, w, c, t), self.reaction_10(y, w, c, t), self.reaction_11(y, w, c, t), self.reaction_12(y, w, c, t), self.reaction_13(y, w, c, t), self.reaction_14(y, w, c, t), self.reaction_15(y, w, c, t), self.reaction_16(y, w, c, t), self.reaction_17(y, w, c, t), self.reaction_18(y, w, c, t), self.reaction_19(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def reaction_1(self, y, w, c, t):
		return c[2] * w[0] * (y[0]/1.0) * c[16]


	def reaction_2(self, y, w, c, t):
		return c[3] * (y[1]/1.0) * c[16]


	def reaction_3(self, y, w, c, t):
		return c[4] * (y[2]/1.0) * (y[1]/1.0) * c[16]


	def reaction_4(self, y, w, c, t):
		return c[5] * (y[3]/1.0) * c[16]


	def reaction_5(self, y, w, c, t):
		return c[6] * (y[3]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[16]


	def reaction_6(self, y, w, c, t):
		return c[7] * (y[3]/1.0) * (y[5]/1.0) * c[16]


	def reaction_7(self, y, w, c, t):
		return c[5] * (y[5]/1.0) * c[16]


	def reaction_8(self, y, w, c, t):
		return c[6] * (y[5]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[16]


	def reaction_9(self, y, w, c, t):
		return c[7] * (y[5]/1.0) * (y[5]/1.0) * c[16]


	def reaction_10(self, y, w, c, t):
		return c[8] * (y[3]/1.0) * c[16]


	def reaction_11(self, y, w, c, t):
		return c[9] * (y[3]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[16]


	def reaction_12(self, y, w, c, t):
		return c[10] * (y[3]/1.0) * (y[5]/1.0) * c[16]


	def reaction_13(self, y, w, c, t):
		return c[8] * (y[4]/1.0) * c[16]


	def reaction_14(self, y, w, c, t):
		return c[9] * (y[4]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[16]


	def reaction_15(self, y, w, c, t):
		return c[10] * (y[4]/1.0) * (y[5]/1.0) * c[16]


	def reaction_16(self, y, w, c, t):
		return c[11] * (y[6]/1.0) * c[16]


	def reaction_17(self, y, w, c, t):
		return c[12] * (y[8]/1.0) * ((y[5]/1.0) + (y[6]/1.0)) * c[16]


	def reaction_18(self, y, w, c, t):
		return c[13] * (y[10]/1.0) * ((y[5]/1.0) + (y[6]/1.0)) * c[16]


	def reaction_19(self, y, w, c, t):
		return c[13] * (y[13]/1.0) * (y[6]/1.0) * c[16]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((c[0]/1.0)**3 * c[15]**2 * (c[1]/1.0) / (((c[1]/1.0) + c[15]) * ((c[0]/1.0)**2 * c[15]**2 + c[14] * (c[1]/1.0)**2 + 2 * c[14] * c[15] * (c[1]/1.0) + c[14] * c[15]**2))))

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

	def __init__(self, y_indexes={'FADD': 0, 'DISC': 1, 'p55free': 2, 'DISCp55': 3, 'p30': 4, 'p43': 5, 'p18': 6, 'p18inactive': 7, 'Bid': 8, 'tBid': 9, 'PrNES_mCherry': 10, 'PrNES': 11, 'mCherry': 12, 'PrER_mGFP': 13, 'PrER': 14, 'mGFP': 15}, w_indexes={'CD95act': 0}, c_indexes={'CD95': 0, 'CD95L': 1, 'kon_FADD': 2, 'koff_FADD': 3, 'kDISC': 4, 'kD216': 5, 'kD216trans_p55': 6, 'kD216trans_p43': 7, 'kD374': 8, 'kD374trans_p55': 9, 'kD374trans_p43': 10, 'kdiss_p18': 11, 'kBid': 12, 'kD374probe': 13, 'KDR': 14, 'KDL': 15, 'cell': 16}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([90.0, 0.0, 127.0, 0.0, 0.0, 0.0, 0.0, 0.0, 224.0, 0.0, 1909.0, 0.0, 0.0, 3316.0, 0.0, 0.0]), w0=jnp.array([2.182477492149729]), c=jnp.array([12.0, 16.6, 0.00108871858684363, 0.00130854998177646, 0.000364965874405544, 0.00639775937416746, 0.000223246421372882, 5.29906975294056e-05, 0.000644612643975149, 0.000543518631342483, 0.00413530054938906, 0.064713651554491, 0.00052134055139547, 0.00153710001025539, 57.2050013008496, 30.0060394758199, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

