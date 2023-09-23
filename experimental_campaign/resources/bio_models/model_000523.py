import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([93.0, 0.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 236.0, 0.0, 973.0, 0.0, 0.0, 5178.0, 0.0, 0.0])
y_indexes = {'FADD': 0, 'DISC': 1, 'p55free': 2, 'DISCp55': 3, 'p30': 4, 'p43': 5, 'p18': 6, 'p18inactive': 7, 'Bid': 8, 'tBid': 9, 'PrNES_mCherry': 10, 'PrNES': 11, 'mCherry': 12, 'PrER_mGFP': 13, 'PrER': 14, 'mGFP': 15}

w0 = jnp.array([59.961266081670196])
w_indexes = {'CD95act': 0}

c = jnp.array([116.0, 16.6, 0.000811711012144556, 0.00566528253772301, 0.000491828591049766, 0.0114186392006403, 0.000446994772958953, 0.00343995957326369, 0.0949914492651531, 0.00052867403363568, 0.00152252549827479, 8.98496674617627, 15.421878766215, 1.0]) 
c_indexes = {'CD95': 0, 'CD95L': 1, 'kon_FADD': 2, 'koff_FADD': 3, 'kDISC': 4, 'kD216': 5, 'kD374trans_p55': 6, 'kD374trans_p43': 7, 'kdiss_p18': 8, 'kBid': 9, 'kD374probe': 10, 'KDR': 11, 'KDL': 12, 'cell': 13}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.reaction_1(y, w, c, t), self.reaction_2(y, w, c, t), self.reaction_3(y, w, c, t), self.reaction_4(y, w, c, t), self.reaction_5(y, w, c, t), self.reaction_6(y, w, c, t), self.reaction_7(y, w, c, t), self.reaction_8(y, w, c, t), self.reaction_9(y, w, c, t), self.reaction_10(y, w, c, t), self.reaction_11(y, w, c, t), self.reaction_12(y, w, c, t), self.reaction_13(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def reaction_1(self, y, w, c, t):
		return c[2] * w[0] * (y[0]/1.0) * c[13]


	def reaction_2(self, y, w, c, t):
		return c[3] * (y[1]/1.0) * c[13]


	def reaction_3(self, y, w, c, t):
		return c[4] * (y[2]/1.0) * (y[1]/1.0) * c[13]


	def reaction_4(self, y, w, c, t):
		return c[5] * (y[3]/1.0) * c[13]


	def reaction_5(self, y, w, c, t):
		return c[5] * (y[5]/1.0) * c[13]


	def reaction_6(self, y, w, c, t):
		return c[6] * (y[3]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[13]


	def reaction_7(self, y, w, c, t):
		return c[7] * (y[3]/1.0) * (y[5]/1.0) * c[13]


	def reaction_8(self, y, w, c, t):
		return c[6] * (y[4]/1.0) * ((y[3]/1.0) + (y[4]/1.0)) * c[13]


	def reaction_9(self, y, w, c, t):
		return c[7] * (y[4]/1.0) * (y[5]/1.0) * c[13]


	def reaction_10(self, y, w, c, t):
		return c[8] * (y[6]/1.0) * c[13]


	def reaction_11(self, y, w, c, t):
		return c[9] * (y[8]/1.0) * ((y[5]/1.0) + (y[6]/1.0)) * c[13]


	def reaction_12(self, y, w, c, t):
		return c[10] * (y[10]/1.0) * ((y[5]/1.0) + (y[6]/1.0)) * c[13]


	def reaction_13(self, y, w, c, t):
		return c[10] * (y[13]/1.0) * (y[6]/1.0) * c[13]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((c[0]/1.0)**3 * c[12]**2 * (c[1]/1.0) / (((c[1]/1.0) + c[12]) * ((c[0]/1.0)**2 * c[12]**2 + c[11] * (c[1]/1.0)**2 + 2 * c[11] * c[12] * (c[1]/1.0) + c[11] * c[12]**2))))

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

	def __init__(self, y_indexes={'FADD': 0, 'DISC': 1, 'p55free': 2, 'DISCp55': 3, 'p30': 4, 'p43': 5, 'p18': 6, 'p18inactive': 7, 'Bid': 8, 'tBid': 9, 'PrNES_mCherry': 10, 'PrNES': 11, 'mCherry': 12, 'PrER_mGFP': 13, 'PrER': 14, 'mGFP': 15}, w_indexes={'CD95act': 0}, c_indexes={'CD95': 0, 'CD95L': 1, 'kon_FADD': 2, 'koff_FADD': 3, 'kDISC': 4, 'kD216': 5, 'kD374trans_p55': 6, 'kD374trans_p43': 7, 'kdiss_p18': 8, 'kBid': 9, 'kD374probe': 10, 'KDR': 11, 'KDL': 12, 'cell': 13}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([93.0, 0.0, 155.0, 0.0, 0.0, 0.0, 0.0, 0.0, 236.0, 0.0, 973.0, 0.0, 0.0, 5178.0, 0.0, 0.0]), w0=jnp.array([59.961266081670196]), c=jnp.array([116.0, 16.6, 0.000811711012144556, 0.00566528253772301, 0.000491828591049766, 0.0114186392006403, 0.000446994772958953, 0.00343995957326369, 0.0949914492651531, 0.00052867403363568, 0.00152252549827479, 8.98496674617627, 15.421878766215, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

