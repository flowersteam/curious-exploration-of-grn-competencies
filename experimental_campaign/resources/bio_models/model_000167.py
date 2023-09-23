import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 14.625, 0.0, 0.0])
y_indexes = {'Pstat_sol': 0, 'PstatDimer_sol': 1, 'Pstat_nuc': 2, 'stat_nuc': 3, 'stat_sol': 4, 'species_test': 5, 'PstatDimer_nuc': 6}

w0 = jnp.array([0.0])
w_indexes = {'statKinase_sol': 0}

c = jnp.array([0.05, 1.0, 14.625, 1.0, 0.6, 0.03, 1.0, 2.0, 1.0, 4.0, 0.6, 0.03, 0.045, 0.3, -0.06, 0.6, 0.003, 3.0]) 
c_indexes = {'statPhosphatase_nuc': 0, 'nuc': 1, 'sol': 2, 'nm': 3, 'PstatDimerisation_Kf_PstatDimerisation': 4, 'PstatDimerisation_Kr_PstatDimerisation': 5, 'statDephosphorylation_Kcat_dephos': 6, 'statDephosphorylation_Km_dephos': 7, 'statPhosphorylation_Kcat_phos': 8, 'statPhosphorylation_Km_phos': 9, 'PstatDimerisationNuc_Kf_PstatDimerisation': 10, 'PstatDimerisationNuc_Kr_PstatDimerisation': 11, 'PstatDimer__import_PstatDimer_impMax': 12, 'PstatDimer__import_Kpsd_imp': 13, 'stat_export_stat_expMax': 14, 'stat_export_Ks_exp': 15, 'stat_import_stat_impMax': 16, 'stat_import_Ks_imp': 17}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, -2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.PstatDimerisation(y, w, c, t), self.statDephosphorylation(y, w, c, t), self.statPhosphorylation(y, w, c, t), self.PstatDimerisationNuc(y, w, c, t), self.PstatDimer__import(y, w, c, t), self.stat_export(y, w, c, t), self.stat_import(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def PstatDimerisation(self, y, w, c, t):
		return (c[4] * (y[0]/14.625)**2 + -(c[5] * (y[1]/14.625))) * c[2]


	def statDephosphorylation(self, y, w, c, t):
		return c[6] * (c[0]/1.0) * (y[2]/1.0) * (1 / (c[7] + (y[2]/1.0))) * c[1]


	def statPhosphorylation(self, y, w, c, t):
		return c[8] * (w[0]/14.625) * (y[4]/14.625) * (1 / (c[9] + (y[4]/14.625))) * c[2]


	def PstatDimerisationNuc(self, y, w, c, t):
		return (c[10] * (y[2]/1.0)**2 + -(c[11] * (y[6]/1.0))) * c[1]


	def PstatDimer__import(self, y, w, c, t):
		return c[12] * (y[1]/14.625) * (1 / (c[13] + (y[1]/14.625))) * c[3]


	def stat_export(self, y, w, c, t):
		return c[1] * c[14] * (y[3]/1.0) * (1 / (c[15] + (y[3]/1.0))) * c[3]


	def stat_import(self, y, w, c, t):
		return c[1] * c[16] * (y[4]/14.625) * (1 / (c[17] + (y[4]/14.625))) * c[3]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(14.625 * (jaxfuncs.piecewise(0.01 * jnp.sin(0.001571 * (-500 + t)), (t > 500) & (t < 2502.54614894971), 0)))

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

	def __init__(self, y_indexes={'Pstat_sol': 0, 'PstatDimer_sol': 1, 'Pstat_nuc': 2, 'stat_nuc': 3, 'stat_sol': 4, 'species_test': 5, 'PstatDimer_nuc': 6}, w_indexes={'statKinase_sol': 0}, c_indexes={'statPhosphatase_nuc': 0, 'nuc': 1, 'sol': 2, 'nm': 3, 'PstatDimerisation_Kf_PstatDimerisation': 4, 'PstatDimerisation_Kr_PstatDimerisation': 5, 'statDephosphorylation_Kcat_dephos': 6, 'statDephosphorylation_Km_dephos': 7, 'statPhosphorylation_Kcat_phos': 8, 'statPhosphorylation_Km_phos': 9, 'PstatDimerisationNuc_Kf_PstatDimerisation': 10, 'PstatDimerisationNuc_Kr_PstatDimerisation': 11, 'PstatDimer__import_PstatDimer_impMax': 12, 'PstatDimer__import_Kpsd_imp': 13, 'stat_export_stat_expMax': 14, 'stat_export_Ks_exp': 15, 'stat_import_stat_impMax': 16, 'stat_import_Ks_imp': 17}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 14.625, 0.0, 0.0]), w0=jnp.array([0.0]), c=jnp.array([0.05, 1.0, 14.625, 1.0, 0.6, 0.03, 1.0, 2.0, 1.0, 4.0, 0.6, 0.03, 0.045, 0.3, -0.06, 0.6, 0.003, 3.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

