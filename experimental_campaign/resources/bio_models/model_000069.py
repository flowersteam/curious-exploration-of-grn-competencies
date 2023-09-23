import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
y_indexes = {'srci': 0, 'srco': 1, 'srca': 2, 'srcc': 3, 'CSK_cytoplasm': 4, 'Cbp_P': 5, 'Cbp_P_CSK': 6, 'PTP': 7, 'PTP_pY789': 8, 'Cbp': 9}

w0 = jnp.array([0.0001, 0.0])
w_indexes = {'src_activity': 0, 'ptp_activity': 1}

c = jnp.array([1.0, 0.8, 1.0, 10.0, 1.0, 1.0, 0.05, 0.15, 0.035, 0.0001, 0.0, 0.1, 0.01, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]) 
c_indexes = {'k1': 0, 'k2': 1, 'k3': 2, 'k4': 3, 'kPTP': 4, 'kCbp': 5, 'p1': 6, 'p2': 7, 'p3': 8, 'src_background': 9, 'PTP_background': 10, 'kCSKon': 11, 'kCSKoff': 12, 'rho_srca': 13, 'rho_srco': 14, 'rho_srcc': 15, 'Kser': 16, 'acsk0': 17, 'default': 18}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.CSK_translocation(y, w, c, t), self.PTP_phosphorylation(y, w, c, t), self.Cbp_phosphorylation(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return (c[1] * w[1] * (y[0]/1.0) - c[0] * (y[6]/1.0) * (y[1]/1.0)) * c[18]


	def v2(self, y, w, c, t):
		return (c[2] * w[0] * (y[1]/1.0) - c[6] * (y[2]/1.0)) * c[18]


	def v3(self, y, w, c, t):
		return (c[0] * (y[6]/1.0) * (y[2]/1.0) - c[1] * w[1] * (y[3]/1.0)) * c[18]


	def v4(self, y, w, c, t):
		return c[18] * c[3] * c[6] * (y[3]/1.0)


	def CSK_translocation(self, y, w, c, t):
		return ((y[5]/1.0) * c[11] * (y[4]/1.0) - c[12] * (y[6]/1.0)) * c[18]


	def PTP_phosphorylation(self, y, w, c, t):
		return c[18] * ((c[4] * w[0] + c[8]) * (y[7]/1.0) - c[7] * (y[8]/1.0))


	def Cbp_phosphorylation(self, y, w, c, t):
		return c[5] * w[0] * (y[9]/1.0) * c[18]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set((c[14] * (y[1]/1.0) + c[13] * (y[2]/1.0) + c[9] + c[15] * (y[3]/1.0)))

		w = w.at[1].set((c[10] + c[16] * (y[8]/1.0)))

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

	def __init__(self, y_indexes={'srci': 0, 'srco': 1, 'srca': 2, 'srcc': 3, 'CSK_cytoplasm': 4, 'Cbp_P': 5, 'Cbp_P_CSK': 6, 'PTP': 7, 'PTP_pY789': 8, 'Cbp': 9}, w_indexes={'src_activity': 0, 'ptp_activity': 1}, c_indexes={'k1': 0, 'k2': 1, 'k3': 2, 'k4': 3, 'kPTP': 4, 'kCbp': 5, 'p1': 6, 'p2': 7, 'p3': 8, 'src_background': 9, 'PTP_background': 10, 'kCSKon': 11, 'kCSKoff': 12, 'rho_srca': 13, 'rho_srco': 14, 'rho_srcc': 15, 'Kser': 16, 'acsk0': 17, 'default': 18}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]), w0=jnp.array([0.0001, 0.0]), c=jnp.array([1.0, 0.8, 1.0, 10.0, 1.0, 1.0, 0.05, 0.15, 0.035, 0.0001, 0.0, 0.1, 0.01, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

