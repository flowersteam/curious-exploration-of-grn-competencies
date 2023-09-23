import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0])
y_indexes = {'Glu': 0, 'Fru': 1, 'C5': 2, 'Formic_acid': 3, 'Triose': 4, 'Cn': 5, 'Acetic_acid': 6, 'lys_R': 7, 'Amadori': 8, 'AMP': 9, 'Melanoidin': 10}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 0.01, 0.00509, 0.00047, 0.0011, 0.00712, 0.00439, 0.00018, 0.11134, 0.14359, 0.00015, 0.12514]) 
c_indexes = {'compartment': 0, '_J1_K1': 1, '_J2_K2': 2, '_J3_K3': 3, '_J4_K4': 4, '_J5_K5': 5, '_J6_K6': 6, '_J7_K7': 7, '_J8_K8': 8, '_J9_K9': 9, '_J10_K10': 10, '_J11_K11': 11}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self._J1(y, w, c, t), self._J2(y, w, c, t), self._J3(y, w, c, t), self._J4(y, w, c, t), self._J5(y, w, c, t), self._J6(y, w, c, t), self._J7(y, w, c, t), self._J8(y, w, c, t), self._J9(y, w, c, t), self._J10(y, w, c, t), self._J11(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def _J1(self, y, w, c, t):
		return c[1] * (y[0]/1.0)


	def _J2(self, y, w, c, t):
		return c[2] * (y[1]/1.0)


	def _J3(self, y, w, c, t):
		return c[3] * (y[0]/1.0)


	def _J4(self, y, w, c, t):
		return c[4] * (y[1]/1.0)


	def _J5(self, y, w, c, t):
		return c[5] * (y[1]/1.0)


	def _J6(self, y, w, c, t):
		return c[6] * (y[4]/1.0)


	def _J7(self, y, w, c, t):
		return c[7] * (y[0]/1.0) * (y[7]/1.0)


	def _J8(self, y, w, c, t):
		return c[8] * (y[8]/1.0)


	def _J9(self, y, w, c, t):
		return c[9] * (y[8]/1.0)


	def _J10(self, y, w, c, t):
		return c[10] * (y[1]/1.0) * (y[7]/1.0)


	def _J11(self, y, w, c, t):
		return c[11] * (y[9]/1.0)

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

	def __init__(self, y_indexes={'Glu': 0, 'Fru': 1, 'C5': 2, 'Formic_acid': 3, 'Triose': 4, 'Cn': 5, 'Acetic_acid': 6, 'lys_R': 7, 'Amadori': 8, 'AMP': 9, 'Melanoidin': 10}, w_indexes={}, c_indexes={'compartment': 0, '_J1_K1': 1, '_J2_K2': 2, '_J3_K3': 3, '_J4_K4': 4, '_J5_K5': 5, '_J6_K6': 6, '_J7_K7': 7, '_J8_K8': 8, '_J9_K9': 9, '_J10_K10': 10, '_J11_K11': 11}, atol=1e-06, rtol=1e-12, mxstep=1000):

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
	def __call__(self, n_steps, y0=jnp.array([160.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 0.01, 0.00509, 0.00047, 0.0011, 0.00712, 0.00439, 0.00018, 0.11134, 0.14359, 0.00015, 0.12514]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

