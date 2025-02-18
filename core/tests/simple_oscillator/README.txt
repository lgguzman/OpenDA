
Introduction

This model implement a simple oscillator model. Because of its small size
(2 state variables) and linear behaviour, this model is ideal as a first
test for assimilation methods. Al should converge quickly to the right 
values.

simple linear oscilator (e.g. mass-spring system with friction)
 d(x)/d(t) = u
 d(u)/d(t) = - omega^2 * x - (2/t_damp) u

Calibration experiment

Observations are generated with the 'true' values for the parameters,
i.e. t_damp=9.0, omega=1.7 . There is no noise added to initial condition,
system forcing or observations. 

Almost all calibration experiments start with t_damp=8.0 and omega=1.5708. 
The following values are found with the present settings:

experiment			t_damp		omega	#evaluations	cost
--------------------------------------------------------------------------------
true				9.0		1.7
initial				8.0		1.5708
--------------------------------------------------------------------------------
dud no constraint		8.998		1.700	 9		5.99E-7
simplex no constraint		8.869		1.701	35		0.0057
powell no constraint		9.000		1.699	89		5.78E-14;
--------------------------------------------------------------------------------
dud weak constraint			8.192		1.699	13		1.243
simplex constraint		8.670		1.701	37		1.195
powell constraint		8.641		1.700	95		1.192

Experiment Dud_parameterConstraints in started from t_damp=8.0 and omega=1.9. 
For these settings the Dud algorithm will produce values t_damp < 0.0, which 
is physically incorrect and results in an NaN number for the cost function 
(though the algorithm does converge to the correct values). With 
parameterConstraints="true" bounds for the values of t_damp and omega may be
specified that prevent the algorithm to try unphysical values. 

experiment			t_damp		omega	#evaluations	cost
--------------------------------------------------------------------------------
true				9.0		1.7
initial				8.0		1.9
--------------------------------------------------------------------------------
dud no constraint		8.972		1.700	 20		1.525E-3
dud_parameterConstraint 9.071		1.699	 11		2.579E-3
--------------------------------------------------------------------------------


Kalman filtering experiment

These experiments use the same parameters as the true model. The only difference
is the stochastic forcing. No noise is added to the observations. The systemnoise 
is rather strong, so filtering should help much.
