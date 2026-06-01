# Track Parameter Resolutions

Analytical estimation of tracking performance in cylindrical-shaped silicon detectors using covariance propagation and multiple-scattering models.

This project provides a lightweight framework for studying how detector geometry, sensor resolution, material budget, magnetic field strength, and track properties affect:

* Track position resolution
* Momentum resolution
* Extrapolation uncertainty
* Multiple scattering contributions

The repository was originally developed to gain an intuitive understanding of track reconstruction performance and to reproduce the main effects that drive detector design decisions.

---

# Motivation

Modern tracking detectors are often evaluated using full detector simulations. While these simulations provide realistic results, they can make it difficult to understand *why* a particular resolution is obtained.

This project takes a complementary approach:

Instead of simulating every detector response in detail, it uses analytical covariance propagation to isolate the dominant effects:

1. Finite detector hit resolution
2. Multiple Coulomb scattering
3. Detector geometry
4. Magnetic field strength
5. Number of measured space points

The goal is not maximum realism but maximum insight.

---

# Physical Model

The framework models a charged particle traversing a layered silicon detector inside a magnetic field.

For each detector layer, the following quantities are specified:

* Radial position
* Material thickness
* Transverse hit resolution
* Longitudinal hit resolution

From these inputs, the code constructs covariance matrices describing:

## Detector Measurement Uncertainty

The detector contribution is represented by a diagonal covariance matrix

[
C_{\mathrm{det}}
]

whose entries are given by the intrinsic sensor resolutions.

## Multiple Scattering

Material interactions are modeled using the Highland formula

[
\sigma_{\theta}
===============

\frac{13,\mathrm{MeV}}{\beta p}
\sqrt{\frac{x}{X_0}}
\left(
1 + 0.038\ln\frac{x}{X_0}
\right)
]

which is used to construct a covariance matrix

[
C_{\mathrm{MS}}
]

describing correlated scattering effects between detector layers.

## Track Model

Two track parameterizations are used:

### Longitudinal Projection

Straight-line model

[
z(r)=a+br
]

### Transverse Projection

Parabolic approximation of a curved trajectory

[
r\phi(r)=a+br+\frac{c}{2}r^2
]

which approximates the circular motion induced by the magnetic field.

---

# Covariance Propagation

Given the track model matrix

[
G
]

and a measurement covariance matrix

[
C
]

the parameter covariance matrix is obtained via

[
C_{\mathrm{par}}
================

\left(
G^T C^{-1} G
\right)^{-1}
]

From this matrix the framework derives:

* Position uncertainties
* Extrapolation uncertainties
* Momentum resolution

at arbitrary radii.

---

# Features

## Position Resolution Studies

Calculate transverse track position uncertainty

```python
detector.transverse_track_position_uncertainty(
    momentum=1.0,
    mass=0.139,
    number_of_hits=7,
    extrapolation_radius=0.01,
    polar_angle=90
)
```

Includes:

* Detector resolution effects
* Multiple scattering in detector layers
* Multiple scattering in air between layers
* Extrapolation uncertainty

---

## Longitudinal Resolution Studies

Calculate uncertainty along the beam direction

```python
detector.longitudinal_track_position_uncertainty(
    momentum=1.0,
    mass=0.139,
    number_of_hits=7,
    extrapolation_radius=0.01,
    polar_angle=90
)
```

---

## Momentum Resolution

Estimate relative transverse momentum resolution

```python
detector.transverse_momentum_reso(
    momentum=1.0,
    mass=0.139,
    number_of_hits=7,
    polar_angle=90
)
```

The momentum resolution is extracted from the curvature parameter covariance obtained from the parabolic fit.

---

# Example Detector Setup

```python
detector = DetectorSetup(
    average_layer_radii,
    layer_thickness,
    detector_resolutions_rphi,
    detector_resolutions_z,
    radiation_length_medium,
    magnetic_field_strength
)
```

where:

| Parameter                 | Description                                     |
| ------------------------- | ----------------------------------------------- |
| average_layer_radii       | Detector layer radii [m]                        |
| layer_thickness           | Material thickness in units of radiation length |
| detector_resolutions_rphi | Transverse hit resolution [m]                   |
| detector_resolutions_z    | Longitudinal hit resolution [m]                 |
| radiation_length_medium   | Radiation length of medium between layers [m]   |
| magnetic_field_strength   | Magnetic field strength [T]                     |

---

# Notebook Walkthrough

The accompanying notebook demonstrates:

1. Construction of detector geometries
2. Definition of material budgets
3. Resolution calculations
4. Momentum dependence studies
5. Multiple-scattering dominated regimes
6. Detector-resolution dominated regimes
7. Interpretation of covariance matrices

The notebook is intended to be educational and can be read sequentially as an introduction to analytical tracking-performance estimation.

---

# Limitations

This framework intentionally uses simplified models.

Not included:

* Full helix fitting
* Energy loss
* Non-uniform magnetic fields
* Detector inefficiencies
* Pattern recognition effects
* Non-Gaussian scattering tails

The objective is analytical understanding rather than full detector simulation.



