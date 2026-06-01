# Track Parameter Resolutions

Analytical estimation of tracking performance in cylindrical-shaped silicon detectors using covariance propagation and multiple-scattering models.

This project provides a lightweight framework for studying how detector geometry, sensor resolution, material budget, magnetic field strength, and track properties affect:

* Track position resolution
* Momentum resolution
* Extrapolation uncertainty
* Multiple scattering contributions

The repository was originally developed to gain an intuitive understanding of track reconstruction performance and to reproduce the main effects that drive detector design decisions.

---

# Physical Model

The framework models a charged particle traversing a layered silicon detector inside a magnetic field.

For each detector layer, the following quantities are specified:

* Radial position
* Material thickness
* Transverse hit resolution
* Longitudinal hit resolution

From this the following features can be calculated:

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

The accompanying notebook demonstrates the use of the Python code.
The notebook is intended to be educational and can be read sequentially together with a paper or other literature on analytical tracking-performance estimation.

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



