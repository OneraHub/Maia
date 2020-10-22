#include "maia/cgns_registry/cgns_keywords.hpp"


// Coordinate system
// ------------------
namespace CGNS {
  namespace Name {
    const char* GridCoordinates        = "GridCoordinates";
    const char* CoordinateNames        = "CoordinateNames";
    const char* CoordinateX            = "CoordinateX";
    const char* CoordinateY            = "CoordinateY";
    const char* CoordinateZ            = "CoordinateZ";
    const char* CoordinateR            = "CoordinateR";
    const char* CoordinateTheta        = "CoordinateTheta";
    const char* CoordinatePhi          = "CoordinatePhi";
    const char* CoordinateNormal       = "CoordinateNormal";
    const char* CoordinateTangential   = "CoordinateTangential";
    const char* CoordinateXi           = "CoordinateXi";
    const char* CoordinateEta          = "CoordinateEta";
    const char* CoordinateZeta         = "CoordinateZeta";
    const char* CoordinateTransform    = "CoordinateTransform";
    const char* InterpolantsDonor      = "InterpolantsDonor";
    const char* ElementConnectivity    = "ElementConnectivity";
    const char* ParentData             = "ParentData";
    const char* ParentElements         = "ParentElements";
    const char* ParentElementsPosition = "ParentElementsPosition";
    const char* ElementSizeBoundary    = "ElementSizeBoundary";
  };
};

// FlowSolution Quantities
// ------------------------
// Patterns
namespace CGNS {
  namespace Name {
    const char* VectorX_p                       = "%sX";
    const char* VectorY_p                       = "%sY";
    const char* VectorZ_p                       = "%sZ";
    const char* VectorTheta_p                   = "%sTheta";
    const char* VectorPhi_p                     = "%sPhi";
    const char* VectorMagnitude_p               = "%sMagnitude";
    const char* VectorNormal_p                  = "%sNormal";
    const char* VectorTangential_p              = "%sTangential";
    const char* Potential                       = "Potential";
    const char* StreamFunction                  = "StreamFunction";
    const char* Density                         = "Density";
    const char* Pressure                        = "Pressure";
    const char* Temperature                     = "Temperature";
    const char* EnergyInternal                  = "EnergyInternal";
    const char* Enthalpy                        = "Enthalpy";
    const char* Entropy                         = "Entropy";
    const char* EntropyApprox                   = "EntropyApprox";
    const char* DensityStagnation               = "DensityStagnation";
    const char* PressureStagnation              = "PressureStagnation";
    const char* TemperatureStagnation           = "TemperatureStagnation";
    const char* EnergyStagnation                = "EnergyStagnation";
    const char* EnthalpyStagnation              = "EnthalpyStagnation";
    const char* EnergyStagnationDensity         = "EnergyStagnationDensity";
    const char* VelocityX                       = "VelocityX";
    const char* VelocityY                       = "VelocityY";
    const char* VelocityZ                       = "VelocityZ";
    const char* VelocityR                       = "VelocityR";
    const char* VelocityTheta                   = "VelocityTheta";
    const char* VelocityPhi                     = "VelocityPhi";
    const char* VelocityMagnitude               = "VelocityMagnitude";
    const char* VelocityNormal                  = "VelocityNormal";
    const char* VelocityTangential              = "VelocityTangential";
    const char* VelocitySound                   = "VelocitySound";
    const char* VelocitySoundStagnation         = "VelocitySoundStagnation";
    const char* MomentumX                       = "MomentumX";
    const char* MomentumY                       = "MomentumY";
    const char* MomentumZ                       = "MomentumZ";
    const char* MomentumMagnitude               = "MomentumMagnitude";
    const char* RotatingVelocityX               = "RotatingVelocityX";
    const char* RotatingVelocityY               = "RotatingVelocityY";
    const char* RotatingVelocityZ               = "RotatingVelocityZ";
    const char* RotatingMomentumX               = "RotatingMomentumX";
    const char* RotatingMomentumY               = "RotatingMomentumY";
    const char* RotatingMomentumZ               = "RotatingMomentumZ";
    const char* RotatingVelocityMagnitude       = "RotatingVelocityMagnitude";
    const char* RotatingPressureStagnation      = "RotatingPressureStagnation";
    const char* RotatingEnergyStagnation        = "RotatingEnergyStagnation";
    const char* RotatingEnergyStagnationDensity = "RotatingEnergyStagnationDensity";
    const char* RotatingEnthalpyStagnation      = "RotatingEnthalpyStagnation";
    const char* EnergyKinetic                   = "EnergyKinetic";
    const char* PressureDynamic                 = "PressureDynamic";
    const char* SoundIntensityDB                = "SoundIntensityDB";
    const char* SoundIntensity                  = "SoundIntensity";
    const char* VorticityX                      = "VorticityX";
    const char* VorticityY                      = "VorticityY";
    const char* VorticityZ                      = "VorticityZ";
    const char* VorticityMagnitude              = "VorticityMagnitude";
    const char* SkinFrictionX                   = "SkinFrictionX";
    const char* SkinFrictionY                   = "SkinFrictionY";
    const char* SkinFrictionZ                   = "SkinFrictionZ";
    const char* SkinFrictionMagnitude           = "SkinFrictionMagnitude";
    const char* VelocityAngleX                  = "VelocityAngleX";
    const char* VelocityAngleY                  = "VelocityAngleY";
    const char* VelocityAngleZ                  = "VelocityAngleZ";
    const char* VelocityUnitVectorX             = "VelocityUnitVectorX";
    const char* VelocityUnitVectorY             = "VelocityUnitVectorY";
    const char* VelocityUnitVectorZ             = "VelocityUnitVectorZ";
    const char* MassFlow                        = "MassFlow";
    const char* ViscosityKinematic              = "ViscosityKinematic";
    const char* ViscosityMolecular              = "ViscosityMolecular";
    const char* ViscosityEddyDynamic            = "ViscosityEddyDynamic";
    const char* ViscosityEddy                   = "ViscosityEddy";
    const char* ThermalConductivity             = "ThermalConductivity";
    const char* ThermalConductivityReference    = "ThermalConductivityReference";
    const char* SpecificHeatPressure            = "SpecificHeatPressure";
    const char* SpecificHeatVolume              = "SpecificHeatVolume";
    const char* ReynoldsStressXX                = "ReynoldsStressXX";
    const char* ReynoldsStressXY                = "ReynoldsStressXY";
    const char* ReynoldsStressXZ                = "ReynoldsStressXZ";
    const char* ReynoldsStressYY                = "ReynoldsStressYY";
    const char* ReynoldsStressYZ                = "ReynoldsStressYZ";
    const char* ReynoldsStressZZ                = "ReynoldsStressZZ";
    const char* LengthReference                 = "LengthReference";
    const char* MolecularWeight                 = "MolecularWeight";
    const char* MolecularWeight_p               = "MolecularWeight%s";
    const char* HeatOfFormation                 = "HeatOfFormation";
    const char* HeatOfFormation_p               = "HeatOfFormation%s";
    const char* FuelAirRatio                    = "FuelAirRatio";
    const char* ReferenceTemperatureHOF         = "ReferenceTemperatureHOF";
    const char* MassFraction                    = "MassFraction";
    const char* MassFraction_p                  = "MassFraction%s";
    const char* LaminarViscosity                = "LaminarViscosity";
    const char* LaminarViscosity_p              = "LaminarViscosity%s";
    const char* ThermalConductivity_p           = "ThermalConductivity%s";
    const char* EnthalpyEnergyRatio             = "EnthalpyEnergyRatio";
    const char* CompressibilityFactor           = "CompressibilityFactor";
    const char* VibrationalElectronEnergy       = "VibrationalElectronEnergy";
    const char* VibrationalElectronTemperature  = "VibrationalElectronTemperature";
    const char* SpeciesDensity                  = "SpeciesDensity";
    const char* SpeciesDensity_p                = "SpeciesDensity%s";
    const char* MoleFraction                    = "MoleFraction";
    const char* MoleFraction_p                  = "MoleFraction%s";
    const char* ElectricFieldX                  = "ElectricFieldX";
    const char* ElectricFieldY                  = "ElectricFieldY";
    const char* ElectricFieldZ                  = "ElectricFieldZ";
    const char* MagneticFieldX                  = "MagneticFieldX";
    const char* MagneticFieldY                  = "MagneticFieldY";
    const char* MagneticFieldZ                  = "MagneticFieldZ";
    const char* CurrentDensityX                 = "CurrentDensityX";
    const char* CurrentDensityY                 = "CurrentDensityY";
    const char* CurrentDensityZ                 = "CurrentDensityZ";
    const char* LorentzForceX                   = "LorentzForceX";
    const char* LorentzForceY                   = "LorentzForceY";
    const char* LorentzForceZ                   = "LorentzForceZ";
    const char* ElectricConductivity            = "ElectricConductivity";
    const char* JouleHeating                    = "JouleHeating";
  };
};

// Typical Turbulence Models
// --------------------------
namespace CGNS {
  namespace Name {
    const char* TurbulentDistance             = "TurbulentDistance";
    const char* TurbulentEnergyKinetic        = "TurbulentEnergyKinetic";
    const char* TurbulentDissipation          = "TurbulentDissipation";
    const char* TurbulentDissipationRate      = "TurbulentDissipationRate";
    const char* TurbulentBBReynolds           = "TurbulentBBReynolds";
    const char* TurbulentSANuTilde            = "TurbulentSANuTilde";
    const char* TurbulentDistanceIndex        = "TurbulentDistanceIndex";
    const char* TurbulentEnergyKineticDensity = "TurbulentEnergyKineticDensity";
    const char* TurbulentDissipationDensity   = "TurbulentDissipationDensity";
    const char* TurbulentSANuTildeDensity     = "TurbulentSANuTildeDensity";
  };
};

// Nondimensional Parameters
// --------------------------
namespace CGNS {
  namespace Name {
    const char* Mach                         = "Mach";
    const char* Mach_Velocity                = "Mach_Velocity";
    const char* Mach_VelocitySound           = "Mach_VelocitySound";
    const char* Reynolds                     = "Reynolds";
    const char* Reynolds_Velocity            = "Reynolds_Velocity";
    const char* Reynolds_Length              = "Reynolds_Length";
    const char* Reynolds_ViscosityKinematic  = "Reynolds_ViscosityKinematic";
    const char* Prandtl                      = "Prandtl";
    const char* Prandtl_ThermalConductivity  = "Prandtl_ThermalConductivity";
    const char* Prandtl_ViscosityMolecular   = "Prandtl_ViscosityMolecular";
    const char* Prandtl_SpecificHeatPressure = "Prandtl_SpecificHeatPressure";
    const char* PrandtlTurbulent             = "PrandtlTurbulent";
    const char* CoefPressure                 = "CoefPressure";
    const char* CoefSkinFrictionX            = "CoefSkinFrictionX";
    const char* CoefSkinFrictionY            = "CoefSkinFrictionY";
    const char* CoefSkinFrictionZ            = "CoefSkinFrictionZ";
    const char* Coef_PressureDynamic         = "Coef_PressureDynamic";
    const char* Coef_PressureReference       = "Coef_PressureReference";
  };
};

// Characteristics and Riemann invariant
// --------------------------------------
namespace CGNS {
  namespace Name {
    const char* Vorticity                   = "Vorticity";
    const char* Acoustic                    = "Acoustic";
    const char* RiemannInvariantPlus        = "RiemannInvariantPlus";
    const char* RiemannInvariantMinus       = "RiemannInvariantMinus";
    const char* CharacteristicEntropy       = "CharacteristicEntropy";
    const char* CharacteristicVorticity1    = "CharacteristicVorticity1";
    const char* CharacteristicVorticity2    = "CharacteristicVorticity2";
    const char* CharacteristicAcousticPlus  = "CharacteristicAcousticPlus";
    const char* CharacteristicAcousticMinus = "CharacteristicAcousticMinus";
  };
};

// Forces and Moments
// -------------------
namespace CGNS {
  namespace Name {
    const char* ForceX               = "ForceX";
    const char* ForceY               = "ForceY";
    const char* ForceZ               = "ForceZ";
    const char* ForceR               = "ForceR";
    const char* ForceTheta           = "ForceTheta";
    const char* ForcePhi             = "ForcePhi";
    const char* Lift                 = "Lift";
    const char* Drag                 = "Drag";
    const char* MomentX              = "MomentX";
    const char* MomentY              = "MomentY";
    const char* MomentZ              = "MomentZ";
    const char* MomentR              = "MomentR";
    const char* MomentTheta          = "MomentTheta";
    const char* MomentPhi            = "MomentPhi";
    const char* MomentXi             = "MomentXi";
    const char* MomentEta            = "MomentEta";
    const char* MomentZeta           = "MomentZeta";
    const char* Moment_CenterX       = "Moment_CenterX";
    const char* Moment_CenterY       = "Moment_CenterY";
    const char* Moment_CenterZ       = "Moment_CenterZ";
    const char* CoefLift             = "CoefLift";
    const char* CoefDrag             = "CoefDrag";
    const char* CoefMomentX          = "CoefMomentX";
    const char* CoefMomentY          = "CoefMomentY";
    const char* CoefMomentZ          = "CoefMomentZ";
    const char* CoefMomentR          = "CoefMomentR";
    const char* CoefMomentTheta      = "CoefMomentTheta";
    const char* CoefMomentPhi        = "CoefMomentPhi";
    const char* CoefMomentXi         = "CoefMomentXi";
    const char* CoefMomentEta        = "CoefMomentEta";
    const char* CoefMomentZeta       = "CoefMomentZeta";
    const char* Coef_Area            = "Coef_Area";
    const char* Coef_Length          = "Coef_Length";
  };
};

// Time dependent flow
// --------------------
namespace CGNS {
  namespace Name {
    const char* TimeValues                   = "TimeValues";
    const char* IterationValues              = "IterationValues";
    const char* NumberOfZones                = "NumberOfZones";
    const char* NumberOfFamilies             = "NumberOfFamilies";
    const char* NumberOfSteps                = "NumberOfSteps";
    const char* DataConversion               = "DataConversion";
    const char* ZonePointers                 = "ZonePointers";
    const char* FamilyPointers               = "FamilyPointers";
    const char* RigidGridMotionPointers      = "RigidGridMotionPointers";
    const char* ArbitraryGridMotionPointers  = "ArbitraryGridMotionPointers";
    const char* GridCoordinatesPointers      = "GridCoordinatesPointers";
    const char* FlowSolutionPointers         = "FlowSolutionPointers";
    const char* ZoneGridConnectivityPointers = "ZoneGridConnectivityPointers";
    const char* ZoneSubRegionPointers        = "ZoneSubRegionPointers";
    const char* OriginLocation               = "OriginLocation";
    const char* Rig_idRotationAngle           = "Rig_idRotationAngle";
    const char* Rig_idVelocity                = "Rig_idVelocity";
    const char* Rig_idRotationRate            = "Rig_idRotationRate";
    const char* GridVelocityX                = "GridVelocityX";
    const char* GridVelocityY                = "GridVelocityY";
    const char* GridVelocityZ                = "GridVelocityZ";
    const char* GridVelocityR                = "GridVelocityR";
    const char* GridVelocityTheta            = "GridVelocityTheta";
    const char* GridVelocityPhi              = "GridVelocityPhi";
    const char* GridVelocityXi               = "GridVelocityXi";
    const char* GridVelocityEta              = "GridVelocityEta";
    const char* GridVelocityZeta             = "GridVelocityZeta";
  };
};

// Miscellanous
// -------------
namespace CGNS {
  namespace Name {
    const char* CGNSLibraryVersion         = "CGNSLibraryVersion";
    const char* CellDimension              = "CellDimension";
    const char* IndexDimension             = "IndexDimension";
    const char* PhysicalDimension          = "PhysicalDimension";
    const char* VertexSize                 = "VertexSize";
    const char* CellSize                   = "CellSize";
    const char* VertexSizeBoundary         = "VertexSizeBoundary";
    const char* ElementsSize               = "ElementsSize";
    const char* ZoneDonorName              = "ZoneDonorName";
    const char* BCRegionName               = "BCRegionName";
    const char* GridConnectivityRegionName = "GridConnectivityRegionName";
    const char* SurfaceArea                = "SurfaceArea";
    const char* RegionName                 = "RegionName";
    const char* Axisymmetry                = "Axisymmetry";
    const char* AxisymmetryReferencePoint  = "AxisymmetryReferencePoint";
    const char* AxisymmetryAxisVector      = "AxisymmetryAxisVector";
    const char* AxisymmetryAngle           = "AxisymmetryAngle";
    const char* ZoneConvergenceHistory     = "ZoneConvergenceHistory";
    const char* GlobalConvergenceHistory   = "GlobalConvergenceHistory";
    const char* NormDefinitions            = "NormDefinitions";
    const char* DimensionalExponents       = "DimensionalExponents";
    const char* DiscreteData               = "DiscreteData";
    const char* FamilyBC                   = "FamilyBC";
    const char* FamilyName                 = "FamilyName";
    const char* AdditionalFamilyName       = "AdditionalFamilyName";
    const char* Family                     = "Family";
    const char* FlowEquationSet            = "FlowEquationSet";
    const char* GasModel                   = "GasModel";
    const char* GeometryReference          = "GeometryReference";
    const char* Gravity                    = "Gravity";
    const char* GravityVector              = "GravityVector";
    const char* GridConnectivityProperty   = "GridConnectivityProperty";
    const char* InwardNormalList           = "InwardNormalList";
    const char* InwardNormalIndex          = "InwardNormalIndex";
    const char* Ordinal                    = "Ordinal";
    const char* Transform                  = "Transform";
    const char* OversetHoles               = "OversetHoles";
    const char* Periodic                   = "Periodic";
    const char* ReferenceState             = "ReferenceState";
    const char* RigidGridMotion            = "RigidGridMotion";
    const char* Rind                       = "Rind";
    const char* RotatingCoordinates        = "RotatingCoordinates";
    const char* RotationRateVector         = "RotationRateVector";
    const char* GoverningEquations         = "GoverningEquations";
    const char* BCTypeSimple               = "BCTypeSimple";
    const char* BCTypeCompound             = "BCTypeCompound";
    const char* ElementRangeList           = "ElementRangeList";
  };
};

//The strings defined below are type names used for node labels
//#############################################################

// Types as strings
// -----------------



