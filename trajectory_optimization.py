import opensim as osim
import numpy as np
from time import time

def build_hopper():

    # Create a new OpenSim model.
    hopper = osim.Model()
    hopper.setName('hopper')
    hopper.setGravity(osim.Vec3(0, -9.80665, 0))

    # Create the pelvis, thigh, and shank bodies.
    pelvisMass = 30.0 
    pelvisSideLength = 0.2
    # Inertia of a brick with side length 0.2 m and mass 30.0 kg.
    Ixx = pelvisMass * (1./12.) * (2. * pelvisSideLength * pelvisSideLength)
    pelvisInertia = osim.Inertia(Ixx, Ixx, Ixx)
    pelvis = osim.Body('pelvis', pelvisMass, osim.Vec3(0), pelvisInertia)

    linkMass = 10.0 
    linkLength = 0.5
    linkHalfLength = linkLength/2. 
    linkRadius = 0.035
    # Inertia of a cylinder with length 0.5 m, radius 0.035 m, and mass 10.0 kg.
    Ixx = linkMass*((linkRadius*linkRadius)/4. + (linkHalfLength*linkHalfLength)/3.)
    Iyy = linkMass*(linkRadius*linkRadius)/2.
    linkInertia = osim.Inertia(Ixx, Iyy, Ixx)
    thigh = osim.Body('thigh', linkMass, osim.Vec3(0), linkInertia)
    shank = osim.Body('shank', linkMass, osim.Vec3(0), linkInertia)

    # Add the bodies to the model (the model takes ownership).
    hopper.addBody(pelvis)
    hopper.addBody(thigh)
    hopper.addBody(shank)

    # Attach the pelvis to ground with a vertical slider joint, and attach the
    # pelvis, thigh, and shank bodies to each other with pin joints.
    sliderOrientation = osim.Vec3(0, 0, np.pi/2.)
    sliderToGround = osim.SliderJoint('slider', hopper.getGround(), osim.Vec3(0),
                        sliderOrientation, pelvis, osim.Vec3(0), sliderOrientation)
    linkDistalPoint = osim.Vec3(0, -linkHalfLength, 0)
    linkProximalPoint = osim.Vec3(0, linkHalfLength, 0)
    # Define the pelvis as the parent so the reported value is hip flexion.
    hip = osim.PinJoint('hip', pelvis, osim.Vec3(0), osim.Vec3(0), thigh,
                            linkProximalPoint, osim.Vec3(0))
    # Define the shank as the parent so the reported value is knee flexion.
    knee = osim.PinJoint('knee', shank, linkProximalPoint, osim.Vec3(0), thigh,
                                linkDistalPoint, osim.Vec3(0))

    # Add the joints to the model.
    hopper.addJoint(sliderToGround)
    hopper.addJoint(hip)
    hopper.addJoint(knee)

    # Set the coordinate names and default values.
    sliderCoord = sliderToGround.updCoordinate(osim.SliderJoint.Coord_TranslationX)
    sliderCoord.setName('yCoord')
    sliderCoord.setDefaultValue(1.)

    hipCoord = hip.updCoordinate(osim.PinJoint.Coord_RotationZ)
    hipCoord.setName('hipFlexion')
    hipCoord.setDefaultValue(0.35)

    kneeCoord = knee.updCoordinate(osim.PinJoint.Coord_RotationZ)
    kneeCoord.setName('kneeFlexion')
    kneeCoord.setDefaultValue(0.75)

    # Limit the range of motion for the hip and knee joints.
    # hipRange = [110., -90.]
    # hipStiff = [20., 20.] 
    # hipDamping = 5. 
    # hipTransition = 10.
    # hipLimitForce = osim.CoordinateLimitForce('hipFlexion', hipRange[0],
    #     hipStiff[0], hipRange[1], hipStiff[1], hipDamping, hipTransition)
    # hip.addComponent(hipLimitForce)

    # kneeRange = [140., 10.]
    # kneeStiff = [50., 40.] 
    # kneeDamping = 2. 
    # kneeTransition = 10.
    # kneeLimitForce = osim.CoordinateLimitForce('kneeFlexion', kneeRange[0],
    #     kneeStiff[0], kneeRange[1], kneeStiff[1], kneeDamping, kneeTransition)
    # knee.addComponent(kneeLimitForce)

    # Create a constraint to keep the foot (distal end of the shank) directly
    # beneath the pelvis (the Y-axis points upwards).
    # constraint = osim.PointOnLineConstraint(hopper.getGround(), osim.Vec3(0,1,0),
    #                     osim.Vec3(0), shank, linkDistalPoint)
    # shank.addComponent(constraint)

    # Use a contact model to prevent the foot (ContactSphere) from passing
    # through the floor (ContactHalfSpace).
    floor = osim.ContactHalfSpace(osim.Vec3(0), osim.Vec3(0, 0, -np.pi/2.),
                                        hopper.getGround(), 'floor')
    footRadius = 0.1
    foot = osim.ContactSphere(footRadius, linkDistalPoint, shank, 'foot')

    contactForce = osim.SmoothSphereHalfSpaceForce('contact_force', foot, floor)
    contactForce.set_stiffness(1.e7)
    contactForce.set_dissipation(2.0)
    contactForce.set_static_friction(0.8)
    contactForce.set_dynamic_friction(0.8)
    contactForce.set_viscous_friction(0.5)

    # Add the contact-related components to the model.
    hopper.addContactGeometry(floor)
    hopper.addContactGeometry(foot)
    hopper.addForce(contactForce)

    # Create the vastus muscle and set its origin and insertion points.
    maxIsometricForce = 4000. 
    optimalFiberLength = 0.55 
    tendonSlacklength = 0.25
    vastus = osim.DeGrooteFregly2016Muscle()
    vastus.setName('vastus')
    vastus.set_max_isometric_force(maxIsometricForce)
    vastus.set_optimal_fiber_length(optimalFiberLength)
    vastus.set_tendon_slack_length(tendonSlacklength)
    vastus.set_ignore_tendon_compliance(True)
    vastus.addNewPathPoint('origin', thigh, osim.Vec3(linkRadius, 0.1, 0))
    vastus.addNewPathPoint('insertion', shank, osim.Vec3(linkRadius, 0.15, 0))
    hopper.addForce(vastus)

    # Attach a cylinder (patella) to the distal end of the thigh over which the
    # vastus muscle can wrap. 
    patellaFrame = osim.PhysicalOffsetFrame('patellaFrame', thigh, 
                                            osim.Transform(linkDistalPoint))
    patella = osim.WrapCylinder()
    patella.setName('patella')
    patella.set_radius(0.08)
    patella.set_length(linkRadius*2.)
    patella.set_quadrant('x')

    patellaFrame.addWrapObject(patella)
    thigh.addComponent(patellaFrame)

    # Configure the vastus muscle to wrap over the patella.
    vastus.updGeometryPath().addPathWrap(patella)

    # Create frames on the thigh and shank segments for attaching the device.
    thighAttachment = osim.PhysicalOffsetFrame('deviceAttachmentPoint',
                            thigh, osim.Transform(osim.Vec3(linkRadius, 0.15, 0)))
    shankAttachment = osim.PhysicalOffsetFrame('deviceAttachmentPoint',
                            shank, osim.Transform(osim.Vec3(linkRadius, 0, 0)))
    thigh.addComponent(thighAttachment)
    shank.addComponent(shankAttachment)

    # Attach geometry to the bodies and enable the visualizer.
    pelvisGeometry = osim.Brick(osim.Vec3(pelvisSideLength/2.))
    pelvisGeometry.setColor(osim.Vec3(0.8, 0.1, 0.1))
    pelvis.attachGeometry(pelvisGeometry)

    linkGeometry = osim.Cylinder(linkRadius, linkLength/2.)
    linkGeometry.setColor(osim.Vec3(0.8, 0.1, 0.1))
    thigh.attachGeometry(linkGeometry)
    shank.attachGeometry(linkGeometry.clone())

    return hopper


hopper = build_hopper()
hopper.initSystem()
hopper.printToXML('hopper_moco.osim')

# Create MocoStudy.
# ================
study = osim.MocoStudy()
study.setName('max_height_jump')

# Define the optimal control problem.
# ===================================
problem = study.updProblem()

# Model (dynamics).
# -----------------
problem.setModel(hopper)

# Bounds.
# -------
# Initial time must be 0, final time can be within [0, 5].
problem.setTimeBounds(0., 5.0)

problem.setStateInfo('/jointset/slider/yCoord/value', [0, 5.0], [1.0])
problem.setStateInfo('/jointset/slider/yCoord/speed', [-10, 10], [0])

problem.setStateInfo('/jointset/hip/hipFlexion/value', 
                     [-90.*(np.pi/180.0), 110.*(np.pi/180.0)], [0.35])
problem.setStateInfo('/jointset/hip/hipFlexion/speed', [-10, 10], [0])

problem.setStateInfo('/jointset/knee/kneeFlexion/value', 
                     [10.*(np.pi/180.0), 140.*(np.pi/180.0)], [0.75])
problem.setStateInfo('/jointset/knee/kneeFlexion/speed', [-10, 10], [0])

problem.setControlInfo('/forceset/vastus', osim.MocoBounds(0, 1))

# Cost.
# -----
max_height_goal = osim.MocoFinalOutputGoal('max_height')
max_height_goal.setOutputPath('/bodyset/pelvis|position')
max_height_goal.setOutputIndex(1)
max_height_goal.setWeight(-1.0) # maximize the height
problem.addGoal(max_height_goal)

# Configure the solver.
# =====================
solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(100)
solver.set_kinematic_constraint_method('Bordalba2023')
solver.set_transcription_scheme('legendre-gauss-radau-2')
solver.set_optim_max_iterations(500)

# Solve the problem.
# ==================
solution = study.solve()
solutionUnsealed = solution.unseal()

solutionUnsealed.write('hopper_jump_solution.sto')

output = 'hopper_jump_solution.pdf'
report = osim.report.Report(hopper, 'hopper_jump_solution.sto',
                            output=output)
report.generate()

