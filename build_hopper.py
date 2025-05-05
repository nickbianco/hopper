import opensim as osim
import numpy as np

def build_hopper(point_on_line_constraint=True, controller='prescribed'):

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
    hipRange = [110., -90.]
    hipStiff = [20., 20.] 
    hipDamping = 5. 
    hipTransition = 10.
    hipLimitForce = osim.CoordinateLimitForce('hipFlexion', hipRange[0],
        hipStiff[0], hipRange[1], hipStiff[1], hipDamping, hipTransition)
    hip.addComponent(hipLimitForce)

    kneeRange = [140., 10.]
    kneeStiff = [50., 40.] 
    kneeDamping = 2. 
    kneeTransition = 10.
    kneeLimitForce = osim.CoordinateLimitForce('kneeFlexion', kneeRange[0],
        kneeStiff[0], kneeRange[1], kneeStiff[1], kneeDamping, kneeTransition)
    knee.addComponent(kneeLimitForce)

    if point_on_line_constraint:
        # Create a constraint to keep the foot (distal end of the shank) directly
        # beneath the pelvis (the Y-axis points upwards).
        constraint = osim.PointOnLineConstraint(hopper.getGround(), osim.Vec3(0,1,0),
                            osim.Vec3(0), shank, linkDistalPoint)
        shank.addComponent(constraint)

    # Use a contact model to prevent the foot (ContactSphere) from passing
    # through the floor (ContactHalfSpace).
    floor = osim.ContactHalfSpace(osim.Vec3(0), osim.Vec3(0, 0, -np.pi/2.),
                                        hopper.getGround(), 'floor')
    footRadius = 0.1
    foot = osim.ContactSphere(footRadius, linkDistalPoint, shank, 'foot')

    contactForce = osim.HuntCrossleyForce()
    contactForce.setStiffness(1.e8)
    contactForce.setDissipation(0.5)
    contactForce.setStaticFriction(0.9)
    contactForce.setDynamicFriction(0.9)
    contactForce.setViscousFriction(0.6)
    contactForce.addGeometry('floor')
    contactForce.addGeometry('foot')

    # Add the contact-related components to the model.
    hopper.addContactGeometry(floor)
    hopper.addContactGeometry(foot)
    hopper.addForce(contactForce)

    # Create the vastus muscle and set its origin and insertion points.
    maxIsometricForce = 4000. 
    optimalFiberLength = 0.55 
    tendonSlacklength = 0.25
    pennationAngle = 0.
    vastus = osim.Millard2012EquilibriumMuscle('vastus', maxIsometricForce, 
                                               optimalFiberLength, tendonSlacklength, 
                                               pennationAngle)
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

    if controller == 'prescribed':
        # Create a controller to excite the vastus muscle.
        brain = osim.PrescribedController()
        brain.setActuators(hopper.updActuators())
        controlFunction = osim.PiecewiseLinearFunction()
        controlFunction.addPoint(0.0, 0.3)
        controlFunction.addPoint(2.0, 1.0)
        controlFunction.addPoint(3.9, 0.1)
        brain.prescribeControlForActuator('vastus', controlFunction)
        hopper.addController(brain)
    elif controller == 'constant':
        brain = osim.PrescribedController()
        brain.setActuators(hopper.updActuators())
        controlFunction = osim.Constant(0.0)
        brain.prescribeControlForActuator('vastus', controlFunction)
        hopper.addController(brain)
    else:
        raise ValueError("Controller must be either 'prescribed' or 'constant'.")

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

    hopper.finalizeConnections()

    return hopper
