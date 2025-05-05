import opensim as osim
from build_hopper import build_hopper


dt = 0.001

model = build_hopper(controller='constant')
state = model.initSystem()

def set_control_and_integrate(control):
    controller = model.updComponent('/controllerset/prescribedcontroller')
    constant = osim.Constant.safeDownCast(controller.upd_ControlFunctions().get(0))
    constant.setValue(control)

    manager = osim.Manager(model)
    manager.setIntegratorMinimumStepSize(dt)
    manager.setIntegratorMaximumStepSize(dt)
    manager.initialize(state)
    manager.integrate(state.getTime() + dt)

    print('control :', control)
    print('udot: ', manager.getState().getUDot())


set_control_and_integrate(0.0)
set_control_and_integrate(1.0)