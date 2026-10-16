"""Engine-owned numerical/visual tools behind the independent stage-two runner."""


def create(inputs, tools, question, mode, binding, output):
    if (mode == 'predictive' and inputs['stage_one_result']['data']['scene']['engine'] == 'rigid'
            and tools.reasoning.rigid_predictive_tracking_only):
        from atw.world_probing.engines.tracking import TrackingTools, NativeTracking
        backend = NativeTracking(inputs, tools, output/'native', tracking_only=True)
        return TrackingTools(backend, tools, question, mode, binding, output)
    from atw.world_probing.engines.native import NativeEvidence
    from atw.world_probing.engines.rigid import RigidTools
    backend = NativeEvidence(inputs, tools, output/'native')
    if backend.engine == 'rigid':implementation = RigidTools
    else:
        from atw.world_probing.engines.deformable import DeformableTools
        implementation = DeformableTools
    return implementation(backend, tools, question, mode, binding, output)
