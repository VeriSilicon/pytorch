# mypy: allow-untyped-defs
from ..common import DeviceOpOverrides, register_device_op_overrides

# mypy: allow-untyped-defs
from textwrap import dedent


class VSIDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def set_device(self, device_idx):
        return f"torch.vsi.set_device({device_idx})"

    def synchronize(self):
        return "pass"

    def device_guard(self, device_idx):
        return f"torch.vsi._DeviceGuard({device_idx})"


register_device_op_overrides("vsi", VSIDeviceOpOverrides())
