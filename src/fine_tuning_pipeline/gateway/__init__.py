"""Optional gateway registration stage."""

from .config import GatewayConfig, GatewayConfigError, resolve_gateway_config
from .manager import GatewayManager

__all__ = ["GatewayConfig", "GatewayConfigError", "GatewayManager", "resolve_gateway_config"]
