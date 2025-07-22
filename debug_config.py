"""
Debug configuration module for controlling debug output across the application.
"""

# Global debug flag - can be set by command-line arguments or other configuration
DEBUG_ENABLED = False

def enable_debug():
    """Enable debug logging."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = True

def disable_debug():
    """Disable debug logging."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = False

def is_debug_enabled():
    """Check if debug logging is enabled."""
    return DEBUG_ENABLED

def debug_print(*args, **kwargs):
    """Print debug message only if debug is enabled."""
    if DEBUG_ENABLED:
        print(*args, **kwargs)
