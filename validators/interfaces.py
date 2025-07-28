# This file is kept for backward compatibility
# It re-exports the interfaces from their new modules

from validators.reporter_interface import ReporterInterface
from validators.validator_interface import ValidatorInterface

# The interfaces were moved to their own modules:
# - ValidatorInterface -> validators.validator_interface
# - ReporterInterface -> validators.reporter_interface
